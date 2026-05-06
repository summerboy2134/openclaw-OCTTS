from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.tools.backtest_fusion_top3_rotation import load_cache, save_cache, selection
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.compare_top3_score_fields import _load_local_stock_universe, _looks_unfillable_limit_up, _safe_float


@dataclass
class Position:
    ts_code: str
    name: str | None
    rank: int | None
    score_value: float | None
    signal_date: str
    entry_date: str
    entry_open: float
    shares: float
    cost: float


def main() -> None:
    p = argparse.ArgumentParser(description="Rolling max-3 portfolio backtest using daily Top3 refills")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--score-field", default="fusion_70_30")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--candidate-limit", type=int, default=200)
    p.add_argument("--exclude-bj", action="store_true")
    p.add_argument("--exclude-st", action="store_true")
    p.add_argument("--pool", choices=["stage2_top20", "stage1_candidate"], default="stage2_top20")
    p.add_argument("--require-fillable-entry", action="store_true")
    p.add_argument("--refill-unfillable", action="store_true")
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--initial-capital", type=float, default=1_000_000.0)
    p.add_argument("--loss-check-hold-days", type=int, default=3)
    p.add_argument("--max-hold-days", type=int, default=5)
    p.add_argument("--weak-profit-swap", action="store_true", help="When full, sell weak-profit held positions with hold_days >= loss_check_hold_days and return <= threshold to buy fillable unheld candidates from previous signal Top3")
    p.add_argument("--weak-profit-threshold", type=float, default=0.04)
    p.add_argument("--max-weak-profit-swaps-per-day", type=int, default=2)
    p.add_argument("--base-cache-dir", default="tmp/compare_top3_score_fields_cache")
    p.add_argument("--selection-cache-file", default="tmp/fusion_70_30_top3_selection_cache.json")
    p.add_argument("--force-refresh-selection-cache", action="store_true")
    p.add_argument("--force-refresh-base-cache", action="store_true")
    p.add_argument("--output-file", default="tmp/fusion_70_30_rolling_max3_portfolio.json")
    args = p.parse_args()

    settings = get_settings()
    log = configure_tool_logging(settings, "backtest_fusion_top3_rolling_portfolio")
    repo = MarketRawDataRepository(settings.database_url)
    engineer = ShortTermFeatureEngineer(settings)
    universe = _load_local_stock_universe(settings.database_url)
    cache = load_cache(args.selection_cache_file)

    start = datetime.strptime(args.start_date.replace("-", ""), "%Y%m%d").date()
    end = datetime.strptime(args.end_date.replace("-", ""), "%Y%m%d").date()
    warmup_start = start - timedelta(days=10)
    trade_dates = repo.list_trading_dates(start_date=warmup_start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    eval_dates = [x for x in trade_dates if start.strftime("%Y%m%d") <= x <= end.strftime("%Y%m%d")]
    prev_by_date = {trade_dates[i]: trade_dates[i - 1] for i in range(1, len(trade_dates))}

    cash = float(args.initial_capital)
    positions: list[Position] = []
    equity_curve = []
    trades = []
    daily = []

    for td in eval_dates:
        current_date = datetime.strptime(td, "%Y%m%d").date()
        prev_td = prev_by_date.get(td)
        log.info("rolling day start: trade_date=%s positions=%s cash=%.2f", td, len(positions), cash)
        codes = sorted({p.ts_code for p in positions})
        bars = repo.get_daily_by_trade_dates(ts_codes=codes, trading_dates=[td]) if codes else {}

        sells = []
        kept = []
        for pos in positions:
            bar = (bars.get(pos.ts_code) or {}).get(td) or {}
            open_price = _safe_float(bar.get("open"))
            hold_days = holding_days(repo, pos.entry_date, td)
            pnl = None if open_price is None else (open_price - pos.entry_open) / pos.entry_open
            reason = None
            if open_price is None:
                kept.append(pos)
                continue
            if hold_days > args.max_hold_days:
                reason = "max_hold_days"
            if reason:
                proceeds = pos.shares * open_price
                cash += proceeds
                trade = {"date": td, "action": "sell", "reason": reason, "ts_code": pos.ts_code, "name": pos.name, "entry_date": pos.entry_date, "entry_open": pos.entry_open, "sell_open": open_price, "hold_days": hold_days, "return": round(pnl, 6) if pnl is not None else None, "proceeds": round(proceeds, 2)}
                sells.append(trade)
                trades.append(trade)
            else:
                kept.append(pos)
        positions = kept

        buys = []
        pick_context = None
        if prev_td:
            signal_date = datetime.strptime(prev_td, "%Y%m%d").date()
            sel = selection(args, repo, engineer, universe, cache, signal_date, log)
            picks = list(sel.get("top_picks") or [])[: max(1, args.top_k)]
            pick_codes = [str(x.get("ts_code") or "").strip().upper() for x in picks if str(x.get("ts_code") or "").strip()]
            pick_bars = repo.get_daily_by_trade_dates(ts_codes=pick_codes, trading_dates=[td]) if pick_codes else {}
            pick_context = (picks, pick_bars)

        if args.weak_profit_swap and prev_td and pick_context:
            picks, pick_bars = pick_context
            current_bars = repo.get_daily_by_trade_dates(ts_codes=[p.ts_code for p in positions], trading_dates=[td]) if positions else {}
            weak_positions = []
            for pos in positions:
                bar = (current_bars.get(pos.ts_code) or {}).get(td) or {}
                open_price = _safe_float(bar.get("open"))
                hold_days = holding_days(repo, pos.entry_date, td)
                pnl = None if open_price is None else (open_price - pos.entry_open) / pos.entry_open
                if open_price is not None and pnl is not None and hold_days >= args.loss_check_hold_days and pnl <= args.weak_profit_threshold:
                    weak_positions.append((pnl, hold_days, pos, open_price))
            weak_positions = sorted(weak_positions, key=lambda x: (x[0], -x[1]))
            swap_limit = max(0, args.max_weak_profit_swaps_per_day)
            for pick in sorted(picks, key=lambda x: int(x.get("rank") or 999)):
                if not weak_positions or swap_limit <= 0:
                    break
                code = str(pick.get("ts_code") or "").strip().upper()
                if not code or any(p.ts_code == code for p in positions):
                    continue
                pick_bar = (pick_bars.get(code) or {}).get(td) or {}
                pick_open = _safe_float(pick_bar.get("open"))
                fillable = pick_open is not None and not _looks_unfillable_limit_up(entry_bar=pick_bar, ts_code=code, name=pick.get("name"), market=pick.get("market"))
                if not fillable:
                    continue
                pnl, hold_days, pos, open_price = weak_positions.pop(0)
                proceeds = pos.shares * open_price
                cash += proceeds
                positions = [p for p in positions if p.ts_code != pos.ts_code]
                sell_trade = {"date": td, "action": "sell", "reason": "weak_profit_swap", "ts_code": pos.ts_code, "name": pos.name, "entry_date": pos.entry_date, "entry_open": pos.entry_open, "sell_open": open_price, "hold_days": hold_days, "return": round(pnl, 6), "proceeds": round(proceeds, 2), "swap_to": code, "swap_to_rank": pick.get("rank")}
                sells.append(sell_trade)
                trades.append(sell_trade)

                budget = proceeds
                shares = budget / pick_open
                cost = shares * pick_open
                cash -= cost
                new_pos = Position(ts_code=code, name=pick.get("name"), rank=pick.get("rank"), score_value=_safe_float(pick.get("score_value")), signal_date=prev_td, entry_date=td, entry_open=pick_open, shares=shares, cost=cost)
                positions.append(new_pos)
                buy_trade = {"date": td, "action": "buy", "reason": "weak_profit_swap", "signal_date": prev_td, "ts_code": code, "name": pick.get("name"), "rank": pick.get("rank"), "entry_open": pick_open, "cost": round(cost, 2), "shares": shares, "swap_from": pos.ts_code}
                buys.append(buy_trade)
                trades.append(buy_trade)
                swap_limit -= 1

        if prev_td and len(positions) < args.max_positions and pick_context:
            picks, pick_bars = pick_context
            for pick in sorted(picks, key=lambda x: int(x.get("rank") or 999)):
                if len(positions) >= args.max_positions or cash <= 0:
                    break
                code = str(pick.get("ts_code") or "").strip().upper()
                if not code or any(p.ts_code == code for p in positions):
                    continue
                bar = (pick_bars.get(code) or {}).get(td) or {}
                open_price = _safe_float(bar.get("open"))
                if open_price is None:
                    continue
                fillable = not _looks_unfillable_limit_up(entry_bar=bar, ts_code=code, name=pick.get("name"), market=pick.get("market"))
                if not fillable:
                    continue
                slots = args.max_positions - len(positions)
                budget = cash / slots
                shares = budget / open_price
                cost = shares * open_price
                cash -= cost
                pos = Position(ts_code=code, name=pick.get("name"), rank=pick.get("rank"), score_value=_safe_float(pick.get("score_value")), signal_date=prev_td, entry_date=td, entry_open=open_price, shares=shares, cost=cost)
                positions.append(pos)
                trade = {"date": td, "action": "buy", "signal_date": prev_td, "ts_code": code, "name": pick.get("name"), "rank": pick.get("rank"), "entry_open": open_price, "cost": round(cost, 2), "shares": shares}
                buys.append(trade)
                trades.append(trade)

        market_value = mark_to_market(repo, positions, td)
        equity = cash + market_value
        equity_curve.append({"date": td, "cash": round(cash, 2), "market_value": round(market_value, 2), "equity": round(equity, 2), "positions": len(positions)})
        daily.append({"date": td, "sells": sells, "buys": buys, "cash": round(cash, 2), "market_value": round(market_value, 2), "equity": round(equity, 2), "positions": [asdict(p) for p in positions]})

    save_cache(args.selection_cache_file, cache)
    start_equity = float(args.initial_capital)
    end_equity = equity_curve[-1]["equity"] if equity_curve else start_equity
    payload = {
        "strategy": "rolling_max3_prev_signal_top3_refill",
        "params": vars(args),
        "summary": {
            "start_equity": round(start_equity, 2),
            "end_equity": end_equity,
            "total_return": round((end_equity - start_equity) / start_equity, 6) if start_equity else None,
            "trade_days": len(eval_dates),
            "buy_count": sum(1 for t in trades if t["action"] == "buy"),
            "sell_count": sum(1 for t in trades if t["action"] == "sell"),
            "final_positions": len(positions),
            "final_cash": round(cash, 2),
        },
        "equity_curve": equity_curve,
        "trades": trades,
        "daily": daily,
    }
    print_json(payload, output_file=args.output_file)


def holding_days(repo, entry_date: str, current_date: str) -> int:
    if entry_date == current_date:
        return 1
    days = repo.list_trading_dates(start_date=entry_date, end_date=current_date)
    return len(days)


def mark_to_market(repo, positions: list[Position], trade_date: str) -> float:
    if not positions:
        return 0.0
    bars = repo.get_daily_by_trade_dates(ts_codes=[p.ts_code for p in positions], trading_dates=[trade_date])
    total = 0.0
    for p in positions:
        bar = (bars.get(p.ts_code) or {}).get(trade_date) or {}
        close_price = _safe_float(bar.get("close")) or _safe_float(bar.get("open")) or p.entry_open
        total += p.shares * close_price
    return total


if __name__ == "__main__":
    main()

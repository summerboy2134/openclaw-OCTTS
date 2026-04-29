from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.indicators.technical import build_technical_snapshot
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.rebuild_large_rule_pool_compare import _load_daily_basic_rows, _load_daily_rows, safe_float


def inspect_trade_date(repo: MarketRawDataRepository, trade_date_text: str) -> Dict[str, Any]:
    trade_date = datetime.strptime(trade_date_text, "%Y-%m-%d").date()
    db_trade_date = trade_date.strftime("%Y%m%d")
    history_start = (trade_date - timedelta(days=90)).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=history_start, end_date=db_trade_date)
    daily_rows = _load_daily_rows(repo, trading_dates)
    basic_rows = _load_daily_basic_rows(repo, trading_dates)
    common = sorted(set(daily_rows.keys()) & set(basic_rows.keys()))

    counts = Counter()
    counts["common_codes"] = len(common)
    failure_examples: Dict[str, List[str]] = {
        "missing_target_day": [],
        "history_lt_30": [],
        "ohlcv_incomplete": [],
        "recommendation_lt_30": [],
        "technical_lt_35": [],
        "volume_ratio_lt_0_8": [],
        "turnover_rate_lt_0_8": [],
        "pct_chg_out_of_range": [],
        "price_position_too_high": [],
        "market_cap_too_high": [],
    }

    for ts_code in common:
        daily_map = daily_rows.get(ts_code, {})
        basic_map = basic_rows.get(ts_code, {})
        if db_trade_date not in daily_map or db_trade_date not in basic_map:
            _append_example(failure_examples, "missing_target_day", ts_code)
            continue
        counts["has_target_day"] += 1
        available_dates = [value for value in trading_dates if value in daily_map and value in basic_map]
        if len(available_dates) < 30:
            _append_example(failure_examples, "history_lt_30", ts_code)
            continue
        counts["history_ge_30"] += 1
        closes = [safe_float(daily_map[value].get("close")) for value in available_dates]
        highs = [safe_float(daily_map[value].get("high")) for value in available_dates]
        lows = [safe_float(daily_map[value].get("low")) for value in available_dates]
        volumes = [safe_float(daily_map[value].get("vol")) for value in available_dates]
        if any(value is None for value in closes[-30:]) or any(value is None for value in highs[-30:]) or any(value is None for value in lows[-30:]) or any(value is None for value in volumes[-30:]):
            _append_example(failure_examples, "ohlcv_incomplete", ts_code)
            continue
        counts["ohlcv_complete_30"] += 1

        snapshot = build_technical_snapshot(
            pd.Series(closes[-30:]),
            pd.Series(highs[-30:]),
            pd.Series(lows[-30:]),
            pd.Series(volumes[-30:]),
        )
        basic = basic_map[db_trade_date]
        pct_change = safe_float(daily_map[db_trade_date].get("pct_chg"))
        turnover_rate = safe_float(basic.get("turnover_rate"))
        volume_ratio = safe_float(basic.get("volume_ratio"))
        market_cap = safe_float(basic.get("total_mv"))
        market_cap_yi = (market_cap / 10000.0) if market_cap is not None else None

        if snapshot.recommendation_score < 30:
            _append_example(failure_examples, "recommendation_lt_30", ts_code)
            continue
        counts["recommendation_ge_30"] += 1
        if snapshot.technical_score < 35:
            _append_example(failure_examples, "technical_lt_35", ts_code)
            continue
        counts["technical_ge_35"] += 1
        if volume_ratio is None or volume_ratio < 0.8:
            _append_example(failure_examples, "volume_ratio_lt_0_8", ts_code)
            continue
        counts["volume_ratio_ge_0_8"] += 1
        if turnover_rate is None or turnover_rate < 0.8:
            _append_example(failure_examples, "turnover_rate_lt_0_8", ts_code)
            continue
        counts["turnover_rate_ge_0_8"] += 1
        if pct_change is None or pct_change < -5.0 or pct_change > 9.8:
            _append_example(failure_examples, "pct_chg_out_of_range", ts_code)
            continue
        counts["pct_chg_in_range"] += 1
        if snapshot.price_position_20d is not None and snapshot.price_position_20d > 0.995:
            _append_example(failure_examples, "price_position_too_high", ts_code)
            continue
        counts["price_position_ok"] += 1
        if market_cap_yi is not None and market_cap_yi > 800:
            _append_example(failure_examples, "market_cap_too_high", ts_code)
            continue
        counts["market_cap_ok"] += 1
        counts["final_candidates"] += 1

    return {
        "trade_date": trade_date_text,
        "trading_dates_count": len(trading_dates),
        "trading_dates_start": trading_dates[0] if trading_dates else None,
        "trading_dates_end": trading_dates[-1] if trading_dates else None,
        "counts": dict(counts),
        "failure_examples": failure_examples,
    }


def _append_example(failure_examples: Dict[str, List[str]], key: str, ts_code: str) -> None:
    examples = failure_examples[key]
    if len(examples) < 5:
        examples.append(str(ts_code))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect why rebuilt light rule pools are empty for given dates.")
    parser.add_argument("--trade-dates", nargs="+", required=True, help="Trade dates like 2026-03-03 2026-03-04")
    args = parser.parse_args()

    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    results = [inspect_trade_date(repo, value) for value in args.trade_dates]
    print_json({"results": results})


if __name__ == "__main__":
    main()

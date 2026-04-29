from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.indicators.technical import build_technical_snapshot
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model
from octts.tools.evaluate_rule_pool_with_raw_model import _load_candidate_pool_by_date, _pool_baseline_rates, NO_LLM_PIPELINE
from octts.services.screening_store import ScreeningStore


POOL_LIMITS = [100, 200, 300, 500]
COMPARE_DATES = ["2026-03-26", "2026-03-27", "2026-03-30"]
STRONG_THRESHOLD_CHOICES = [0.01, 0.02, 0.03]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild no-LLM large rule pools from raw market data and compare against saved historical pools.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", help="Regression target column")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"})
        return

    labeled = frame[frame[args.target].notna()].copy()
    if labeled.empty:
        print_json({"evaluated": False, "reason": "no_labeled_rows", "target": args.target})
        return

    labeled["trade_date"] = pd.to_datetime(labeled["trade_date"])
    labeled = labeled.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    feature_columns = [column for column in RAW_MARKET_FEATURE_COLUMNS if column in labeled.columns]
    features = labeled[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target = pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0)

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    x_train = features.iloc[:split_index]
    y_train = target.iloc[:split_index]
    x_test = features.iloc[split_index:].reset_index(drop=True)
    y_test = target.iloc[split_index:].reset_index(drop=True)
    meta_test = labeled.iloc[split_index:][["trade_date", "ts_code"]].reset_index(drop=True)

    model = _fit_model(args.model_type, x_train, y_train, is_regression=True)
    predictions = pd.Series(model.predict(x_test), name="model_score")
    scored = pd.concat([meta_test, predictions, y_test.rename(args.target)], axis=1)
    scored = scored[scored["trade_date"].dt.strftime("%Y-%m-%d").isin(COMPARE_DATES)].reset_index(drop=True)

    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    store = ScreeningStore(settings)

    trade_dates = [datetime.strptime(value, "%Y-%m-%d").date() for value in COMPARE_DATES]
    saved_pool_100 = _load_candidate_pool_by_date(store, trade_dates, 100)
    rebuilt_pools = _rebuild_candidate_pools(repo, trade_dates)

    saved_summary = _evaluate_candidate_pool_sizes(scored, saved_pool_100, [100], args.final_pick)
    rebuilt_summary = _evaluate_candidate_pool_sizes(scored, rebuilt_pools, POOL_LIMITS, args.final_pick)
    optimization_notes = _build_optimization_notes(rebuilt_pools)

    print_json(
        {
            "evaluated": True,
            "pipeline": NO_LLM_PIPELINE,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "compare_dates": COMPARE_DATES,
            "scored_rows": int(len(scored)),
            "saved_historical_pool": saved_summary,
            "rebuilt_large_rule_pools": rebuilt_summary,
            "optimization_notes": optimization_notes,
        }
    )


def _rebuild_candidate_pools(repo: MarketRawDataRepository, trade_dates: List[Any], *, exclude_bj: bool = False) -> Dict[Any, List[str]]:
    result: Dict[Any, List[str]] = {}
    for trade_date in trade_dates:
        result[trade_date] = _rebuild_single_trade_date_pool(repo, trade_date, exclude_bj=exclude_bj)
    return result


def _rebuild_single_trade_date_pool(repo: MarketRawDataRepository, trade_date: Any, *, exclude_bj: bool = False) -> List[str]:
    trade_date_text = trade_date.strftime("%Y%m%d")
    history_start = (trade_date - timedelta(days=90)).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=history_start, end_date=trade_date_text)
    daily_rows = _load_daily_rows(repo, trading_dates)
    basic_rows = _load_daily_basic_rows(repo, trading_dates)
    candidates: List[tuple[str, float]] = []

    for ts_code in sorted(set(daily_rows.keys()) & set(basic_rows.keys())):
        if exclude_bj and str(ts_code).endswith(".BJ"):
            continue
        daily_map = daily_rows.get(ts_code, {})
        basic_map = basic_rows.get(ts_code, {})
        if trade_date_text not in daily_map or trade_date_text not in basic_map:
            continue
        available_dates = [value for value in trading_dates if value in daily_map and value in basic_map]
        if len(available_dates) < 30:
            continue
        closes = [safe_float(daily_map[value].get("close")) for value in available_dates]
        highs = [safe_float(daily_map[value].get("high")) for value in available_dates]
        lows = [safe_float(daily_map[value].get("low")) for value in available_dates]
        volumes = [safe_float(daily_map[value].get("vol")) for value in available_dates]
        if any(value is None for value in closes[-30:]) or any(value is None for value in highs[-30:]) or any(value is None for value in lows[-30:]) or any(value is None for value in volumes[-30:]):
            continue

        snapshot = build_technical_snapshot(
            pd.Series(closes[-30:]),
            pd.Series(highs[-30:]),
            pd.Series(lows[-30:]),
            pd.Series(volumes[-30:]),
        )
        basic = basic_map[trade_date_text]
        pct_change = safe_float(daily_map[trade_date_text].get("pct_chg"))
        turnover_rate = safe_float(basic.get("turnover_rate"))
        volume_ratio = safe_float(basic.get("volume_ratio"))
        market_cap = safe_float(basic.get("total_mv"))
        market_cap_yi = (market_cap / 10000.0) if market_cap is not None else None

        if snapshot.recommendation_score < 30:
            continue
        if snapshot.technical_score < 35:
            continue
        if volume_ratio is None or volume_ratio < 0.8:
            continue
        if turnover_rate is None or turnover_rate < 0.8:
            continue
        if pct_change is None or pct_change < -5.0 or pct_change > 9.8:
            continue
        if snapshot.price_position_20d is not None and snapshot.price_position_20d > 0.995:
            continue
        if market_cap_yi is not None and market_cap_yi > 800:
            continue

        score = float(snapshot.recommendation_score)
        score += float(snapshot.technical_score) * 0.35
        score += float(volume_ratio or 0.0) * 3.0
        score += float(turnover_rate or 0.0) * 0.6
        score += (float(snapshot.price_position_20d) - 0.45) * 6.0 if snapshot.price_position_20d is not None else 0.0
        if snapshot.trend_status == "bullish":
            score += 5.0
        elif snapshot.trend_status == "improving":
            score += 3.0
        if snapshot.momentum_status in {"bullish", "bullish_rising", "strong"}:
            score += 3.5
        candidates.append((str(ts_code).strip(), score))

    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    return [ts_code for ts_code, _ in ordered]


def _load_daily_rows(repo: MarketRawDataRepository, trading_dates: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not trading_dates:
        return {}
    start_date = trading_dates[0]
    end_date = trading_dates[-1]
    session = repo._db.get_session()
    try:
        from octts.models.screening_models import MarketDaily

        rows = (
            session.query(MarketDaily)
            .filter(
                MarketDaily.trade_date >= repo._parse_date(start_date),
                MarketDaily.trade_date <= repo._parse_date(end_date),
            )
            .all()
        )
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            trade_key = row.trade_date.strftime("%Y%m%d")
            result.setdefault(row.ts_code, {})[trade_key] = repo._serialize_market_daily(row)
        return result
    finally:
        session.close()


def _load_daily_basic_rows(repo: MarketRawDataRepository, trading_dates: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not trading_dates:
        return {}
    start_date = trading_dates[0]
    end_date = trading_dates[-1]
    session = repo._db.get_session()
    try:
        from octts.models.screening_models import MarketDailyBasic

        rows = (
            session.query(MarketDailyBasic)
            .filter(
                MarketDailyBasic.trade_date >= repo._parse_date(start_date),
                MarketDailyBasic.trade_date <= repo._parse_date(end_date),
            )
            .all()
        )
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            trade_key = row.trade_date.strftime("%Y%m%d")
            result.setdefault(row.ts_code, {})[trade_key] = repo._serialize_market_daily_basic(row)
        return result
    finally:
        session.close()


def _evaluate_candidate_pool_sizes(scored: pd.DataFrame, candidate_pool_by_date: Dict[Any, List[str]], pool_limits: List[int], final_pick: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for pool_limit in pool_limits:
        limited_pool = {trade_date: codes[:pool_limit] for trade_date, codes in candidate_pool_by_date.items()}
        daily_results: List[Dict[str, Any]] = []
        overlap_debug: List[Dict[str, Any]] = []
        for trade_day, day_frame in scored.groupby("trade_date", sort=True):
            trade_date = trade_day.date()
            candidate_codes = limited_pool.get(trade_date)
            if not candidate_codes:
                overlap_debug.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "pool_count": 0,
                        "day_rows": int(len(day_frame)),
                        "day_codes": int(day_frame["ts_code"].nunique()),
                        "overlap_count": 0,
                    }
                )
                continue
            day_codes = set(day_frame["ts_code"].astype(str))
            overlap_codes = [code for code in candidate_codes if str(code) in day_codes]
            overlap_debug.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "pool_count": int(len(candidate_codes)),
                    "day_rows": int(len(day_frame)),
                    "day_codes": int(len(day_codes)),
                    "overlap_count": int(len(overlap_codes)),
                    "sample_overlap_codes": overlap_codes[:10],
                }
            )
            if not overlap_codes:
                continue
            selected = day_frame[day_frame["ts_code"].astype(str).isin(overlap_codes)].nlargest(
                min(final_pick, len(overlap_codes)), columns="model_score"
            )
            if selected.empty:
                continue
            daily_results.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "pool_count": int(len(candidate_codes)),
                    "overlap_count": int(len(overlap_codes)),
                    "avg_target_return": float(selected["vs_market_1d"].mean()),
                    "hit_rates": {
                        _threshold_key(threshold): float((selected["vs_market_1d"] >= threshold).mean())
                        for threshold in STRONG_THRESHOLD_CHOICES
                    },
                    "picked_codes": selected["ts_code"].tolist(),
                }
            )
        baseline = _pool_baseline_rates(scored, limited_pool)
        results.append(
            {
                "pool_limit": pool_limit,
                "evaluated_days": int(len(daily_results)),
                "avg_target_return": float(sum(item["avg_target_return"] for item in daily_results) / len(daily_results)) if daily_results else 0.0,
                "hit_rates": {
                    key: float(sum(item["hit_rates"][key] for item in daily_results) / len(daily_results)) if daily_results else 0.0
                    for key in [_threshold_key(threshold) for threshold in STRONG_THRESHOLD_CHOICES]
                },
                "pool_baseline": baseline,
                "daily_results": daily_results,
                "overlap_debug": overlap_debug,
            }
        )
    return {"results": results}


def _build_optimization_notes(rebuilt_pools: Dict[Any, List[str]]) -> List[str]:
    pool_sizes = [len(values) for values in rebuilt_pools.values() if values]
    notes: List[str] = []
    if pool_sizes:
        notes.append(f"重建后的纯规则候选池规模明显大于历史保存池，当前3日规模分别为 {pool_sizes}。")
    notes.append("现有预设规则偏重强条件交集，直接用于历史落盘时会过早收口，建议拆成粗筛规则与终筛规则两层。")
    notes.append("当前 volume_breakout / golden_cross 对量比、MACD、价格位置同时要求较高，适合终筛，不适合第一层大候选池。")
    notes.append("建议粗筛阶段放宽为 recommendation_score、technical_score、volume_ratio、turnover_rate、price_position 的软阈值组合，再交给模型精排。")
    return notes


def _threshold_key(threshold: float) -> str:
    return f">={int(threshold * 100)}%"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()

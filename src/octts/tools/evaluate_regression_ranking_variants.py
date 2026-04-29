from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.evaluate_light_rule_pool_long_range import _build_month_stats
from octts.tools.rebuild_large_rule_pool_compare import _evaluate_candidate_pool_sizes, _rebuild_single_trade_date_pool
from octts.tools.train_raw_market_model import _fit_model, resolve_feature_columns


SUPPORTED_TARGETS = ["vs_market_1d", "vs_market_3d", "vs_market_5d"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate regression-ranking variants with optional rule-score fusion.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", choices=SUPPORTED_TARGETS)
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--pool-limit", type=int, default=200)
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-trade-days", type=int, default=20)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--score-mode", default="model_only", choices=["model_only", "model_rule_blend"])
    parser.add_argument("--rule-weight", type=float, default=0.3)
    parser.add_argument("--feature-columns", default="", help="Comma-separated feature columns to use")
    parser.add_argument("--feature-file", default="", help="Path to newline-delimited feature list")
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
    labeled["ts_code"] = labeled["ts_code"].astype(str).str.strip()
    feature_columns = resolve_feature_columns(
        labeled,
        feature_columns_arg=args.feature_columns,
        feature_file_arg=args.feature_file,
    )
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
    scored["model_score"] = pd.to_numeric(scored["model_score"], errors="coerce").fillna(0.0)

    if args.start_date:
        scored = scored[scored["trade_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        scored = scored[scored["trade_date"] <= pd.Timestamp(args.end_date)]
    scored = scored.reset_index(drop=True)
    if scored.empty:
        print_json({"evaluated": False, "reason": "empty_scored_range"})
        return

    trade_dates = sorted({value.date() for value in scored["trade_date"]})
    if args.max_trade_days > 0:
        trade_dates = trade_dates[-args.max_trade_days:]
        scored = scored[scored["trade_date"].dt.date.isin(trade_dates)].reset_index(drop=True)

    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    rebuilt_pool_info = {
        trade_date: _rebuild_single_trade_date_pool_with_scores(repo, trade_date, exclude_bj=args.exclude_bj)
        for trade_date in trade_dates
    }
    candidate_pools = {trade_date: [item["ts_code"] for item in items] for trade_date, items in rebuilt_pool_info.items()}

    if args.score_mode == "model_rule_blend":
        scored = _apply_blended_scores(scored, rebuilt_pool_info, rule_weight=args.rule_weight)

    evaluation = _evaluate_candidate_pool_sizes_for_target(
        scored,
        candidate_pools,
        [args.pool_limit],
        args.final_pick,
        args.target,
    )
    result = evaluation["results"][0]
    month_stats = _build_month_stats_for_target(scored, candidate_pools, args.pool_limit, args.final_pick, args.target)

    print_json(
        {
            "evaluated": True,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "pool_limit": int(args.pool_limit),
            "final_pick": int(args.final_pick),
            "exclude_bj": bool(args.exclude_bj),
            "score_mode": args.score_mode,
            "rule_weight": float(args.rule_weight),
            "date_range": {
                "start": min(trade_dates).isoformat(),
                "end": max(trade_dates).isoformat(),
                "trade_days": int(len(trade_dates)),
            },
            "summary": {
                "evaluated_days": result["evaluated_days"],
                "avg_target_return": result["avg_target_return"],
                "hit_rates": result["hit_rates"],
                "pool_baseline": result["pool_baseline"],
            },
            "month_stats": month_stats,
            "sample_daily_results": result["daily_results"][:10],
        }
    )


def _rebuild_single_trade_date_pool_with_scores(
    repo: MarketRawDataRepository,
    trade_date: Any,
    *,
    exclude_bj: bool,
) -> List[Dict[str, Any]]:
    from octts.tools.rebuild_large_rule_pool_compare import _load_daily_basic_rows, _load_daily_rows, safe_float
    from octts.indicators.technical import build_technical_snapshot
    from datetime import timedelta

    trade_date_text = trade_date.strftime("%Y%m%d")
    history_start = (trade_date - timedelta(days=90)).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=history_start, end_date=trade_date_text)
    daily_rows = _load_daily_rows(repo, trading_dates)
    basic_rows = _load_daily_basic_rows(repo, trading_dates)
    candidates: List[Dict[str, Any]] = []

    for ts_code in sorted(set(daily_rows.keys()) & set(basic_rows.keys())):
        ts_code_text = str(ts_code).strip()
        if exclude_bj and ts_code_text.endswith(".BJ"):
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

        snapshot = build_technical_snapshot(pd.Series(closes[-30:]), pd.Series(highs[-30:]), pd.Series(lows[-30:]), pd.Series(volumes[-30:]))
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

        rule_score = float(snapshot.recommendation_score)
        rule_score += float(snapshot.technical_score) * 0.35
        rule_score += float(volume_ratio or 0.0) * 3.0
        rule_score += float(turnover_rate or 0.0) * 0.6
        rule_score += (float(snapshot.price_position_20d) - 0.45) * 6.0 if snapshot.price_position_20d is not None else 0.0
        if snapshot.trend_status == "bullish":
            rule_score += 5.0
        elif snapshot.trend_status == "improving":
            rule_score += 3.0
        if snapshot.momentum_status in {"bullish", "bullish_rising", "strong"}:
            rule_score += 3.5
        candidates.append({"ts_code": ts_code_text, "rule_score": float(rule_score)})

    return sorted(candidates, key=lambda item: item["rule_score"], reverse=True)


def _apply_blended_scores(scored: pd.DataFrame, rebuilt_pool_info: Dict[Any, List[Dict[str, Any]]], *, rule_weight: float) -> pd.DataFrame:
    rule_weight = max(0.0, min(1.0, float(rule_weight)))
    model_weight = 1.0 - rule_weight
    rows: List[Dict[str, Any]] = []
    for trade_date, items in rebuilt_pool_info.items():
        if not items:
            continue
        rule_scores = pd.Series([item["rule_score"] for item in items], dtype=float)
        rule_min = float(rule_scores.min())
        rule_max = float(rule_scores.max())
        scale = rule_max - rule_min
        for item in items:
            normalized = (item["rule_score"] - rule_min) / scale if scale > 0 else 0.0
            rows.append({"trade_date": pd.Timestamp(trade_date), "ts_code": item["ts_code"], "rule_score_norm": normalized})
    if not rows:
        return scored
    rule_frame = pd.DataFrame(rows)
    merged = scored.merge(rule_frame, on=["trade_date", "ts_code"], how="left")
    merged["rule_score_norm"] = merged["rule_score_norm"].fillna(0.0)

    blended_rows = []
    for trade_day, group in merged.groupby("trade_date", sort=False):
        model_scores = pd.to_numeric(group["model_score"], errors="coerce").fillna(0.0)
        model_min = float(model_scores.min())
        model_max = float(model_scores.max())
        scale = model_max - model_min
        normalized_model = (model_scores - model_min) / scale if scale > 0 else pd.Series([0.0] * len(group), index=group.index)
        group = group.copy()
        group["model_score_norm"] = normalized_model.values
        group["model_score"] = model_weight * group["model_score_norm"] + rule_weight * group["rule_score_norm"]
        blended_rows.append(group)
    return pd.concat(blended_rows, ignore_index=True)


def _evaluate_candidate_pool_sizes_for_target(
    scored: pd.DataFrame,
    candidate_pool_by_date: Dict[Any, List[str]],
    pool_limits: List[int],
    final_pick: int,
    target_column: str,
) -> Dict[str, Any]:
    adapted = scored.copy()
    adapted["vs_market_1d"] = adapted[target_column]
    return _evaluate_candidate_pool_sizes(adapted, candidate_pool_by_date, pool_limits, final_pick)


def _build_month_stats_for_target(
    scored: pd.DataFrame,
    rebuilt_pools: Dict[Any, List[str]],
    pool_limit: int,
    final_pick: int,
    target_column: str,
) -> List[Dict[str, Any]]:
    return _build_month_stats(scored, rebuilt_pools, pool_limit, final_pick, target_column)


if __name__ == "__main__":
    main()

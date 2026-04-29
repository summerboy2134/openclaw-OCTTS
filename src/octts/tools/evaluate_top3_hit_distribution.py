from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.evaluate_regression_ranking_variants import _apply_blended_scores, _rebuild_single_trade_date_pool_with_scores
from octts.tools.train_raw_market_model import _fit_model, resolve_feature_columns


def _distribution(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    return {str(count): int(freq) for count, freq in sorted(Counter(int(item[key]) for item in items).items())}


def _safe_mean(frame: pd.Series) -> float:
    return float(frame.mean()) if len(frame) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Top3 daily hit distribution for rule-pool plus rerank flow.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", choices=["vs_market_1d", "vs_market_3d", "vs_market_5d"])
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--pool-limit", type=int, default=200)
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-trade-days", type=int, default=20)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--score-mode", default="model_rule_blend", choices=["model_only", "model_rule_blend"])
    parser.add_argument("--rule-weight", type=float, default=0.3)
    parser.add_argument("--feature-columns", default="", help="Comma-separated feature columns to use")
    parser.add_argument("--feature-file", default="", help="Path to newline-delimited feature list")
    parser.add_argument("--output-daily-limit", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"})
        return

    target_to_absolute_column = {
        "vs_market_1d": "return_1d",
        "vs_market_3d": "return_3d",
        "vs_market_5d": "return_5d",
    }
    absolute_column = target_to_absolute_column[args.target]

    required_columns = ["trade_date", "ts_code", args.target, absolute_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        print_json({"evaluated": False, "reason": "missing_columns", "columns": missing_columns})
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
    meta_test = labeled.iloc[split_index:][["trade_date", "ts_code", absolute_column, args.target]].reset_index(drop=True)

    model = _fit_model(args.model_type, x_train, y_train, is_regression=True)
    predictions = pd.Series(model.predict(x_test), name="model_score")
    scored = pd.concat([meta_test, predictions], axis=1)
    scored["model_score"] = pd.to_numeric(scored["model_score"], errors="coerce").fillna(0.0)
    scored[absolute_column] = pd.to_numeric(scored[absolute_column], errors="coerce")
    scored[args.target] = pd.to_numeric(scored[args.target], errors="coerce")

    if args.start_date:
        scored = scored[scored["trade_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        scored = scored[scored["trade_date"] <= pd.Timestamp(args.end_date)]
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

    daily_results: List[Dict[str, Any]] = []
    for trade_day, day_frame in scored.groupby("trade_date", sort=True):
        trade_date = trade_day.date()
        candidate_codes = candidate_pools.get(trade_date, [])[: args.pool_limit]
        if not candidate_codes:
            continue
        day_codes = set(day_frame["ts_code"].astype(str))
        overlap_codes = [code for code in candidate_codes if str(code) in day_codes]
        if not overlap_codes:
            continue

        selected = day_frame[day_frame["ts_code"].astype(str).isin(overlap_codes)].nlargest(
            min(args.final_pick, len(overlap_codes)), columns="model_score"
        )
        if selected.empty:
            continue

        daily_results.append(
            {
                "trade_date": trade_date.isoformat(),
                "picked_codes": selected["ts_code"].astype(str).tolist(),
                "absolute_returns": [None if pd.isna(value) else float(value) for value in selected[absolute_column].tolist()],
                "relative_returns": [None if pd.isna(value) else float(value) for value in selected[args.target].tolist()],
                "avg_absolute_return": _safe_mean(selected[absolute_column]),
                "avg_relative_return": _safe_mean(selected[args.target]),
                "count_abs_ge_0": int((selected[absolute_column] >= 0.0).sum()),
                "count_abs_gt_0": int((selected[absolute_column] > 0.0).sum()),
                "count_rel_ge_0": int((selected[args.target] >= 0.0).sum()),
                "count_rel_gt_0": int((selected[args.target] > 0.0).sum()),
                "count_rel_ge_1pct": int((selected[args.target] >= 0.01).sum()),
                "count_rel_ge_2pct": int((selected[args.target] >= 0.02).sum()),
                "count_rel_ge_3pct": int((selected[args.target] >= 0.03).sum()),
            }
        )

    if not daily_results:
        print_json({"evaluated": False, "reason": "no_daily_results"})
        return

    total_picks = sum(len(item["picked_codes"]) for item in daily_results)
    summary = {
        "evaluated_days": int(len(daily_results)),
        "pool_limit": int(args.pool_limit),
        "final_pick": int(args.final_pick),
        "target": args.target,
        "absolute_target": absolute_column,
        "score_mode": args.score_mode,
        "rule_weight": float(args.rule_weight),
        "avg_absolute_return": float(sum(item["avg_absolute_return"] for item in daily_results) / len(daily_results)),
        "avg_relative_return": float(sum(item["avg_relative_return"] for item in daily_results) / len(daily_results)),
        "per_stock_hit_rates": {
            "abs_ge_0": float(sum(item["count_abs_ge_0"] for item in daily_results) / total_picks),
            "abs_gt_0": float(sum(item["count_abs_gt_0"] for item in daily_results) / total_picks),
            "rel_ge_0": float(sum(item["count_rel_ge_0"] for item in daily_results) / total_picks),
            "rel_gt_0": float(sum(item["count_rel_gt_0"] for item in daily_results) / total_picks),
            "rel_ge_1pct": float(sum(item["count_rel_ge_1pct"] for item in daily_results) / total_picks),
            "rel_ge_2pct": float(sum(item["count_rel_ge_2pct"] for item in daily_results) / total_picks),
            "rel_ge_3pct": float(sum(item["count_rel_ge_3pct"] for item in daily_results) / total_picks),
        },
        "daily_hit_distribution": {
            "abs_ge_0": _distribution(daily_results, "count_abs_ge_0"),
            "abs_gt_0": _distribution(daily_results, "count_abs_gt_0"),
            "rel_ge_0": _distribution(daily_results, "count_rel_ge_0"),
            "rel_gt_0": _distribution(daily_results, "count_rel_gt_0"),
            "rel_ge_1pct": _distribution(daily_results, "count_rel_ge_1pct"),
            "rel_ge_2pct": _distribution(daily_results, "count_rel_ge_2pct"),
            "rel_ge_3pct": _distribution(daily_results, "count_rel_ge_3pct"),
        },
    }

    print_json(
        {
            "evaluated": True,
            "input": args.input,
            "model_type": args.model_type,
            "feature_count": int(len(feature_columns)),
            "date_range": {
                "start": min(trade_dates).isoformat() if trade_dates else None,
                "end": max(trade_dates).isoformat() if trade_dates else None,
                "trade_days": int(len(trade_dates)),
            },
            "summary": summary,
            "daily_results": daily_results[: args.output_daily_limit],
            "daily_result_count": int(len(daily_results)),
        }
    )


if __name__ == "__main__":
    main()

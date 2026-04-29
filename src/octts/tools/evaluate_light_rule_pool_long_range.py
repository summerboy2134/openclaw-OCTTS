from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.rebuild_large_rule_pool_compare import (
    NO_LLM_PIPELINE,
    _evaluate_candidate_pool_sizes,
    _rebuild_single_trade_date_pool,
)
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model
from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository


DEFAULT_POOL_LIMIT = 300
DEFAULT_FINAL_PICK = 3
STRONG_THRESHOLD_CHOICES = [0.01, 0.02, 0.03]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate light coarse rule pool plus model top3 on longer history.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", help="Regression target column")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--pool-limit", type=int, default=DEFAULT_POOL_LIMIT)
    parser.add_argument("--final-pick", type=int, default=DEFAULT_FINAL_PICK)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--start-date", default="", help="Optional inclusive start date, e.g. 2026-03-01")
    parser.add_argument("--end-date", default="", help="Optional inclusive end date, e.g. 2026-03-30")
    parser.add_argument("--max-trade-days", type=int, default=20)
    parser.add_argument("--exclude-bj", action="store_true", help="Exclude BJ stocks from rebuilt rule pool")
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
    rebuilt_pools = {
        trade_date: _rebuild_single_trade_date_pool(repo, trade_date, exclude_bj=args.exclude_bj)
        for trade_date in trade_dates
    }
    evaluation = _evaluate_candidate_pool_sizes(scored, rebuilt_pools, [args.pool_limit], args.final_pick)
    result = evaluation["results"][0]

    month_stats = _build_month_stats(scored, rebuilt_pools, args.pool_limit, args.final_pick, args.target)
    pool_size_debug = {
        trade_date.isoformat(): int(len(rebuilt_pools.get(trade_date, [])))
        for trade_date in trade_dates[:10]
    }

    print_json(
        {
            "evaluated": True,
            "pipeline": NO_LLM_PIPELINE,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "pool_limit": int(args.pool_limit),
            "final_pick": int(args.final_pick),
            "exclude_bj": bool(args.exclude_bj),
            "date_range": {
                "start": min(trade_dates).isoformat(),
                "end": max(trade_dates).isoformat(),
                "trade_days": int(len(trade_dates)),
            },
            "feature_count": int(len(feature_columns)),
            "summary": {
                "evaluated_days": result["evaluated_days"],
                "avg_target_return": result["avg_target_return"],
                "hit_rates": result["hit_rates"],
                "pool_baseline": result["pool_baseline"],
            },
            "pool_size_debug": pool_size_debug,
            "overlap_debug": result.get("overlap_debug", [])[:10],
            "month_stats": month_stats,
            "sample_daily_results": result["daily_results"][:10],
        }
    )


def _build_month_stats(
    scored: pd.DataFrame,
    rebuilt_pools: Dict[Any, List[str]],
    pool_limit: int,
    final_pick: int,
    target_column: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    limited_pool = {trade_date: codes[:pool_limit] for trade_date, codes in rebuilt_pools.items()}

    for trade_day, day_frame in scored.groupby("trade_date", sort=True):
        trade_date = trade_day.date()
        candidate_codes = limited_pool.get(trade_date)
        if not candidate_codes:
            continue
        overlap_codes = [code for code in candidate_codes if str(code).strip() in set(day_frame["ts_code"].astype(str).str.strip())]
        if not overlap_codes:
            continue
        selected = day_frame[day_frame["ts_code"].astype(str).isin(overlap_codes)].nlargest(
            min(final_pick, len(overlap_codes)), columns="model_score"
        )
        if selected.empty:
            continue
        rows.append(
            {
                "month": trade_day.strftime("%Y-%m"),
                "avg_target_return": float(selected[target_column].mean()),
                ">=1%": float((selected[target_column] >= 0.01).mean()),
                ">=2%": float((selected[target_column] >= 0.02).mean()),
                ">=3%": float((selected[target_column] >= 0.03).mean()),
            }
        )

    if not rows:
        return []

    month_frame = pd.DataFrame(rows)
    results: List[Dict[str, Any]] = []
    for month, group in month_frame.groupby("month", sort=True):
        results.append(
            {
                "month": month,
                "days": int(len(group)),
                "avg_target_return": float(group["avg_target_return"].mean()),
                "hit_rates": {
                    key: float(group[key].mean())
                    for key in [">=1%", ">=2%", ">=3%"]
                },
            }
        )
    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import timedelta
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.evaluate_rule_pool_with_raw_model import NO_LLM_PIPELINE, _pool_baseline_rates
from octts.tools.rebuild_large_rule_pool_compare import _evaluate_candidate_pool_sizes, _rebuild_candidate_pools
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate light coarse rule pool over full test-history range.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", help="Regression target column")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--pool-limit", type=int, default=300)
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--debug-days", type=int, default=20)
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

    test_trade_dates = sorted({value.date() for value in meta_test["trade_date"]})
    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    rebuilt_pools = _rebuild_candidate_pools(repo, test_trade_dates)
    evaluation = _evaluate_candidate_pool_sizes(scored, rebuilt_pools, [args.pool_limit], args.final_pick)
    result = evaluation["results"][0]

    baseline_all = {
        "days": int(scored["trade_date"].nunique()),
        "avg_target_return": float(scored[args.target].mean()) if len(scored) else 0.0,
        "hit_rates": {
            ">=1%": float((scored[args.target] >= 0.01).mean()) if len(scored) else 0.0,
            ">=2%": float((scored[args.target] >= 0.02).mean()) if len(scored) else 0.0,
            ">=3%": float((scored[args.target] >= 0.03).mean()) if len(scored) else 0.0,
        },
    }

    daily_results = result.get("daily_results", [])
    positive_days = sum(1 for item in daily_results if float(item.get("avg_target_return", 0.0)) > 0)
    negative_days = sum(1 for item in daily_results if float(item.get("avg_target_return", 0.0)) <= 0)

    print_json(
        {
            "evaluated": True,
            "pipeline": NO_LLM_PIPELINE,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "pool_limit": args.pool_limit,
            "final_pick": args.final_pick,
            "test_trade_days": int(len(test_trade_dates)),
            "baseline_all": baseline_all,
            "pool300_result": result,
            "stability_summary": {
                "evaluated_days": int(result.get("evaluated_days", 0)),
                "positive_days": int(positive_days),
                "negative_or_flat_days": int(negative_days),
                "positive_day_ratio": float(positive_days / len(daily_results)) if daily_results else 0.0,
            },
            "debug_overlap_sample": result.get("overlap_debug", [])[: args.debug_days],
            "daily_result_sample": daily_results[: args.debug_days],
        }
    )


if __name__ == "__main__":
    main()

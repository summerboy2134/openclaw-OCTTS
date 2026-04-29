from __future__ import annotations

import argparse
from typing import Any, Dict, List

from octts.tools.common import print_json
from octts.tools.evaluate_rule_pool_with_raw_model import NO_LLM_PIPELINE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate-pool size ablation for rule pool plus model top3 rerank.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", help="Regression target column")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    from octts.tools.evaluate_rule_pool_with_raw_model import _pool_baseline_rates, _load_candidate_pool_by_date
    from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model
    from octts.config import get_settings
    from octts.services.screening_store import ScreeningStore
    import pandas as pd

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

    settings = get_settings()
    store = ScreeningStore(settings)
    test_trade_dates = sorted({value.date() for value in meta_test["trade_date"]})

    results: List[Dict[str, Any]] = []
    for pool_limit in [100, 200, 300, 500]:
        candidate_pool_by_date = _load_candidate_pool_by_date(store, test_trade_dates, pool_limit)
        daily_results: List[Dict[str, Any]] = []
        for trade_day, day_frame in scored.groupby("trade_date", sort=True):
            trade_date = trade_day.date()
            candidate_codes = candidate_pool_by_date.get(trade_date)
            if not candidate_codes:
                continue
            selected = day_frame[day_frame["ts_code"].isin(candidate_codes)].nlargest(
                min(args.final_pick, len(candidate_codes)), columns="model_score"
            )
            if selected.empty:
                continue
            daily_results.append(
                {
                    "avg_target_return": float(selected[args.target].mean()),
                    ">=1%": float((selected[args.target] >= 0.01).mean()),
                    ">=2%": float((selected[args.target] >= 0.02).mean()),
                    ">=3%": float((selected[args.target] >= 0.03).mean()),
                }
            )
        if not daily_results:
            results.append({"pool_limit": pool_limit, "evaluated_days": 0})
            continue
        pool_baseline = _pool_baseline_rates(scored, candidate_pool_by_date)
        results.append(
            {
                "pool_limit": pool_limit,
                "evaluated_days": int(len(daily_results)),
                "final_pick": int(args.final_pick),
                "avg_target_return": float(sum(item["avg_target_return"] for item in daily_results) / len(daily_results)),
                "hit_rates": {
                    key: float(sum(item[key] for item in daily_results) / len(daily_results))
                    for key in [">=1%", ">=2%", ">=3%"]
                },
                "pool_baseline": pool_baseline,
            }
        )

    print_json(
        {
            "evaluated": True,
            "pipeline": NO_LLM_PIPELINE,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "feature_count": int(len(feature_columns)),
            "results": results,
        }
    )


if __name__ == "__main__":
    main()

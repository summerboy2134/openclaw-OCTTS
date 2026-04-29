from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.services.screening_store import ScreeningStore
from octts.tools.common import print_json
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model


DEFAULT_POOL_LIMIT = 100
DEFAULT_FINAL_PICK = 3
STRONG_THRESHOLD_CHOICES = [0.01, 0.02, 0.03]
NO_LLM_PIPELINE = {
    "candidate_pool": "rule_based_screening_only",
    "model_rerank": "vs_market_1d_regression",
    "llm_final_stage": "final_top10_review_only",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-top100 plus model-rerank top3 historical hit rates.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", help="Regression target column")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--pool-limit", type=int, default=DEFAULT_POOL_LIMIT)
    parser.add_argument("--final-pick", type=int, default=DEFAULT_FINAL_PICK)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-daily-limit", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"})
        return

    required_columns = [args.target, "trade_date", "ts_code"]
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
    candidate_pool_by_date = _load_candidate_pool_by_date(store, test_trade_dates, args.pool_limit)

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
                "trade_date": trade_date.isoformat(),
                "pool_count": int(len(candidate_codes)),
                "selected_count": int(len(selected)),
                "avg_target_return": float(selected[args.target].mean()),
                "hits": {
                    _threshold_key(threshold): float((selected[args.target] >= threshold).mean())
                    for threshold in STRONG_THRESHOLD_CHOICES
                },
                "picked_codes": selected["ts_code"].tolist(),
            }
        )

    if not daily_results:
        print_json(
            {
                "evaluated": False,
                "reason": "no_overlap_between_test_dates_and_rule_pool",
                "test_trade_dates": [value.isoformat() for value in test_trade_dates],
            }
        )
        return

    summary = {
        "days": int(len(daily_results)),
        "pool_limit": int(args.pool_limit),
        "final_pick": int(args.final_pick),
        "avg_target_return": float(sum(item["avg_target_return"] for item in daily_results) / len(daily_results)),
        "hit_rates": {
            key: float(sum(item["hits"][key] for item in daily_results) / len(daily_results))
            for key in [_threshold_key(threshold) for threshold in STRONG_THRESHOLD_CHOICES]
        },
    }

    baseline_pool_hit_rates = _pool_baseline_rates(scored, candidate_pool_by_date)

    print_json(
        {
            "evaluated": True,
            "pipeline": NO_LLM_PIPELINE,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "feature_count": int(len(feature_columns)),
            "summary": summary,
            "pool_baseline": baseline_pool_hit_rates,
            "daily_results": daily_results[:args.output_daily_limit],
            "daily_result_count": int(len(daily_results)),
        }
    )


def _load_candidate_pool_by_date(store: ScreeningStore, trade_dates: List[Any], pool_limit: int) -> Dict[Any, List[str]]:
    result: Dict[Any, List[str]] = {}
    for trade_date in trade_dates:
        states = store.load_recommendation_pool_state(trade_date=trade_date)
        if not states:
            continue
        ordered = sorted(
            [item for item in states if item.get("ts_code")],
            key=lambda item: (
                item.get("recommend_rank") is None,
                int(item.get("recommend_rank") or 9999),
                -float(item.get("recommendation_score") or 0.0),
                item.get("ts_code") or "",
            ),
        )
        result[trade_date] = [str(item.get("ts_code")) for item in ordered[:pool_limit]]
    return result


def _pool_baseline_rates(scored: pd.DataFrame, candidate_pool_by_date: Dict[Any, List[str]]) -> Dict[str, Any]:
    daily_pool_returns: List[float] = []
    daily_pool_hits: Dict[str, List[float]] = defaultdict(list)

    for trade_day, day_frame in scored.groupby("trade_date", sort=True):
        trade_date = trade_day.date()
        candidate_codes = candidate_pool_by_date.get(trade_date)
        if not candidate_codes:
            continue
        pool_frame = day_frame[day_frame["ts_code"].isin(candidate_codes)]
        if pool_frame.empty:
            continue
        daily_pool_returns.append(float(pool_frame["vs_market_1d"].mean()))
        for threshold in STRONG_THRESHOLD_CHOICES:
            daily_pool_hits[_threshold_key(threshold)].append(float((pool_frame["vs_market_1d"] >= threshold).mean()))

    return {
        "days": int(len(daily_pool_returns)),
        "avg_target_return": float(sum(daily_pool_returns) / len(daily_pool_returns)) if daily_pool_returns else 0.0,
        "hit_rates": {
            key: float(sum(values) / len(values)) if values else 0.0
            for key, values in daily_pool_hits.items()
        },
    }


def _threshold_key(threshold: float) -> str:
    return f">={int(threshold * 100)}%"


if __name__ == "__main__":
    main()

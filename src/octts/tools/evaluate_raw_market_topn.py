from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model


TOP_N_CHOICES = [10, 20, 50]
STRONG_THRESHOLD_CHOICES = [0.01, 0.02, 0.03]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate daily Top-N performance for raw-market models.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", help="Regression target column")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--test-size", type=float, default=0.2)
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
    predictions = pd.Series(model.predict(x_test), name="prediction")
    scored = pd.concat(
        [
            meta_test,
            predictions,
            y_test.rename(args.target),
        ],
        axis=1,
    )
    for threshold in STRONG_THRESHOLD_CHOICES:
        column_name = _strong_label_name(threshold)
        scored[column_name] = scored[args.target] >= threshold

    baseline = {
        "mean_target": float(y_test.mean()) if len(y_test) else 0.0,
        "test_days": int(scored["trade_date"].nunique()),
        "test_rows": int(len(scored)),
        "strong_rates": {
            _threshold_key(threshold): float(scored[_strong_label_name(threshold)].mean())
            for threshold in STRONG_THRESHOLD_CHOICES
        },
    }

    topn_metrics: Dict[str, Dict[str, Any]] = {}
    for top_n in TOP_N_CHOICES:
        daily = _evaluate_daily_topn(scored, target_column=args.target, top_n=top_n)
        topn_metrics[f"top_{top_n}"] = daily

    result = {
        "evaluated": True,
        "input": args.input,
        "target": args.target,
        "model_type": args.model_type,
        "feature_count": int(len(feature_columns)),
        "baseline": baseline,
        "topn_metrics": topn_metrics,
    }
    print_json(result)


def _evaluate_daily_topn(scored: pd.DataFrame, *, target_column: str, top_n: int) -> Dict[str, Any]:
    daily_target_means: List[float] = []
    picked_rows = 0
    daily_hit_rates: Dict[str, List[float]] = {
        _threshold_key(threshold): [] for threshold in STRONG_THRESHOLD_CHOICES
    }

    for _, day_frame in scored.groupby("trade_date", sort=True):
        selected = day_frame.nlargest(min(top_n, len(day_frame)), columns="prediction")
        if selected.empty:
            continue
        picked_rows += int(len(selected))
        daily_target_means.append(float(selected[target_column].mean()))
        for threshold in STRONG_THRESHOLD_CHOICES:
            daily_hit_rates[_threshold_key(threshold)].append(float(selected[_strong_label_name(threshold)].mean()))

    return {
        "days": int(len(daily_target_means)),
        "picked_rows": int(picked_rows),
        "avg_target_return": float(sum(daily_target_means) / len(daily_target_means)) if daily_target_means else 0.0,
        "strong_hit_rates": {
            key: float(sum(values) / len(values)) if values else 0.0
            for key, values in daily_hit_rates.items()
        },
    }


def _strong_label_name(threshold: float) -> str:
    return f"strong_ge_{int(threshold * 100)}pct"


def _threshold_key(threshold: float) -> str:
    return f">={int(threshold * 100)}%"


if __name__ == "__main__":
    main()

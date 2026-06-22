from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model, _impute_missing_features, resolve_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy backward feature selection optimized for daily Top-N raw-market ranking.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="return_3d", help="Model training target")
    parser.add_argument("--eval-target", default="return_1d", help="Metric target used to select features")
    parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "xgboost", "logistic"])
    parser.add_argument("--feature-columns", default="", help="Comma-separated initial feature columns")
    parser.add_argument("--feature-file", default="", help="Optional newline-delimited initial feature list")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--start-date", default="", help="Optional evaluation start date")
    parser.add_argument("--end-date", default="", help="Optional evaluation end date")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-features", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-improvement", type=float, default=0.0005)
    parser.add_argument("--score-metric", default="positive_rate", choices=["positive_rate", "mean_return", "combined"])
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--output-feature-file", default="data/raw_market_top100_selected_features.txt")
    parser.add_argument("--output-file", default="")
    parser.add_argument("--lightgbm-num-leaves", type=int, default=15)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=120)
    parser.add_argument("--lightgbm-reg-alpha", type=float, default=0.05)
    parser.add_argument("--lightgbm-reg-lambda", type=float, default=1.0)
    parser.add_argument("--lightgbm-verbosity", type=int, default=-1)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"selected": False, "reason": "empty_dataset"}, output_file=args.output_file or None)
        return
    required_columns = {"trade_date", "ts_code", args.target, args.eval_target}
    missing_required = sorted(column for column in required_columns if column not in frame.columns)
    if missing_required:
        print_json({"selected": False, "reason": "missing_columns", "columns": missing_required}, output_file=args.output_file or None)
        return

    if args.exclude_bj:
        frame = frame[~frame["ts_code"].astype(str).str.upper().str.endswith(".BJ")].copy()

    labeled = frame[frame[args.target].notna() & frame[args.eval_target].notna()].copy()
    if labeled.empty:
        print_json({"selected": False, "reason": "no_labeled_rows"}, output_file=args.output_file or None)
        return

    labeled["trade_date"] = pd.to_datetime(labeled["trade_date"])
    labeled["ts_code"] = labeled["ts_code"].astype(str).str.strip().str.upper()
    labeled = labeled.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    initial_columns = resolve_feature_columns(
        labeled,
        feature_columns_arg=args.feature_columns,
        feature_file_arg=args.feature_file,
    )
    if not initial_columns:
        initial_columns = [column for column in RAW_MARKET_FEATURE_COLUMNS if column in labeled.columns]

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    y_train = pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0).iloc[:split_index]
    eval_frame = labeled.iloc[split_index:][["trade_date", "ts_code", args.eval_target]].reset_index(drop=True)
    if args.start_date:
        eval_frame = eval_frame[eval_frame["trade_date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        eval_frame = eval_frame[eval_frame["trade_date"] <= pd.Timestamp(args.end_date)].copy()

    model_kwargs = _build_model_kwargs(args)
    history: List[Dict[str, Any]] = []
    current_columns = list(initial_columns)
    current_result = _evaluate_columns(
        labeled,
        feature_columns=current_columns,
        split_index=split_index,
        y_train=y_train,
        eval_frame=eval_frame,
        train_target=args.target,
        eval_target=args.eval_target,
        model_type=args.model_type,
        model_kwargs=model_kwargs,
        top_n=args.top_n,
        score_metric=args.score_metric,
    )
    history.append({"step": 0, "action": "baseline", "removed_feature": None, "result": current_result})

    step = 0
    while len(current_columns) > args.min_features and step < args.max_steps:
        step += 1
        trials: List[Dict[str, Any]] = []
        for feature in current_columns:
            trial_columns = [column for column in current_columns if column != feature]
            trial_result = _evaluate_columns(
                labeled,
                feature_columns=trial_columns,
                split_index=split_index,
                y_train=y_train,
                eval_frame=eval_frame,
                train_target=args.target,
                eval_target=args.eval_target,
                model_type=args.model_type,
                model_kwargs=model_kwargs,
                top_n=args.top_n,
                score_metric=args.score_metric,
            )
            trials.append({"removed_feature": feature, "result": trial_result})

        trials.sort(key=lambda item: item["result"]["selection_score"], reverse=True)
        best_trial = trials[0]
        improvement = best_trial["result"]["selection_score"] - current_result["selection_score"]
        history.append(
            {
                "step": step,
                "action": "remove_candidate",
                "removed_feature": best_trial["removed_feature"],
                "improvement": float(improvement),
                "result": best_trial["result"],
                "top_trials": trials[:5],
            }
        )
        if improvement < args.min_improvement:
            break
        current_columns = [column for column in current_columns if column != best_trial["removed_feature"]]
        current_result = best_trial["result"]

    output_feature_path = Path(args.output_feature_file)
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    output_feature_path.write_text("\n".join(current_columns) + "\n", encoding="utf-8")

    payload = {
        "selected": True,
        "input": args.input,
        "train_target": args.target,
        "eval_target": args.eval_target,
        "top_n": int(args.top_n),
        "score_metric": args.score_metric,
        "initial_feature_count": int(len(initial_columns)),
        "selected_feature_count": int(len(current_columns)),
        "selected_features": current_columns,
        "output_feature_file": str(output_feature_path),
        "baseline_result": history[0]["result"],
        "final_result": current_result,
        "history": history,
    }
    print_json(payload, output_file=args.output_file or None)


def _build_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    if args.model_type != "lightgbm":
        return {}
    return {
        "num_leaves": args.lightgbm_num_leaves,
        "min_child_samples": args.lightgbm_min_child_samples,
        "reg_alpha": args.lightgbm_reg_alpha,
        "reg_lambda": args.lightgbm_reg_lambda,
        "verbosity": args.lightgbm_verbosity,
    }


def _evaluate_columns(
    labeled: pd.DataFrame,
    *,
    feature_columns: List[str],
    split_index: int,
    y_train: pd.Series,
    eval_frame: pd.DataFrame,
    train_target: str,
    eval_target: str,
    model_type: str,
    model_kwargs: Dict[str, Any],
    top_n: int,
    score_metric: str,
) -> Dict[str, Any]:
    raw_features = labeled[feature_columns].apply(pd.to_numeric, errors="coerce")
    x_train_raw = raw_features.iloc[:split_index]
    x_test_raw = raw_features.iloc[split_index:].reset_index(drop=True)
    x_train, _ = _impute_missing_features(x_train_raw)
    x_test, _ = _impute_missing_features(x_test_raw, reference_frame=x_train_raw)
    model = _fit_model(model_type, x_train, y_train, is_regression=True, model_kwargs=model_kwargs)
    predictions = pd.Series(model.predict(x_test), name="prediction")
    scored = eval_frame.reset_index(drop=True).copy()
    scored["prediction"] = predictions.loc[scored.index].to_numpy()
    scored[eval_target] = pd.to_numeric(scored[eval_target], errors="coerce")
    daily_values: List[Dict[str, float]] = []
    for _, day_frame in scored.groupby("trade_date", sort=True):
        selected = day_frame.nlargest(min(top_n, len(day_frame)), columns="prediction")
        baseline_mean = float(day_frame[eval_target].mean()) if len(day_frame) else 0.0
        topn_mean = float(selected[eval_target].mean()) if len(selected) else 0.0
        baseline_positive_rate = float((day_frame[eval_target] > 0).mean()) if len(day_frame) else 0.0
        topn_positive_rate = float((selected[eval_target] > 0).mean()) if len(selected) else 0.0
        daily_values.append(
            {
                "baseline_mean": baseline_mean,
                "topn_mean": topn_mean,
                "excess_mean": topn_mean - baseline_mean,
                "baseline_positive_rate": baseline_positive_rate,
                "topn_positive_rate": topn_positive_rate,
                "excess_positive_rate": topn_positive_rate - baseline_positive_rate,
            }
        )
    summary = _average_daily_values(daily_values)
    if score_metric == "positive_rate":
        selection_score = summary["avg_topn_positive_rate"]
    elif score_metric == "mean_return":
        selection_score = summary["avg_topn_mean"]
    else:
        selection_score = summary["avg_topn_positive_rate"] + summary["avg_topn_mean"] * 5.0
    return {
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "evaluated_days": int(len(daily_values)),
        "selection_score": float(selection_score),
        **summary,
    }


def _average_daily_values(values: List[Dict[str, float]]) -> Dict[str, float]:
    if not values:
        return {
            "avg_baseline_mean": 0.0,
            "avg_topn_mean": 0.0,
            "avg_excess_mean": 0.0,
            "avg_baseline_positive_rate": 0.0,
            "avg_topn_positive_rate": 0.0,
            "avg_excess_positive_rate": 0.0,
        }
    keys = values[0].keys()
    return {f"avg_{key}": float(sum(item[key] for item in values) / len(values)) for key in keys}


if __name__ == "__main__":
    main()

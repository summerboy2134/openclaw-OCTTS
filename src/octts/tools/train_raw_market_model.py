from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from octts.config import get_settings
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.modeling import save_model_artifact


RAW_MARKET_FEATURE_COLUMNS = [
    "close",
    "pct_change",
    "turnover_rate",
    "volume_ratio",
    "market_cap",
    "pe_ttm",
    "pb",
    "amount",
    "vol",
    "volatility_5d",
    "volatility_10d",
    "max_drawdown_10d_past",
    "close_to_ma5",
    "close_to_ma10",
    "close_to_ma20",
    "price_position_20d",
    "price_position_10d",
    "avg_turnover_rate_5d",
    "avg_volume_ratio_5d",
    "market_return_1d",
    "market_return_3d",
    "market_return_5d",
    "market_up_ratio_1d",
    "market_up_ratio_3d_avg",
    "market_up_days_5d",
    "stock_vs_market_return_1d",
    "stock_vs_market_return_2d",
    "stock_vs_market_return_3d",
    "stock_vs_market_return_5d",
    "stock_vs_market_return_10d",
    "pct_change_rank_pct",
    "turnover_rate_rank_pct",
    "volume_ratio_rank_pct",
    "up_days_3d",
    "up_days_5d",
    "new_high_gap_20d",
    "new_high_gap_10d",
    "new_low_gap_20d",
    "amount_ratio_1d_5d",
    "amount_ratio_3d_10d",
    "turnover_rate_change_1d",
    "turnover_rate_change_5d",
]

CLASSIFICATION_TARGETS = [
    "label_up_1d",
    "label_up_3d",
    "label_up_5d",
    "label_vs_market_1d",
    "label_vs_market_3d",
    "label_vs_market_5d",
    "label_strong_1d",
]

REGRESSION_TARGETS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "vs_market_1d",
    "vs_market_3d",
    "vs_market_5d",
]

ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS


def resolve_feature_columns(frame: pd.DataFrame, *, feature_columns_arg: str = "", feature_file_arg: str = "") -> list[str]:
    requested_columns: list[str] = []
    if feature_columns_arg:
        requested_columns.extend([column.strip() for column in feature_columns_arg.split(",") if column.strip()])
    if feature_file_arg:
        feature_file_path = Path(feature_file_arg)
        requested_columns.extend(
            [line.strip() for line in feature_file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    if requested_columns:
        deduped_columns: list[str] = []
        seen = set()
        for column in requested_columns:
            if column in seen:
                continue
            seen.add(column)
            deduped_columns.append(column)
        return [column for column in deduped_columns if column in frame.columns]
    return [column for column in RAW_MARKET_FEATURE_COLUMNS if column in frame.columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline model from raw-market CSV dataset.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--target", default="label_up_1d", choices=ALL_TARGETS)
    parser.add_argument("--output-name", default="raw_market_latest")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--feature-columns", default="", help="Comma-separated feature columns to use")
    parser.add_argument("--feature-file", default="", help="Path to newline-delimited feature list")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "train_raw_market_model")
    frame = pd.read_csv(args.input)
    if frame.empty:
        result = {"trained": False, "reason": "empty_dataset"}
        logger.warning("Training skipped: %s", result)
        print_json(result)
        return

    labeled = frame[frame[args.target].notna()].copy()
    if labeled.empty:
        result = {"trained": False, "reason": "no_labeled_rows", "target": args.target}
        logger.warning("Training skipped: %s", result)
        print_json(result)
        return

    labeled["trade_date"] = pd.to_datetime(labeled["trade_date"])
    labeled = labeled.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    feature_columns = resolve_feature_columns(
        labeled,
        feature_columns_arg=args.feature_columns,
        feature_file_arg=args.feature_file,
    )
    features = labeled[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    is_regression = args.target in REGRESSION_TARGETS
    labels = (
        pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0)
        if is_regression
        else labeled[args.target].astype(bool).astype(int)
    )

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    x_train = features.iloc[:split_index]
    y_train = labels.iloc[:split_index]
    x_test = features.iloc[split_index:]
    y_test = labels.iloc[split_index:]

    model = _fit_model(args.model_type, x_train, y_train, is_regression=is_regression)
    metrics = _evaluate_model(model, x_test, y_test, is_regression=is_regression)
    ranking_metrics = _evaluate_ranking(model, x_test, y_test, is_regression=is_regression)

    model_dir = Path(settings.history_dir_path) / "short_term_models"
    artifact_path = model_dir / f"{args.output_name}.{args.model_type}.pkl"
    save_model_artifact(
        artifact_path,
        {
            "model_type": args.model_type,
            "target": args.target,
            "task_type": "regression" if is_regression else "classification",
            "dataset_path": args.input,
            "feature_columns": feature_columns,
            "metrics": metrics,
            "ranking_metrics": ranking_metrics,
            "model": model,
        },
    )

    result = {
        "trained": True,
        "model_type": args.model_type,
        "target": args.target,
        "task_type": "regression" if is_regression else "classification",
        "dataset_path": args.input,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_columns)),
        "metrics": metrics,
        "ranking_metrics": ranking_metrics,
        "artifact_path": str(artifact_path),
    }
    logger.info("Raw-market model training complete: %s", result)
    print_json(result)


def _fit_model(model_type: str, features, labels, *, is_regression: bool):
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed. Install it before training this model.") from exc
        model_cls = lgb.LGBMRegressor if is_regression else lgb.LGBMClassifier
        model = model_cls(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(features, labels)
        return model
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed. Install it before training this model.") from exc
        model_cls = XGBRegressor if is_regression else XGBClassifier
        kwargs = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        if not is_regression:
            kwargs["eval_metric"] = "logloss"
        model = model_cls(**kwargs)
        model.fit(features, labels)
        return model
    try:
        from sklearn.linear_model import LogisticRegression, LinearRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is not installed. Install it before training baseline models.") from exc
    model = LinearRegression() if is_regression else LogisticRegression(max_iter=5000)
    model.fit(features, labels)
    return model


def _evaluate_model(model, x_test, y_test, *, is_regression: bool):
    if is_regression:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        predictions = model.predict(x_test)
        mse = float(mean_squared_error(y_test, predictions))
        return {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": mse ** 0.5,
            "r2": float(r2_score(y_test, predictions)),
        }

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    return metrics


def _evaluate_ranking(model, x_test, y_test, *, is_regression: bool):
    predictions = pd.Series(model.predict(x_test)).reset_index(drop=True)
    labels = pd.Series(y_test).reset_index(drop=True)
    ranking_metrics = {}

    if is_regression:
        ranking_metrics["baseline_mean_return"] = float(labels.mean()) if len(labels) else 0.0
        for pct in (0.01, 0.03, 0.05, 0.10):
            bucket_size = max(1, int(len(predictions) * pct))
            top_indices = predictions.nlargest(bucket_size).index
            top_labels = labels.loc[top_indices]
            ranking_metrics[f"top_{int(pct * 100)}pct_count"] = int(bucket_size)
            ranking_metrics[f"top_{int(pct * 100)}pct_mean_return"] = float(top_labels.mean()) if len(top_labels) else 0.0
            ranking_metrics[f"top_{int(pct * 100)}pct_excess_return"] = (
                float(top_labels.mean()) - float(labels.mean()) if len(top_labels) else 0.0
            )
        return ranking_metrics

    if not hasattr(model, "predict_proba"):
        return {}
    probabilities = pd.Series(model.predict_proba(x_test)[:, 1])
    labels = labels.astype(int)
    baseline_positive_rate = float(labels.mean()) if len(labels) else 0.0
    ranking_metrics = {
        "baseline_positive_rate": baseline_positive_rate,
    }
    for pct in (0.01, 0.03, 0.05, 0.10):
        bucket_size = max(1, int(len(probabilities) * pct))
        top_indices = probabilities.nlargest(bucket_size).index
        top_labels = labels.loc[top_indices]
        ranking_metrics[f"top_{int(pct * 100)}pct_count"] = int(bucket_size)
        ranking_metrics[f"top_{int(pct * 100)}pct_positive_rate"] = float(top_labels.mean()) if len(top_labels) else 0.0
        ranking_metrics[f"top_{int(pct * 100)}pct_lift"] = (
            float(top_labels.mean()) / baseline_positive_rate
            if len(top_labels) and baseline_positive_rate > 0
            else 0.0
        )
    return ranking_metrics


if __name__ == "__main__":
    main()

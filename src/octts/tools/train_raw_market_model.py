from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from octts.config import get_settings
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.modeling import save_model_artifact
from octts.tools.modeling_weights import (
    ANTI_CHASE_PROFILES,
    SAMPLE_WEIGHT_PROFILES,
    apply_limit_up_downweight,
    build_limit_up_mask,
    build_sample_weights,
    clip_return_target,
    count_clipped_return_target,
)

try:
    from sklearn.impute import SimpleImputer
except ImportError:  # pragma: no cover - handled at runtime when sklearn is missing
    SimpleImputer = None


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
    "weak_market_flag",
    "high_position_flag",
    "high_position_acceleration_flag",
    "weak_market_high_position_flag",
    "open_gap_pct",
    "open_gap_signed_pct",
    "intraday_return",
    "amplitude",
    "close_position_in_day",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "close_to_high",
    "close_to_low",
    "missing_window_flag",
    "missing_feature_count",
    "recent_runup_5d",
    "turnover_spike_ratio",
    "prev_day_limit_up",
    "prev_day_limit_open_times",
    "prev_day_limit_first_time",
    "prev_day_limit_last_time",
    "prev_day_limit_amount",
    "prev_day_fd_amount",
    "prev_day_limit_times",
    "prev_day_up_stat_success",
    "prev_day_up_stat_total",
    "prev_day_up_stat_ratio",
    "prev_day_one_word_limit_flag",
    "limit_chase_failure_risk_score",
]

CLASSIFICATION_TARGETS = [
    "label_up_1d",
    "label_up_3d",
    "label_up_5d",
    "label_vs_market_1d",
    "label_vs_market_3d",
    "label_vs_market_5d",
    "label_strong_1d",
    "label_limit_relay_success_1d",
    "label_limit_relay_strong_1d",
    "label_limit_relay_success_3d",
    "label_limit_relay_limit_up_1d",
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
    parser.add_argument("--lightgbm-num-leaves", type=int, default=15, help="LightGBM num_leaves; default matches max_depth=4")
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=120, help="LightGBM min_child_samples")
    parser.add_argument("--lightgbm-reg-alpha", type=float, default=0.05, help="LightGBM L1 regularization")
    parser.add_argument("--lightgbm-reg-lambda", type=float, default=1.0, help="LightGBM L2 regularization")
    parser.add_argument("--lightgbm-verbosity", type=int, default=-1, help="LightGBM verbosity; -1 suppresses split warnings")
    parser.add_argument("--sample-weight-mode", choices=["none", "anti_chase", "regime_anti_chase"], default="none", help="Sample weighting mode; default keeps legacy unweighted training")
    parser.add_argument("--anti-chase-profile", choices=sorted(ANTI_CHASE_PROFILES.keys()), default="default", help="Anti-chase threshold profile")
    parser.add_argument("--sample-weight-profile", choices=sorted(SAMPLE_WEIGHT_PROFILES.keys()), default="balanced", help="Sample weight strength profile")
    parser.add_argument("--limit-up-sample-mode", choices=["none", "drop", "downweight"], default="none", help="How to handle near/at limit-up training rows")
    parser.add_argument("--limit-up-pct-threshold", type=float, default=9.5, help="pct_change threshold for near/at limit-up rows")
    parser.add_argument("--limit-up-sample-weight", type=float, default=0.1, help="Multiplier for limit-up rows when --limit-up-sample-mode=downweight")
    parser.add_argument("--enable-return-clip", action="store_true", help="Clip return_* regression targets during training")
    parser.add_argument("--return-clip-low", type=float, default=-0.15, help="Lower bound for --enable-return-clip")
    parser.add_argument("--return-clip-high", type=float, default=0.20, help="Upper bound for --enable-return-clip")
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
    initial_labeled_rows = int(len(labeled))
    limit_up_mask = build_limit_up_mask(labeled, threshold=args.limit_up_pct_threshold)
    limit_up_sample_count = int(limit_up_mask.sum())
    limit_up_samples_dropped = 0
    if args.limit_up_sample_mode == "drop" and limit_up_sample_count:
        labeled = labeled.loc[~limit_up_mask].copy().reset_index(drop=True)
        limit_up_samples_dropped = limit_up_sample_count
        logger.info(
            "Dropped limit-up training samples: threshold=%.2f dropped=%s remaining=%s",
            args.limit_up_pct_threshold,
            limit_up_samples_dropped,
            len(labeled),
        )
    if labeled.empty:
        result = {
            "trained": False,
            "reason": "no_rows_after_limit_up_drop",
            "target": args.target,
            "limit_up_sample_mode": args.limit_up_sample_mode,
            "limit_up_sample_count": limit_up_sample_count,
        }
        logger.warning("Training skipped: %s", result)
        print_json(result)
        return
    feature_columns = resolve_feature_columns(
        labeled,
        feature_columns_arg=args.feature_columns,
        feature_file_arg=args.feature_file,
    )
    numeric_features = labeled[feature_columns].apply(pd.to_numeric, errors="coerce")
    all_missing_features = [column for column in numeric_features.columns if numeric_features[column].isna().all()]
    if all_missing_features:
        logger.warning("Dropping all-missing features before training: %s", all_missing_features)
        numeric_features = numeric_features.drop(columns=all_missing_features)
        feature_columns = [column for column in feature_columns if column not in all_missing_features]
    feature_null_stats = {
        column: float(numeric_features[column].isna().mean())
        for column in numeric_features.columns
        if numeric_features[column].isna().any()
    }
    is_regression = args.target in REGRESSION_TARGETS
    labels = (
        pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0)
        if is_regression
        else labeled[args.target].astype(bool).astype(int)
    )
    return_clip_count = 0
    if is_regression:
        labels_before_clip = labels.copy()
        labels = clip_return_target(
            labels,
            target_name=args.target,
            enabled=bool(args.enable_return_clip),
            lower=args.return_clip_low,
            upper=args.return_clip_high,
        )
        return_clip_count = count_clipped_return_target(labels_before_clip, labels)
        if args.enable_return_clip:
            logger.info(
                "Return target clip: target=%s enabled=%s low=%.4f high=%.4f clipped=%s",
                args.target,
                args.target.startswith("return_"),
                args.return_clip_low,
                args.return_clip_high,
                return_clip_count,
            )

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    x_train_raw = numeric_features.iloc[:split_index]
    y_train = labels.iloc[:split_index]
    x_test_raw = numeric_features.iloc[split_index:]
    y_test = labels.iloc[split_index:]
    train_labeled = labeled.iloc[:split_index]
    train_limit_up_sample_count = int(
        build_limit_up_mask(train_labeled, threshold=args.limit_up_pct_threshold).sum()
    )
    sample_weight = build_sample_weights(
        train_labeled,
        target_values=pd.Series(y_train, index=train_labeled.index),
        mode=args.sample_weight_mode,
        profile=args.anti_chase_profile,
        weight_profile=args.sample_weight_profile,
    )
    if args.limit_up_sample_mode == "downweight":
        sample_weight = apply_limit_up_downweight(
            sample_weight,
            train_labeled,
            threshold=args.limit_up_pct_threshold,
            weight=args.limit_up_sample_weight,
        )
    train_avg_sample_weight = (
        float(sample_weight.mean()) if sample_weight is not None and len(sample_weight) else None
    )

    x_train, feature_medians = _impute_missing_features(x_train_raw)
    x_test, _ = _impute_missing_features(x_test_raw, reference_frame=x_train_raw)

    model_kwargs = {}
    if args.model_type == "lightgbm":
        model_kwargs = {
            "num_leaves": args.lightgbm_num_leaves,
            "min_child_samples": args.lightgbm_min_child_samples,
            "reg_alpha": args.lightgbm_reg_alpha,
            "reg_lambda": args.lightgbm_reg_lambda,
            "verbosity": args.lightgbm_verbosity,
        }
    model = _fit_model(
        args.model_type,
        x_train,
        y_train,
        is_regression=is_regression,
        model_kwargs=model_kwargs,
        sample_weight=sample_weight,
    )
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
            "feature_null_stats": feature_null_stats,
            "missing_value_strategy": "median_imputation",
            "feature_medians": feature_medians,
            "model_params": model_kwargs,
            "sample_weight_mode": args.sample_weight_mode,
            "anti_chase_profile": args.anti_chase_profile,
            "sample_weight_profile": args.sample_weight_profile,
            "train_avg_sample_weight": train_avg_sample_weight,
            "limit_up_sample_mode": args.limit_up_sample_mode,
            "limit_up_pct_threshold": args.limit_up_pct_threshold,
            "limit_up_sample_weight": args.limit_up_sample_weight if args.limit_up_sample_mode == "downweight" else None,
            "limit_up_sample_count": limit_up_sample_count,
            "limit_up_samples_dropped": limit_up_samples_dropped,
            "train_limit_up_sample_count": train_limit_up_sample_count,
            "initial_labeled_rows": initial_labeled_rows,
            "return_clip_enabled": bool(args.enable_return_clip),
            "return_clip_low": args.return_clip_low if args.enable_return_clip else None,
            "return_clip_high": args.return_clip_high if args.enable_return_clip else None,
            "return_clip_count": return_clip_count,
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
        "initial_labeled_rows": initial_labeled_rows,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(feature_columns)),
        "sample_weight_mode": args.sample_weight_mode,
        "anti_chase_profile": args.anti_chase_profile,
        "sample_weight_profile": args.sample_weight_profile,
        "train_avg_sample_weight": train_avg_sample_weight,
        "limit_up_sample_mode": args.limit_up_sample_mode,
        "limit_up_pct_threshold": args.limit_up_pct_threshold,
        "limit_up_sample_weight": args.limit_up_sample_weight if args.limit_up_sample_mode == "downweight" else None,
        "limit_up_sample_count": limit_up_sample_count,
        "limit_up_samples_dropped": limit_up_samples_dropped,
        "train_limit_up_sample_count": train_limit_up_sample_count,
        "return_clip_enabled": bool(args.enable_return_clip),
        "return_clip_low": args.return_clip_low if args.enable_return_clip else None,
        "return_clip_high": args.return_clip_high if args.enable_return_clip else None,
        "return_clip_count": return_clip_count,
        "metrics": metrics,
        "ranking_metrics": ranking_metrics,
        "artifact_path": str(artifact_path),
    }
    logger.info("Raw-market model training complete: %s", result)
    print_json(result)


def _impute_missing_features(
    frame: pd.DataFrame,
    *,
    reference_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    if frame.empty:
        return frame.copy(), {}
    if SimpleImputer is None:
        raise RuntimeError("scikit-learn is not installed. Install it before training baseline models.")
    imputer = SimpleImputer(strategy="median")
    if reference_frame is None:
        transformed = imputer.fit_transform(frame)
        medians = {
            column: float(value)
            for column, value in zip(frame.columns, imputer.statistics_)
            if value is not None
        }
        return pd.DataFrame(transformed, columns=frame.columns, index=frame.index), medians
    imputer.fit(reference_frame)
    transformed = imputer.transform(frame)
    medians = {
        column: float(value)
        for column, value in zip(frame.columns, imputer.statistics_)
        if value is not None
    }
    return pd.DataFrame(transformed, columns=frame.columns, index=frame.index), medians


def _fit_model(
    model_type: str,
    features,
    labels,
    *,
    is_regression: bool,
    model_kwargs: Optional[dict] = None,
    sample_weight=None,
):
    model_kwargs = model_kwargs or {}
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
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
            **model_kwargs,
        )
        try:
            model.fit(features, labels, **fit_kwargs)
        except TypeError:
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
        kwargs.update(model_kwargs)
        model = model_cls(**kwargs)
        try:
            model.fit(features, labels, **fit_kwargs)
        except TypeError:
            model.fit(features, labels)
        return model
    try:
        from sklearn.linear_model import LogisticRegression, LinearRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is not installed. Install it before training baseline models.") from exc
    model = LinearRegression() if is_regression else LogisticRegression(max_iter=5000)
    try:
        model.fit(features, labels, **fit_kwargs)
    except TypeError:
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

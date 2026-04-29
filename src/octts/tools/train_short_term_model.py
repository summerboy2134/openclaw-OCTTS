from __future__ import annotations

import argparse
from pathlib import Path

from octts.config import get_settings
from octts.services.screening_store import ScreeningStore
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.modeling import build_feature_matrix, build_training_frame, save_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train short-term next-day-up model.")
    parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "xgboost", "logistic"])
    parser.add_argument("--output-name", default="latest")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "train_short_term_model")
    store = ScreeningStore(settings)
    samples = store.list_short_term_training_samples()
    labeled_samples = [sample for sample in samples if sample.get("label_up_1d") is not None]
    frame = build_training_frame(labeled_samples)
    features, labels = build_feature_matrix(frame)
    if features.empty or labels.empty:
        result = {"trained": False, "reason": "no_labeled_samples"}
        logger.warning("Training skipped: %s", result)
        print_json(result)
        return

    model = _fit_model(args.model_type, features, labels)
    model_dir = Path(settings.history_dir_path) / "short_term_models"
    artifact_path = model_dir / f"{args.output_name}.{args.model_type}.pkl"
    save_model_artifact(
        artifact_path,
        {
            "model_type": args.model_type,
            "feature_columns": list(features.columns),
            "model": model,
        },
    )
    result = {
        "trained": True,
        "model_type": args.model_type,
        "sample_count": int(len(features)),
        "feature_count": int(len(features.columns)),
        "artifact_path": str(artifact_path),
    }
    logger.info("Training complete: %s", result)
    print_json(result)


def _fit_model(model_type: str, features, labels):
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed. Install it before training this model.") from exc
        model = lgb.LGBMClassifier(
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
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed. Install it before training this model.") from exc
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(features, labels)
        return model
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is not installed. Install it before training logistic baseline.") from exc
    model = LogisticRegression(max_iter=2000)
    model.fit(features, labels)
    return model


if __name__ == "__main__":
    main()

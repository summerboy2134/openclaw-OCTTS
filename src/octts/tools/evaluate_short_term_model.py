from __future__ import annotations

import argparse
from pathlib import Path

from octts.config import get_settings
from octts.services.screening_store import ScreeningStore
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.modeling import build_feature_matrix, build_training_frame, load_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate short-term next-day-up model.")
    parser.add_argument("--artifact-path", required=True)
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "evaluate_short_term_model")
    artifact_path = Path(args.artifact_path)
    artifact = load_model_artifact(artifact_path)

    store = ScreeningStore(settings)
    samples = store.list_short_term_training_samples()
    labeled_samples = [sample for sample in samples if sample.get("label_up_1d") is not None]
    frame = build_training_frame(labeled_samples)
    features, labels = build_feature_matrix(frame)
    if features.empty or labels.empty:
        result = {"evaluated": False, "reason": "no_labeled_samples"}
        logger.warning("Evaluation skipped: %s", result)
        print_json(result)
        return

    feature_columns = artifact["feature_columns"]
    model = artifact["model"]
    for column in feature_columns:
        if column not in features.columns:
            features[column] = 0.0
    features = features[feature_columns]
    predictions = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else model.predict(features)
    result = _evaluate_predictions(labels, predictions)
    result.update(
        {
            "evaluated": True,
            "artifact_path": str(artifact_path),
            "sample_count": int(len(features)),
            "model_type": artifact.get("model_type"),
        }
    )
    logger.info("Evaluation complete: %s", result)
    print_json(result)


def _evaluate_predictions(labels, predictions):
    try:
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    except ImportError:
        binary_predictions = [1 if value >= 0.5 else 0 for value in predictions]
        accuracy = sum(int(pred == truth) for pred, truth in zip(binary_predictions, labels)) / len(labels)
        return {
            "accuracy": accuracy,
            "roc_auc": None,
            "log_loss": None,
        }
    binary_predictions = [1 if value >= 0.5 else 0 for value in predictions]
    return {
        "accuracy": float(accuracy_score(labels, binary_predictions)),
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "log_loss": float(log_loss(labels, predictions)),
    }


if __name__ == "__main__":
    main()

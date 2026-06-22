from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.modeling import load_model_artifact
from octts.tools.evaluate_raw_market_artifact_topn import _summarize_daily, _summarize_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dual raw-market model artifacts by daily Top-N picks.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--primary-artifact-path", required=True, help="Primary artifact, usually return_3d model")
    parser.add_argument("--secondary-artifact-path", required=True, help="Secondary artifact, usually label_up_1d model")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--stage1-n", type=int, default=300, help="First-stage pool size for two_stage mode")
    parser.add_argument("--target", default="return_3d")
    parser.add_argument("--secondary-target", default="return_1d")
    parser.add_argument("--mode", default="all", choices=["all", "primary", "secondary", "rank_blend", "two_stage"])
    parser.add_argument("--primary-weight", type=float, default=0.5)
    parser.add_argument("--secondary-weight", type=float, default=0.5)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"}, output_file=args.output_file or None)
        return
    required = {"trade_date", "ts_code", args.target, args.secondary_target}
    missing = sorted(column for column in required if column not in frame.columns)
    if missing:
        print_json({"evaluated": False, "reason": "missing_columns", "columns": missing}, output_file=args.output_file or None)
        return

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    if args.exclude_bj:
        frame = frame[~frame["ts_code"].str.endswith(".BJ")].copy()
    if args.start_date:
        frame = frame[frame["trade_date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        frame = frame[frame["trade_date"] <= pd.Timestamp(args.end_date)].copy()
    frame = frame[frame[args.target].notna() & frame[args.secondary_target].notna()].copy()
    if frame.empty:
        print_json({"evaluated": False, "reason": "no_labeled_rows"}, output_file=args.output_file or None)
        return

    primary = load_model_artifact(Path(args.primary_artifact_path))
    secondary = load_model_artifact(Path(args.secondary_artifact_path))
    scored = frame[["trade_date", "ts_code", args.target, args.secondary_target]].copy()
    scored[args.target] = pd.to_numeric(scored[args.target], errors="coerce")
    scored[args.secondary_target] = pd.to_numeric(scored[args.secondary_target], errors="coerce")
    scored["primary_score"] = _predict_artifact(frame, primary)
    scored["secondary_score"] = _predict_artifact(frame, secondary)

    modes = ["primary", "secondary", "rank_blend", "two_stage"] if args.mode == "all" else [args.mode]
    results: Dict[str, Any] = {}
    for mode in modes:
        daily_result = _evaluate_mode(
            scored,
            mode=mode,
            top_n=args.top_n,
            stage1_n=args.stage1_n,
            target=args.target,
            secondary_target=args.secondary_target,
            primary_weight=args.primary_weight,
            secondary_weight=args.secondary_weight,
        )
        results[mode] = daily_result

    payload = {
        "evaluated": True,
        "input": args.input,
        "primary_artifact_path": args.primary_artifact_path,
        "secondary_artifact_path": args.secondary_artifact_path,
        "target": args.target,
        "secondary_target": args.secondary_target,
        "top_n": int(args.top_n),
        "stage1_n": int(args.stage1_n),
        "weights": {"primary": float(args.primary_weight), "secondary": float(args.secondary_weight)},
        "rows": int(len(scored)),
        "date_range": {
            "start": scored["trade_date"].min().date().isoformat(),
            "end": scored["trade_date"].max().date().isoformat(),
            "trade_days": int(scored["trade_date"].nunique()),
        },
        "baseline": _summarize_frame(scored, target=args.target, secondary_target=args.secondary_target),
        "results": results,
    }
    print_json(payload, output_file=args.output_file or None)


def _predict_artifact(frame: pd.DataFrame, artifact: Dict[str, Any]) -> pd.Series:
    feature_columns = list(artifact.get("feature_columns") or [])
    feature_medians = artifact.get("feature_medians") or {}
    model = artifact.get("model")
    if not feature_columns or model is None:
        raise RuntimeError("invalid artifact: missing feature_columns or model")
    feature_frame = frame.copy()
    for column in feature_columns:
        if column not in feature_frame.columns:
            feature_frame[column] = None
    features = feature_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if feature_medians:
        features = features.fillna(pd.Series(feature_medians))
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    if hasattr(model, "predict_proba") and str(artifact.get("task_type") or "").lower() == "classification":
        values = model.predict_proba(features)[:, 1]
    elif hasattr(model, "predict_proba") and str(artifact.get("target") or "").startswith("label_"):
        values = model.predict_proba(features)[:, 1]
    else:
        values = model.predict(features)
    return pd.Series(values, index=frame.index).astype(float)


def _evaluate_mode(
    scored: pd.DataFrame,
    *,
    mode: str,
    top_n: int,
    stage1_n: int,
    target: str,
    secondary_target: str,
    primary_weight: float,
    secondary_weight: float,
) -> Dict[str, Any]:
    daily: List[Dict[str, Any]] = []
    for trade_date, day_frame in scored.groupby("trade_date", sort=True):
        day_frame = day_frame.copy()
        day_frame["primary_rank_pct"] = day_frame["primary_score"].rank(method="average", pct=True)
        day_frame["secondary_rank_pct"] = day_frame["secondary_score"].rank(method="average", pct=True)
        if mode == "primary":
            selected = day_frame.nlargest(min(top_n, len(day_frame)), columns="primary_score")
        elif mode == "secondary":
            selected = day_frame.nlargest(min(top_n, len(day_frame)), columns="secondary_score")
        elif mode == "rank_blend":
            day_frame["blend_score"] = primary_weight * day_frame["primary_rank_pct"] + secondary_weight * day_frame["secondary_rank_pct"]
            selected = day_frame.nlargest(min(top_n, len(day_frame)), columns="blend_score")
        elif mode == "two_stage":
            stage1 = day_frame.nlargest(min(stage1_n, len(day_frame)), columns="primary_score")
            selected = stage1.nlargest(min(top_n, len(stage1)), columns="secondary_score")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        baseline = _summarize_frame(day_frame, target=target, secondary_target=secondary_target)
        topn = _summarize_frame(selected, target=target, secondary_target=secondary_target)
        daily.append({
            "trade_date": trade_date.date().isoformat(),
            "universe_rows": int(len(day_frame)),
            "picked_rows": int(len(selected)),
            "baseline": baseline,
            "topn": topn,
            "excess": {
                target: topn[f"mean_{target}"] - baseline[f"mean_{target}"],
                secondary_target: topn[f"mean_{secondary_target}"] - baseline[f"mean_{secondary_target}"],
                f"positive_rate_{target}": topn[f"positive_rate_{target}"] - baseline[f"positive_rate_{target}"],
                f"positive_rate_{secondary_target}": topn[f"positive_rate_{secondary_target}"] - baseline[f"positive_rate_{secondary_target}"],
            },
            "top_codes": selected.head(10)["ts_code"].tolist(),
        })
    return {"summary": _summarize_daily(daily, target=target, secondary_target=secondary_target), "daily": daily}


if __name__ == "__main__":
    main()

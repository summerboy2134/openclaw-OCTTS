from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.modeling import load_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained raw-market model artifact by daily Top-N picks.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--artifact-path", required=True, help="Model artifact path")
    parser.add_argument("--start-date", default="", help="Optional evaluation start date, e.g. 2026-05-16")
    parser.add_argument("--end-date", default="", help="Optional evaluation end date, e.g. 2026-06-01")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--target", default="return_3d", help="Target return column for ranking quality")
    parser.add_argument("--secondary-target", default="return_1d", help="Secondary return column, usually return_1d")
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--output-file", default="", help="Optional JSON output file")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"}, output_file=args.output_file or None)
        return

    required_columns = {"trade_date", "ts_code", args.target, args.secondary_target}
    missing_columns = sorted(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        print_json(
            {"evaluated": False, "reason": "missing_columns", "columns": missing_columns},
            output_file=args.output_file or None,
        )
        return

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    if args.exclude_bj:
        frame = frame[~frame["ts_code"].str.endswith(".BJ")].copy()
    if args.start_date:
        frame = frame[frame["trade_date"] >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        frame = frame[frame["trade_date"] <= pd.Timestamp(args.end_date)].copy()

    frame = frame[frame[args.target].notna()].copy()
    if frame.empty:
        print_json({"evaluated": False, "reason": "no_labeled_rows", "target": args.target}, output_file=args.output_file or None)
        return

    artifact = load_model_artifact(Path(args.artifact_path))
    feature_columns = list(artifact.get("feature_columns") or [])
    feature_medians = artifact.get("feature_medians") or {}
    model = artifact.get("model")
    if not feature_columns or model is None:
        print_json({"evaluated": False, "reason": "invalid_artifact"}, output_file=args.output_file or None)
        return

    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = None
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if feature_medians:
        features = features.fillna(pd.Series(feature_medians))
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)

    predictions = model.predict(features)
    scored = frame[["trade_date", "ts_code", args.target, args.secondary_target]].copy()
    scored["prediction"] = pd.Series(predictions, index=scored.index).astype(float)
    scored[args.target] = pd.to_numeric(scored[args.target], errors="coerce")
    scored[args.secondary_target] = pd.to_numeric(scored[args.secondary_target], errors="coerce")

    daily_results = _evaluate_daily_topn(
        scored,
        top_n=args.top_n,
        target=args.target,
        secondary_target=args.secondary_target,
    )
    payload = {
        "evaluated": True,
        "input": args.input,
        "artifact_path": args.artifact_path,
        "target": args.target,
        "secondary_target": args.secondary_target,
        "top_n": int(args.top_n),
        "feature_count": int(len(feature_columns)),
        "rows": int(len(scored)),
        "date_range": {
            "start": scored["trade_date"].min().date().isoformat() if not scored.empty else None,
            "end": scored["trade_date"].max().date().isoformat() if not scored.empty else None,
            "trade_days": int(scored["trade_date"].nunique()),
        },
        "baseline": _summarize_frame(scored, target=args.target, secondary_target=args.secondary_target),
        "topn_summary": daily_results["summary"],
        "daily": daily_results["daily"],
    }
    print_json(payload, output_file=args.output_file or None)


def _evaluate_daily_topn(scored: pd.DataFrame, *, top_n: int, target: str, secondary_target: str) -> Dict[str, Any]:
    daily: List[Dict[str, Any]] = []
    for trade_date, day_frame in scored.groupby("trade_date", sort=True):
        selected = day_frame.nlargest(min(top_n, len(day_frame)), columns="prediction")
        day_baseline = _summarize_frame(day_frame, target=target, secondary_target=secondary_target)
        day_topn = _summarize_frame(selected, target=target, secondary_target=secondary_target)
        daily.append(
            {
                "trade_date": trade_date.date().isoformat(),
                "universe_rows": int(len(day_frame)),
                "picked_rows": int(len(selected)),
                "baseline": day_baseline,
                "topn": day_topn,
                "excess": {
                    target: day_topn[f"mean_{target}"] - day_baseline[f"mean_{target}"],
                    secondary_target: day_topn[f"mean_{secondary_target}"] - day_baseline[f"mean_{secondary_target}"],
                    f"positive_rate_{target}": day_topn[f"positive_rate_{target}"] - day_baseline[f"positive_rate_{target}"],
                    f"positive_rate_{secondary_target}": day_topn[f"positive_rate_{secondary_target}"] - day_baseline[f"positive_rate_{secondary_target}"],
                },
                "top_codes": selected.head(10)["ts_code"].tolist(),
            }
        )

    summary = _summarize_daily(daily, target=target, secondary_target=secondary_target)
    return {"summary": summary, "daily": daily}


def _summarize_frame(frame: pd.DataFrame, *, target: str, secondary_target: str) -> Dict[str, float]:
    return {
        f"mean_{target}": float(frame[target].mean()) if len(frame) else 0.0,
        f"median_{target}": float(frame[target].median()) if len(frame) else 0.0,
        f"positive_rate_{target}": float((frame[target] > 0).mean()) if len(frame) else 0.0,
        f"mean_{secondary_target}": float(frame[secondary_target].mean()) if len(frame) else 0.0,
        f"median_{secondary_target}": float(frame[secondary_target].median()) if len(frame) else 0.0,
        f"positive_rate_{secondary_target}": float((frame[secondary_target] > 0).mean()) if len(frame) else 0.0,
    }


def _summarize_daily(daily: List[Dict[str, Any]], *, target: str, secondary_target: str) -> Dict[str, float]:
    if not daily:
        return {}
    fields = [
        f"mean_{target}",
        f"median_{target}",
        f"positive_rate_{target}",
        f"mean_{secondary_target}",
        f"median_{secondary_target}",
        f"positive_rate_{secondary_target}",
    ]
    summary: Dict[str, float] = {"trade_days": float(len(daily))}
    for field in fields:
        summary[f"avg_daily_topn_{field}"] = float(sum(item["topn"][field] for item in daily) / len(daily))
        summary[f"avg_daily_baseline_{field}"] = float(sum(item["baseline"][field] for item in daily) / len(daily))
    summary[f"avg_daily_excess_{target}"] = float(sum(item["excess"][target] for item in daily) / len(daily))
    summary[f"avg_daily_excess_{secondary_target}"] = float(
        sum(item["excess"][secondary_target] for item in daily) / len(daily)
    )
    summary[f"avg_daily_excess_positive_rate_{target}"] = float(
        sum(item["excess"][f"positive_rate_{target}"] for item in daily) / len(daily)
    )
    summary[f"avg_daily_excess_positive_rate_{secondary_target}"] = float(
        sum(item["excess"][f"positive_rate_{secondary_target}"] for item in daily) / len(daily)
    )
    return summary


if __name__ == "__main__":
    main()

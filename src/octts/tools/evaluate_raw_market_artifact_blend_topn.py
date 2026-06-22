from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.modeling import load_model_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blended raw-market model artifacts by daily Top-N picks.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact spec: name=path=weight. Can be provided multiple times.",
    )
    parser.add_argument("--mode", default="rank_blend", choices=["rank_blend", "two_stage"])
    parser.add_argument("--primary-artifact", default="", help="For two_stage mode: name of primary artifact")
    parser.add_argument("--secondary-artifact", default="", help="For two_stage mode: name of secondary artifact")
    parser.add_argument("--primary-top-n", type=int, default=300, help="For two_stage mode")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--target", default="return_3d")
    parser.add_argument("--secondary-target", default="return_1d")
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--output-file", default="")
    args = parser.parse_args()

    specs = [_parse_artifact_spec(value) for value in args.artifact]
    if not specs:
        print_json({"evaluated": False, "reason": "missing_artifacts"}, output_file=args.output_file or None)
        return

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"}, output_file=args.output_file or None)
        return

    required_columns = {"trade_date", "ts_code", args.target, args.secondary_target}
    missing_columns = sorted(column for column in required_columns if column not in frame.columns)
    if missing_columns:
        print_json({"evaluated": False, "reason": "missing_columns", "columns": missing_columns}, output_file=args.output_file or None)
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

    scored = frame[["trade_date", "ts_code", args.target, args.secondary_target]].copy()
    artifact_payloads = []
    for spec in specs:
        artifact = load_model_artifact(Path(spec["path"]))
        score_column = f"score__{spec['name']}"
        scored[score_column] = _predict_artifact(frame, artifact)
        artifact_payloads.append({**spec, "score_column": score_column, "feature_count": len(artifact.get("feature_columns") or [])})

    if args.mode == "rank_blend":
        scored["blend_score"] = _build_rank_blend(scored, artifact_payloads)
        selected_by_day = _select_rank_blend(scored, top_n=args.top_n)
    else:
        selected_by_day = _select_two_stage(
            scored,
            artifact_payloads,
            primary_name=args.primary_artifact,
            secondary_name=args.secondary_artifact,
            primary_top_n=args.primary_top_n,
            top_n=args.top_n,
        )

    daily = _evaluate_daily(
        scored,
        selected_by_day,
        target=args.target,
        secondary_target=args.secondary_target,
    )
    payload = {
        "evaluated": True,
        "input": args.input,
        "mode": args.mode,
        "artifacts": artifact_payloads,
        "target": args.target,
        "secondary_target": args.secondary_target,
        "top_n": int(args.top_n),
        "primary_top_n": int(args.primary_top_n) if args.mode == "two_stage" else None,
        "date_range": {
            "start": scored["trade_date"].min().date().isoformat() if not scored.empty else None,
            "end": scored["trade_date"].max().date().isoformat() if not scored.empty else None,
            "trade_days": int(scored["trade_date"].nunique()),
        },
        "baseline": _summarize_frame(scored, target=args.target, secondary_target=args.secondary_target),
        "topn_summary": _summarize_daily(daily, target=args.target, secondary_target=args.secondary_target),
        "daily": daily,
    }
    print_json(payload, output_file=args.output_file or None)


def _parse_artifact_spec(value: str) -> Dict[str, Any]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise ValueError("artifact spec must be name=path=weight")
    name, path, weight = parts
    return {"name": name, "path": path, "weight": float(weight)}


def _predict_artifact(frame: pd.DataFrame, artifact: Dict[str, Any]) -> pd.Series:
    feature_columns = list(artifact.get("feature_columns") or [])
    feature_medians = artifact.get("feature_medians") or {}
    model = artifact.get("model")
    if not feature_columns or model is None:
        raise RuntimeError("artifact missing feature_columns or model")
    features = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    if feature_medians:
        features = features.fillna(pd.Series(feature_medians))
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    if hasattr(model, "predict_proba"):
        try:
            return pd.Series(model.predict_proba(features)[:, 1], index=frame.index, dtype=float)
        except Exception:
            pass
    return pd.Series(model.predict(features), index=frame.index, dtype=float)


def _build_rank_blend(scored: pd.DataFrame, artifact_specs: List[Dict[str, Any]]) -> pd.Series:
    blended = pd.Series(0.0, index=scored.index)
    total_weight = sum(float(spec["weight"]) for spec in artifact_specs) or 1.0
    for spec in artifact_specs:
        ranks = scored.groupby("trade_date")[spec["score_column"]].rank(method="average", pct=True)
        blended = blended.add(ranks * (float(spec["weight"]) / total_weight), fill_value=0.0)
    return blended


def _select_rank_blend(scored: pd.DataFrame, *, top_n: int) -> Dict[pd.Timestamp, pd.DataFrame]:
    return {
        trade_date: day_frame.nlargest(min(top_n, len(day_frame)), columns="blend_score")
        for trade_date, day_frame in scored.groupby("trade_date", sort=True)
    }


def _select_two_stage(
    scored: pd.DataFrame,
    artifact_specs: List[Dict[str, Any]],
    *,
    primary_name: str,
    secondary_name: str,
    primary_top_n: int,
    top_n: int,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    columns_by_name = {spec["name"]: spec["score_column"] for spec in artifact_specs}
    primary_column = columns_by_name.get(primary_name)
    secondary_column = columns_by_name.get(secondary_name)
    if primary_column is None or secondary_column is None:
        raise RuntimeError("two_stage requires valid --primary-artifact and --secondary-artifact names")
    selected: Dict[pd.Timestamp, pd.DataFrame] = {}
    for trade_date, day_frame in scored.groupby("trade_date", sort=True):
        primary_pool = day_frame.nlargest(min(primary_top_n, len(day_frame)), columns=primary_column)
        selected[trade_date] = primary_pool.nlargest(min(top_n, len(primary_pool)), columns=secondary_column)
    return selected


def _evaluate_daily(
    scored: pd.DataFrame,
    selected_by_day: Dict[pd.Timestamp, pd.DataFrame],
    *,
    target: str,
    secondary_target: str,
) -> List[Dict[str, Any]]:
    daily: List[Dict[str, Any]] = []
    for trade_date, day_frame in scored.groupby("trade_date", sort=True):
        selected = selected_by_day.get(trade_date, day_frame.iloc[0:0])
        baseline = _summarize_frame(day_frame, target=target, secondary_target=secondary_target)
        topn = _summarize_frame(selected, target=target, secondary_target=secondary_target)
        daily.append(
            {
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
            }
        )
    return daily


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
    summary[f"avg_daily_excess_{secondary_target}"] = float(sum(item["excess"][secondary_target] for item in daily) / len(daily))
    summary[f"avg_daily_excess_positive_rate_{target}"] = float(
        sum(item["excess"][f"positive_rate_{target}"] for item in daily) / len(daily)
    )
    summary[f"avg_daily_excess_positive_rate_{secondary_target}"] = float(
        sum(item["excess"][f"positive_rate_{secondary_target}"] for item in daily) / len(daily)
    )
    return summary


if __name__ == "__main__":
    main()

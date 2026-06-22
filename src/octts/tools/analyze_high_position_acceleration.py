from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import pandas as pd

from octts.tools.common import print_json


DEFAULT_PRICE_POSITION_THRESHOLD = 0.88
DEFAULT_PCT_CHANGE_THRESHOLD = 5.0
DEFAULT_VOLUME_RATIO_THRESHOLD = 2.5
DEFAULT_TURNOVER_RATE_THRESHOLD = 8.0
DEFAULT_RETURN_5D_PAST_THRESHOLD = 0.08
DEFAULT_MIN_GROUP_ROWS = 50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether high-position acceleration samples earn stronger 1-3 day forward returns."
    )
    parser.add_argument("--input", required=True, help="Raw-market training CSV path")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--price-position-threshold", type=float, default=DEFAULT_PRICE_POSITION_THRESHOLD)
    parser.add_argument("--pct-change-threshold", type=float, default=DEFAULT_PCT_CHANGE_THRESHOLD)
    parser.add_argument("--volume-ratio-threshold", type=float, default=DEFAULT_VOLUME_RATIO_THRESHOLD)
    parser.add_argument("--turnover-rate-threshold", type=float, default=DEFAULT_TURNOVER_RATE_THRESHOLD)
    parser.add_argument("--return-5d-past-threshold", type=float, default=DEFAULT_RETURN_5D_PAST_THRESHOLD)
    parser.add_argument("--min-group-rows", type=int, default=DEFAULT_MIN_GROUP_ROWS)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if frame.empty:
        print_json({"analyzed": False, "reason": "empty_dataset"}, output_file=args.output or None)
        return

    required_columns = [
        "trade_date",
        "ts_code",
        "price_position_20d",
        "pct_change",
        "volume_ratio",
        "turnover_rate",
        "return_1d",
        "return_3d",
    ]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        print_json(
            {
                "analyzed": False,
                "reason": "missing_columns",
                "columns": missing_columns,
            },
            output_file=args.output or None,
        )
        return

    numeric_columns = [
        "price_position_20d",
        "pct_change",
        "volume_ratio",
        "turnover_rate",
        "return_5d_past",
        "return_1d",
        "return_3d",
        "return_5d",
        "vs_market_1d",
        "vs_market_3d",
        "vs_market_5d",
    ]
    prepared = frame.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    analyzed = prepared[
        prepared["trade_date"].notna()
        & prepared["price_position_20d"].notna()
        & prepared["pct_change"].notna()
        & prepared["volume_ratio"].notna()
        & prepared["turnover_rate"].notna()
        & prepared["return_1d"].notna()
        & prepared["return_3d"].notna()
    ].copy()
    if analyzed.empty:
        print_json({"analyzed": False, "reason": "no_usable_rows"}, output_file=args.output or None)
        return

    price_position_high = analyzed["price_position_20d"] >= args.price_position_threshold
    same_day_acceleration = (
        (analyzed["pct_change"] >= args.pct_change_threshold)
        & (analyzed["volume_ratio"] >= args.volume_ratio_threshold)
        & (analyzed["turnover_rate"] >= args.turnover_rate_threshold)
    )
    runup_acceleration = pd.Series(False, index=analyzed.index)
    if "return_5d_past" in analyzed.columns:
        runup_acceleration = (
            analyzed["return_5d_past"].notna()
            & (analyzed["return_5d_past"] >= args.return_5d_past_threshold)
            & (analyzed["volume_ratio"] >= args.volume_ratio_threshold)
        )

    high_position_same_day_accel = price_position_high & same_day_acceleration
    high_position_runup_accel = price_position_high & runup_acceleration
    high_position_any_accel = price_position_high & (same_day_acceleration | runup_acceleration)
    low_position_same_day_accel = (~price_position_high) & same_day_acceleration

    groups: Dict[str, pd.Series] = {
        "baseline_all": pd.Series(True, index=analyzed.index),
        "high_position": price_position_high,
        "same_day_acceleration": same_day_acceleration,
        "high_position_same_day_acceleration": high_position_same_day_accel,
        "high_position_runup_acceleration_proxy": high_position_runup_accel,
        "high_position_any_acceleration": high_position_any_accel,
        "low_position_same_day_acceleration": low_position_same_day_accel,
    }

    summaries = {
        group_name: _summarize_group(
            analyzed.loc[mask],
            total_rows=len(analyzed),
            total_days=int(analyzed["trade_date"].nunique()),
        )
        for group_name, mask in groups.items()
    }

    hypothesis_checks = _build_hypothesis_checks(
        summaries=summaries,
        min_group_rows=args.min_group_rows,
        focus_group="high_position_any_acceleration",
        compare_groups=["baseline_all", "high_position", "same_day_acceleration", "low_position_same_day_acceleration"],
    )

    bucket_analysis = _build_bucket_analysis(
        analyzed,
        min_group_rows=args.min_group_rows,
    )

    result: Dict[str, Any] = {
        "analyzed": True,
        "input": args.input,
        "dataset_summary": {
            "rows": int(len(analyzed)),
            "days": int(analyzed["trade_date"].nunique()),
            "symbols": int(analyzed["ts_code"].nunique()),
            "date_min": analyzed["trade_date"].min().date().isoformat(),
            "date_max": analyzed["trade_date"].max().date().isoformat(),
        },
        "thresholds": {
            "price_position_20d": args.price_position_threshold,
            "pct_change": args.pct_change_threshold,
            "volume_ratio": args.volume_ratio_threshold,
            "turnover_rate": args.turnover_rate_threshold,
            "return_5d_past_proxy": args.return_5d_past_threshold,
            "min_group_rows": args.min_group_rows,
        },
        "notes": [
            "high_position 使用 price_position_20d 阈值。",
            "same_day_acceleration 使用当日 pct_change + volume_ratio + turnover_rate。",
            "runup_acceleration_proxy 使用 return_5d_past 作为 recent_runup_5d 的近似代理，不是逐日 pct_change 求和。",
        ],
        "group_summaries": summaries,
        "hypothesis_checks": hypothesis_checks,
        "bucket_analysis": bucket_analysis,
    }
    print_json(result, output_file=args.output or None)


def _summarize_group(frame: pd.DataFrame, *, total_rows: int, total_days: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": int(len(frame)),
        "days": int(frame["trade_date"].nunique()) if not frame.empty else 0,
        "row_share": float(len(frame) / total_rows) if total_rows else 0.0,
        "day_share": float(frame["trade_date"].nunique() / total_days) if total_days else 0.0,
    }
    if frame.empty:
        return summary

    for column in [
        "price_position_20d",
        "pct_change",
        "volume_ratio",
        "turnover_rate",
        "return_5d_past",
        "return_1d",
        "return_3d",
        "return_5d",
        "vs_market_1d",
        "vs_market_3d",
        "vs_market_5d",
    ]:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        summary[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
        }
        if column.startswith("return_") or column.startswith("vs_market_"):
            summary[column]["win_rate"] = float((series > 0).mean())

    return summary


def _build_hypothesis_checks(
    *,
    summaries: Dict[str, Dict[str, Any]],
    min_group_rows: int,
    focus_group: str,
    compare_groups: List[str],
) -> Dict[str, Any]:
    focus_summary = summaries.get(focus_group) or {}
    focus_rows = int(focus_summary.get("rows") or 0)
    checks: Dict[str, Any] = {
        "focus_group": focus_group,
        "focus_group_rows": focus_rows,
        "focus_group_large_enough": focus_rows >= min_group_rows,
        "comparisons": {},
    }
    if focus_rows <= 0:
        return checks

    for compare_group in compare_groups:
        compare_summary = summaries.get(compare_group) or {}
        comparison: Dict[str, Any] = {
            "compare_group_rows": int(compare_summary.get("rows") or 0),
        }
        for metric in ["return_1d", "return_3d", "return_5d", "vs_market_1d", "vs_market_3d", "vs_market_5d"]:
            focus_metric = _metric_mean(focus_summary, metric)
            compare_metric = _metric_mean(compare_summary, metric)
            if focus_metric is None or compare_metric is None:
                continue
            comparison[metric] = {
                "focus_mean": focus_metric,
                "compare_mean": compare_metric,
                "difference": focus_metric - compare_metric,
                "focus_beats_compare": focus_metric > compare_metric,
            }
        checks["comparisons"][compare_group] = comparison
    return checks


def _build_bucket_analysis(frame: pd.DataFrame, *, min_group_rows: int) -> Dict[str, Any]:
    price_position_labels = ["<0.50", "0.50-0.80", "0.80-0.88", "0.88-0.95", ">=0.95"]
    frame = frame.copy()
    frame["price_position_bucket"] = pd.cut(
        frame["price_position_20d"],
        bins=[-float("inf"), 0.50, 0.80, 0.88, 0.95, float("inf")],
        labels=price_position_labels,
        right=False,
    )

    bucket_result: Dict[str, Any] = {
        "price_position_buckets": [],
        "price_position_x_runup_proxy": [],
    }

    for label in price_position_labels:
        bucket_frame = frame[frame["price_position_bucket"] == label]
        if len(bucket_frame) < min_group_rows:
            continue
        bucket_summary = _summarize_group(
            bucket_frame,
            total_rows=len(frame),
            total_days=int(frame["trade_date"].nunique()),
        )
        bucket_result["price_position_buckets"].append({"bucket": label, **bucket_summary})

    if "return_5d_past" not in frame.columns:
        return bucket_result

    runup_labels = ["<0%", "0%-3%", "3%-8%", ">=8%"]
    frame["runup_proxy_bucket"] = pd.cut(
        frame["return_5d_past"],
        bins=[-float("inf"), 0.0, 0.03, 0.08, float("inf")],
        labels=runup_labels,
        right=False,
    )
    for price_label in ["0.88-0.95", ">=0.95"]:
        for runup_label in runup_labels:
            bucket_frame = frame[
                (frame["price_position_bucket"] == price_label)
                & (frame["runup_proxy_bucket"] == runup_label)
            ]
            if len(bucket_frame) < min_group_rows:
                continue
            bucket_summary = _summarize_group(
                bucket_frame,
                total_rows=len(frame),
                total_days=int(frame["trade_date"].nunique()),
            )
            bucket_result["price_position_x_runup_proxy"].append(
                {
                    "price_position_bucket": price_label,
                    "runup_proxy_bucket": runup_label,
                    **bucket_summary,
                }
            )
    return bucket_result


def _metric_mean(summary: Dict[str, Any], metric: str) -> Optional[float]:
    metric_summary = summary.get(metric)
    if not isinstance(metric_summary, dict):
        return None
    mean_value = metric_summary.get("mean")
    if mean_value is None:
        return None
    return float(mean_value)


if __name__ == "__main__":
    main()

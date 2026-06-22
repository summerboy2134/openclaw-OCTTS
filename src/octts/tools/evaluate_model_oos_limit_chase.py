from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from octts.config import get_settings
from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.modeling import load_model_artifact

TOP_PCTS = [0.01, 0.03, 0.05, 0.10]
PCT_BUCKETS = [
    ("limit_like_ge_9_5", 9.5, None),
    ("strong_7_5_to_9_5", 7.5, 9.5),
    ("mid_3_to_7_5", 3.0, 7.5),
    ("mild_0_to_3", 0.0, 3.0),
    ("negative", None, 0.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Out-of-sample diagnostics for raw-market model artifacts and recent actual Top3."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument(
        "--artifacts",
        required=True,
        help="Comma-separated artifact paths. Optional label=path entries are supported.",
    )
    parser.add_argument(
        "--feature-source",
        choices=["training_features", "dynamic"],
        default="training_features",
        help="training_features reads persisted feature rows; dynamic rebuilds samples from raw market data.",
    )
    parser.add_argument("--target", default="return_3d", help="Forward return column to evaluate")
    parser.add_argument("--limit-up-threshold", type=float, default=9.5)
    parser.add_argument("--output-file", help="Optional JSON output file")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "evaluate_model_oos_limit_chase")
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    artifacts = _parse_artifacts(args.artifacts, settings=settings)

    logger.info(
        "Loading OOS features: source=%s, start=%s, end=%s",
        args.feature_source,
        start_date.isoformat(),
        end_date.isoformat(),
    )
    frame = _load_feature_frame(settings, start_date, end_date, source=args.feature_source)
    if frame.empty:
        result = {
            "ok": False,
            "reason": "empty_feature_frame",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "feature_source": args.feature_source,
        }
        print_json(result)
        return
    if args.target not in frame.columns:
        raise ValueError(f"Target column not available in feature frame: {args.target}")
    frame = frame[frame[args.target].notna()].copy()
    if frame.empty:
        result = {
            "ok": False,
            "reason": "empty_labeled_frame",
            "target": args.target,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        print_json(result)
        return
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["pct_change"] = pd.to_numeric(frame.get("pct_change"), errors="coerce")
    frame[args.target] = pd.to_numeric(frame[args.target], errors="coerce")

    model_results = []
    for label, artifact_path in artifacts:
        logger.info("Evaluating artifact: %s -> %s", label, artifact_path)
        artifact = load_model_artifact(artifact_path)
        model_results.append(
            _evaluate_artifact(
                label=label,
                artifact_path=artifact_path,
                artifact=artifact,
                frame=frame,
                target=args.target,
                limit_up_threshold=args.limit_up_threshold,
            )
        )

    actual_top3 = _evaluate_actual_top3(
        settings.database_url,
        start_date=start_date,
        end_date=end_date,
        limit_up_threshold=args.limit_up_threshold,
    )
    result = {
        "ok": True,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "feature_source": args.feature_source,
        "target": args.target,
        "rows": int(len(frame)),
        "trade_dates": int(frame["trade_date"].nunique()),
        "baseline_mean_return": _safe_mean(frame[args.target]),
        "pct_change_bucket_forward_returns": _summarize_pct_buckets(
            frame,
            target=args.target,
            limit_up_threshold=args.limit_up_threshold,
        ),
        "models": model_results,
        "actual_top3": actual_top3,
    }
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logger.info("Saved OOS diagnostic output: %s", output_path)
    print_json(result)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_artifacts(raw: str, *, settings) -> List[Tuple[str, Path]]:
    model_dir = Path(settings.history_dir_path) / "short_term_models"
    result = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            label, path_text = token.split("=", 1)
            label = label.strip()
            path_text = path_text.strip()
        else:
            path_text = token
            label = Path(path_text).stem
        path = Path(path_text)
        if not path.is_absolute():
            if path.exists():
                path = path.resolve()
            else:
                path = model_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        result.append((label or path.stem, path))
    if not result:
        raise ValueError("--artifacts resolved to an empty list")
    return result


def _load_feature_frame(settings, start_date: date, end_date: date, *, source: str) -> pd.DataFrame:
    if source == "dynamic":
        builder = RawMarketTrainingDatasetBuilder(settings)
        samples = builder.build_samples(start_date=start_date, end_date=end_date, min_history_days=20)
        return pd.DataFrame([sample.model_dump(mode="python") for sample in samples])
    database_path = _sqlite_path(settings.database_url)
    conn = sqlite3.connect(database_path)
    try:
        query = "SELECT * FROM training_features WHERE trade_date >= ? AND trade_date <= ?"
        return pd.read_sql_query(query, conn, params=(start_date.isoformat(), end_date.isoformat()))
    finally:
        conn.close()


def _sqlite_path(database_url: str) -> str:
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return database_url[len(prefix):]
    raise ValueError(f"Only sqlite database_url is supported by this diagnostic tool: {database_url}")


def _evaluate_artifact(
    *,
    label: str,
    artifact_path: Path,
    artifact: Dict[str, Any],
    frame: pd.DataFrame,
    target: str,
    limit_up_threshold: float,
) -> Dict[str, Any]:
    feature_columns = list(artifact.get("feature_columns") or [])
    model = artifact.get("model")
    if not feature_columns or model is None:
        raise ValueError(f"Artifact missing feature_columns or model: {artifact_path}")
    missing_columns = [column for column in feature_columns if column not in frame.columns]
    rows = []
    for column in feature_columns:
        rows.append(pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(0.0, index=frame.index))
    X = pd.concat(rows, axis=1)
    X.columns = feature_columns
    medians = artifact.get("feature_medians") or {}
    for column in X.columns:
        if column in medians:
            X[column] = X[column].fillna(float(medians[column]))
    X = X.fillna(0.0)
    predictions = model.predict(X)
    scored_columns = ["trade_date", "ts_code", "pct_change", target]
    for optional_column in ("prev_day_limit_up", "limit_chase_failure_risk_score"):
        if optional_column in frame.columns:
            scored_columns.append(optional_column)
    scored = frame[scored_columns].copy()
    scored["pred"] = predictions
    return {
        "label": label,
        "artifact_path": str(artifact_path),
        "artifact_target": artifact.get("target"),
        "artifact_sample_weight_mode": artifact.get("sample_weight_mode"),
        "artifact_limit_up_sample_mode": artifact.get("limit_up_sample_mode"),
        "artifact_return_clip_enabled": artifact.get("return_clip_enabled"),
        "feature_count": len(feature_columns),
        "missing_feature_columns": missing_columns,
        "ranking": _summarize_model_ranking(scored, target=target, limit_up_threshold=limit_up_threshold),
    }


def _summarize_model_ranking(scored: pd.DataFrame, *, target: str, limit_up_threshold: float) -> Dict[str, Any]:
    baseline = _safe_mean(scored[target])
    result: Dict[str, Any] = {
        "baseline_mean_return": baseline,
        "daily_topn_performance": _summarize_daily_topn_performance(
            scored,
            target=target,
            limit_up_threshold=limit_up_threshold,
        ),
    }
    scored = scored.sort_values("pred", ascending=False)
    for pct in TOP_PCTS:
        count = max(1, int(len(scored) * pct))
        top = scored.head(count)
        mean_return = _safe_mean(top[target])
        key = f"top_{int(pct * 100)}pct"
        pct_change = pd.to_numeric(top["pct_change"], errors="coerce")
        result[key] = {
            "count": int(count),
            "mean_return": mean_return,
            "excess_return": None if mean_return is None or baseline is None else mean_return - baseline,
            "limit_like_ratio": _safe_mean((pct_change >= limit_up_threshold).astype(float)),
            "strong_move_ratio": _safe_mean((pct_change >= 7.5).astype(float)),
            "mid_1_to_7_ratio": _safe_mean(((pct_change >= 1.0) & (pct_change <= 7.0)).astype(float)),
            "negative_ratio": _safe_mean((pct_change < 0).astype(float)),
            "prev_day_limit_up_ratio": _safe_mean(pd.to_numeric(top.get("prev_day_limit_up"), errors="coerce")) if "prev_day_limit_up" in top.columns else None,
            "avg_limit_chase_failure_risk_score": _safe_mean(top.get("limit_chase_failure_risk_score")) if "limit_chase_failure_risk_score" in top.columns else None,
            "risk_bucket_distribution": _summarize_topn_risk_distribution(top, target=target)
            if "limit_chase_failure_risk_score" in top.columns
            else [],
            "daily_performance": _summarize_daily_performance(
                top,
                target=target,
                baseline_by_date=scored.groupby("trade_date")[target].mean(),
                limit_up_threshold=limit_up_threshold,
            ),
        }
    if "prev_day_limit_up" in scored.columns:
        prev_mask = pd.to_numeric(scored["prev_day_limit_up"], errors="coerce").fillna(0.0) > 0
        prev_subset = scored.loc[prev_mask]
        result["prev_day_limit_up_subset"] = {
            "count": int(len(prev_subset)),
            "mean_return": _safe_mean(prev_subset[target]) if not prev_subset.empty else None,
            "win_rate": _safe_mean((prev_subset[target] > 0).astype(float)) if not prev_subset.empty else None,
        }
    if "limit_chase_failure_risk_score" in scored.columns:
        result["limit_chase_failure_risk_buckets"] = _summarize_numeric_buckets(
            scored,
            column="limit_chase_failure_risk_score",
            target=target,
            buckets=[("low_0_1", None, 2.0), ("mid_2_4", 2.0, 4.0), ("high_4_plus", 4.0, None)],
        )
    return result


def _summarize_pct_buckets(frame: pd.DataFrame, *, target: str, limit_up_threshold: float) -> List[Dict[str, Any]]:
    buckets = [("limit_like_ge_threshold", limit_up_threshold, None)] + PCT_BUCKETS[1:]
    pct_change = pd.to_numeric(frame["pct_change"], errors="coerce")
    rows = []
    for name, lower, upper in buckets:
        mask = pd.Series(True, index=frame.index)
        if lower is not None:
            mask &= pct_change >= lower
        if upper is not None:
            mask &= pct_change < upper
        subset = frame.loc[mask]
        rows.append({
            "bucket": name,
            "count": int(len(subset)),
            "mean_return": _safe_mean(subset[target]) if not subset.empty else None,
            "win_rate": _safe_mean((subset[target] > 0).astype(float)) if not subset.empty else None,
        })
    return rows


def _summarize_topn_risk_distribution(frame: pd.DataFrame, *, target: str) -> List[Dict[str, Any]]:
    if frame.empty or "limit_chase_failure_risk_score" not in frame.columns:
        return []
    rows = _summarize_numeric_buckets(
        frame,
        column="limit_chase_failure_risk_score",
        target=target,
        buckets=[("low_0_1", None, 2.0), ("mid_2_4", 2.0, 4.0), ("high_4_plus", 4.0, None)],
    )
    total = float(len(frame))
    for row in rows:
        row["ratio"] = None if total <= 0 else row["count"] / total
    return rows


def _summarize_daily_performance(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline_by_date: pd.Series,
    limit_up_threshold: float,
) -> List[Dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for trade_date, subset in frame.groupby("trade_date", sort=True):
        baseline = _safe_float(baseline_by_date.get(trade_date))
        rows.append(
            _summarize_return_subset(
                subset,
                target=target,
                baseline=baseline,
                limit_up_threshold=limit_up_threshold,
                extra_fields={"trade_date": _format_trade_date(trade_date)},
            )
        )
    return rows


def _summarize_daily_topn_performance(
    scored: pd.DataFrame,
    *,
    target: str,
    limit_up_threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {
        f"top_{int(pct * 100)}pct": [] for pct in TOP_PCTS
    }
    if scored.empty:
        return result
    for trade_date, day_frame in scored.groupby("trade_date", sort=True):
        day_sorted = day_frame.sort_values("pred", ascending=False)
        baseline = _safe_mean(day_sorted[target])
        trade_date_text = _format_trade_date(trade_date)
        day_count = len(day_sorted)
        for pct in TOP_PCTS:
            count = max(1, int(day_count * pct))
            top = day_sorted.head(count)
            key = f"top_{int(pct * 100)}pct"
            result[key].append(
                _summarize_return_subset(
                    top,
                    target=target,
                    baseline=baseline,
                    limit_up_threshold=limit_up_threshold,
                    extra_fields={
                        "trade_date": trade_date_text,
                        "universe_count": int(day_count),
                        "top_pct": pct,
                    },
                )
            )
    return result


def _summarize_return_subset(
    subset: pd.DataFrame,
    *,
    target: str,
    baseline: Optional[float],
    limit_up_threshold: float,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pct_change = pd.to_numeric(subset["pct_change"], errors="coerce")
    target_values = pd.to_numeric(subset[target], errors="coerce")
    mean_return = _safe_mean(target_values)
    row: Dict[str, Any] = dict(extra_fields or {})
    row.update({
        "count": int(len(subset)),
        "mean_return": mean_return,
        "baseline_mean_return": baseline,
        "excess_return": None if mean_return is None or baseline is None else mean_return - baseline,
        "win_rate": _safe_mean((target_values > 0).astype(float)),
        "limit_like_ratio": _safe_mean((pct_change >= limit_up_threshold).astype(float)),
        "prev_day_limit_up_ratio": _safe_mean(pd.to_numeric(subset.get("prev_day_limit_up"), errors="coerce")) if "prev_day_limit_up" in subset.columns else None,
        "avg_limit_chase_failure_risk_score": _safe_mean(subset.get("limit_chase_failure_risk_score")) if "limit_chase_failure_risk_score" in subset.columns else None,
    })
    return row


def _format_trade_date(value: Any) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def _summarize_numeric_buckets(
    frame: pd.DataFrame,
    *,
    column: str,
    target: str,
    buckets: List[Tuple[str, Optional[float], Optional[float]]],
) -> List[Dict[str, Any]]:
    values = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for name, lower, upper in buckets:
        mask = values.notna()
        if lower is not None:
            mask &= values >= lower
        if upper is not None:
            mask &= values < upper
        subset = frame.loc[mask]
        rows.append({
            "bucket": name,
            "count": int(len(subset)),
            "mean_return": _safe_mean(subset[target]) if not subset.empty else None,
            "win_rate": _safe_mean((subset[target] > 0).astype(float)) if not subset.empty else None,
        })
    return rows


def _evaluate_actual_top3(
    database_url: str,
    *,
    start_date: date,
    end_date: date,
    limit_up_threshold: float,
) -> Dict[str, Any]:
    database_path = _sqlite_path(database_url)
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
        WITH top AS (
            SELECT r.trade_date AS d, r.ts_code, r.name, r.recommend_rank, r.pct_change
            FROM recommendation_pool_states r
            WHERE r.source_tag='今日Top3'
              AND r.recommend_rank IS NOT NULL
              AND r.trade_date >= ?
              AND r.trade_date <= ?
        ),
        cal AS (
            SELECT trade_date, ROW_NUMBER() OVER (ORDER BY trade_date) rn
            FROM market_trade_calendar
            WHERE is_open=1 AND exchange='SSE'
        ),
        joined AS (
            SELECT top.*, c.rn
            FROM top JOIN cal c ON c.trade_date=top.d
        )
        SELECT
            j.d AS trade_date,
            j.ts_code,
            j.name,
            j.recommend_rank,
            j.pct_change,
            d1.pct_chg AS next1,
            d2.pct_chg AS next2,
            d3.pct_chg AS next3
        FROM joined j
        LEFT JOIN cal c1 ON c1.rn=j.rn+1
        LEFT JOIN market_daily d1 ON d1.trade_date=c1.trade_date AND d1.ts_code=j.ts_code
        LEFT JOIN cal c2 ON c2.rn=j.rn+2
        LEFT JOIN market_daily d2 ON d2.trade_date=c2.trade_date AND d2.ts_code=j.ts_code
        LEFT JOIN cal c3 ON c3.rn=j.rn+3
        LEFT JOIN market_daily d3 ON d3.trade_date=c3.trade_date AND d3.ts_code=j.ts_code
        ORDER BY j.d, j.recommend_rank
        """
        rows = [dict(row) for row in conn.execute(query, (start_date.isoformat(), end_date.isoformat())).fetchall()]
    finally:
        conn.close()
    if not rows:
        return {"count": 0, "items": [], "by_bucket": []}
    frame = pd.DataFrame(rows)
    frame["pct_change"] = pd.to_numeric(frame["pct_change"], errors="coerce")
    by_bucket = []
    for name, mask in [
        ("limit_like", frame["pct_change"] >= limit_up_threshold),
        ("non_limit", frame["pct_change"] < limit_up_threshold),
    ]:
        subset = frame.loc[mask]
        by_bucket.append({
            "bucket": name,
            "count": int(len(subset)),
            "next1_mean_pct": _safe_mean(subset["next1"]),
            "next2_mean_pct": _safe_mean(subset["next2"]),
            "next3_mean_pct": _safe_mean(subset["next3"]),
            "next1_win_rate": _safe_mean((pd.to_numeric(subset["next1"], errors="coerce") > 0).astype(float)) if not subset.empty else None,
        })
    return {"count": int(len(rows)), "items": rows, "by_bucket": by_bucket}


def _safe_mean(values) -> Optional[float]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()

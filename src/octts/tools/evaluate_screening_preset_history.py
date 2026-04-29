from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from octts.config import get_settings
from octts.indicators.technical import build_technical_snapshot
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.stock_screener import StockScreener
from octts.tools.common import print_json
from octts.tools.modeling import load_model_artifact
from octts.tools.rebuild_large_rule_pool_compare import _load_daily_basic_rows, _load_daily_rows, safe_float


THRESHOLDS = [0.01, 0.02, 0.03]


def _threshold_key(threshold: float) -> str:
    return f">={int(threshold * 100)}%"


def _build_hit_rates(frame: pd.DataFrame, target_column: str) -> Dict[str, float]:
    return {
        _threshold_key(threshold): float((frame[target_column] >= threshold).mean()) if len(frame) else 0.0
        for threshold in THRESHOLDS
    }


def _summarize_frame(frame: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    return {
        "count": int(len(frame)),
        "avg_target_return": float(frame[target_column].mean()) if len(frame) else 0.0,
        "hit_rates": _build_hit_rates(frame, target_column),
    }


def _resolve_preset(screener: StockScreener, preset_id: str):
    for preset in screener.get_presets():
        if preset.id == preset_id:
            return preset
    available = [preset.id for preset in screener.get_presets()]
    raise RuntimeError(f"Unknown preset_id={preset_id}. Available presets: {available}")


def _build_rule_pool_for_preset(
    repo: MarketRawDataRepository,
    *,
    trade_date: Any,
    preset_id: str,
    pool_limit: int,
) -> List[str]:
    trade_date_text = trade_date.strftime("%Y%m%d")
    history_start = (trade_date - timedelta(days=120)).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=history_start, end_date=trade_date_text)
    daily_rows = _load_daily_rows(repo, trading_dates)
    basic_rows = _load_daily_basic_rows(repo, trading_dates)
    candidates: List[tuple[str, float]] = []

    for ts_code in sorted(set(daily_rows.keys()) & set(basic_rows.keys())):
        ts_code_text = str(ts_code).strip().upper()
        if ts_code_text.endswith(".BJ"):
            continue
        daily_map = daily_rows.get(ts_code, {})
        basic_map = basic_rows.get(ts_code, {})
        if trade_date_text not in daily_map or trade_date_text not in basic_map:
            continue
        available_dates = [value for value in trading_dates if value in daily_map and value in basic_map]
        if len(available_dates) < 60:
            continue
        closes = [safe_float(daily_map[value].get("close")) for value in available_dates]
        highs = [safe_float(daily_map[value].get("high")) for value in available_dates]
        lows = [safe_float(daily_map[value].get("low")) for value in available_dates]
        volumes = [safe_float(daily_map[value].get("vol")) for value in available_dates]
        if any(value is None for value in closes[-60:]) or any(value is None for value in highs[-60:]) or any(value is None for value in lows[-60:]) or any(value is None for value in volumes[-60:]):
            continue

        snapshot = build_technical_snapshot(
            pd.Series(closes[-60:]),
            pd.Series(highs[-60:]),
            pd.Series(lows[-60:]),
            pd.Series(volumes[-60:]),
        )
        basic = basic_map[trade_date_text]
        pct_change = safe_float(daily_map[trade_date_text].get("pct_chg"))
        turnover_rate = safe_float(basic.get("turnover_rate"))
        volume_ratio = safe_float(basic.get("volume_ratio"))
        market_cap = safe_float(basic.get("total_mv"))
        market_cap_yi = (market_cap / 10000.0) if market_cap is not None else None

        if preset_id == "ma60_breakout_pullback":
            if snapshot.ma60 is None or snapshot.close <= snapshot.ma60:
                continue
            if snapshot.distance_to_ma60_pct is None or snapshot.distance_to_ma60_pct < -1.5 or snapshot.distance_to_ma60_pct > 8.0:
                continue
            if snapshot.technical_score < 36:
                continue
            if snapshot.recommendation_score < 48:
                continue
            if volume_ratio is None or volume_ratio < 0.9:
                continue
            if snapshot.price_position_20d is None or snapshot.price_position_20d < 0.3 or snapshot.price_position_20d > 0.82:
                continue
            if snapshot.price_position_20d > 0.95:
                continue
            if market_cap_yi is not None and market_cap_yi > 800:
                continue
            score = float(snapshot.recommendation_score)
            score += float(snapshot.technical_score) * 0.35
            score += max(0.0, 8.0 - abs(float(snapshot.distance_to_ma60_pct or 0.0))) * 1.2
            score += float(volume_ratio or 0.0) * 2.0
            score += float(turnover_rate or 0.0) * 0.3
            if snapshot.trend_status == "bullish":
                score += 4.0
            elif snapshot.trend_status == "improving":
                score += 2.0
            if snapshot.momentum_status in {"bullish", "bullish_rising", "strong"}:
                score += 2.5
            candidates.append((ts_code_text, score))
            continue

        raise RuntimeError(f"Unsupported preset_id={preset_id}")

    ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
    return [ts_code for ts_code, _ in ordered[:pool_limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a screening preset over history with optional rerank model.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--preset-id", required=True, help="Screen preset id from StockScreener.get_presets()")
    parser.add_argument("--target", default="vs_market_1d", choices=["vs_market_1d", "vs_market_3d", "vs_market_5d"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-trade-days", type=int, default=20)
    parser.add_argument("--pool-limit", type=int, default=200)
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--artifact-path", default="", help="Optional model artifact path for rerank top picks")
    parser.add_argument("--output-daily-limit", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"})
        return

    labeled = frame[frame[args.target].notna()].copy()
    if labeled.empty:
        print_json({"evaluated": False, "reason": "no_labeled_rows", "target": args.target})
        return

    labeled["trade_date"] = pd.to_datetime(labeled["trade_date"])
    labeled = labeled.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    labeled["ts_code"] = labeled["ts_code"].astype(str).str.strip().str.upper()

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    scored = labeled.iloc[split_index:].copy().reset_index(drop=True)

    if args.start_date:
        scored = scored[scored["trade_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        scored = scored[scored["trade_date"] <= pd.Timestamp(args.end_date)]
    if scored.empty:
        print_json({"evaluated": False, "reason": "empty_scored_range"})
        return

    trade_dates = sorted({value.date() for value in scored["trade_date"]})
    if args.max_trade_days > 0:
        trade_dates = trade_dates[-args.max_trade_days:]
    scored = scored[scored["trade_date"].dt.date.isin(trade_dates)].reset_index(drop=True)

    settings = get_settings()
    screener = StockScreener(settings)
    preset = _resolve_preset(screener, args.preset_id)
    repo = MarketRawDataRepository(settings.database_url)

    artifact: Optional[Dict[str, Any]] = None
    feature_columns: List[str] = []
    if args.artifact_path:
        artifact = load_model_artifact(Path(args.artifact_path))
        feature_columns = list(artifact.get("feature_columns") or [])
        if not feature_columns or artifact.get("model") is None:
            raise RuntimeError("Artifact missing feature_columns or model")

    daily_results: List[Dict[str, Any]] = []
    pool_returns: List[float] = []
    rerank_returns: List[float] = []
    pool_hits: Dict[str, List[float]] = defaultdict(list)
    rerank_hits: Dict[str, List[float]] = defaultdict(list)

    for trade_date in trade_dates:
        candidate_codes = _build_rule_pool_for_preset(
            repo,
            trade_date=trade_date,
            preset_id=args.preset_id,
            pool_limit=args.pool_limit,
        )
        day_frame = scored[scored["trade_date"].dt.date == trade_date].copy()
        if not candidate_codes or day_frame.empty:
            continue

        pool_frame = day_frame[day_frame["ts_code"].isin(candidate_codes)].copy()
        if pool_frame.empty:
            continue

        pool_summary = _summarize_frame(pool_frame, args.target)
        pool_returns.append(float(pool_summary["avg_target_return"]))
        for key, value in pool_summary["hit_rates"].items():
            pool_hits[key].append(float(value))

        picked_codes: List[str] = []
        rerank_summary: Optional[Dict[str, Any]] = None
        if artifact is not None and feature_columns:
            usable_columns = [column for column in feature_columns if column in pool_frame.columns]
            feature_frame = pool_frame[usable_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            predictions = artifact["model"].predict(feature_frame)
            ranked = pool_frame.assign(model_score=predictions).sort_values(["model_score", args.target], ascending=[False, False])
            picked = ranked.head(args.final_pick).copy()
            picked_codes = picked["ts_code"].tolist()
            rerank_summary = _summarize_frame(picked, args.target)
            rerank_returns.append(float(rerank_summary["avg_target_return"]))
            for key, value in rerank_summary["hit_rates"].items():
                rerank_hits[key].append(float(value))

        daily_results.append(
            {
                "trade_date": trade_date.isoformat(),
                "preset_pool_count": int(len(candidate_codes)),
                "pool_overlap_count": int(len(pool_frame)),
                "pool_summary": pool_summary,
                "rerank_summary": rerank_summary,
                "picked_codes": picked_codes,
                "sample_pool_codes": candidate_codes[:10],
            }
        )

    if not daily_results:
        print_json({
            "evaluated": False,
            "reason": "no_daily_results",
            "preset_id": args.preset_id,
            "trade_days": [value.isoformat() for value in trade_dates],
        })
        return

    output = {
        "evaluated": True,
        "input": args.input,
        "preset_id": preset.id,
        "preset_name": preset.name,
        "target": args.target,
        "artifact_path": args.artifact_path or None,
        "pool_limit": int(args.pool_limit),
        "final_pick": int(args.final_pick),
        "date_range": {
            "start": min(trade_dates).isoformat() if trade_dates else None,
            "end": max(trade_dates).isoformat() if trade_dates else None,
            "trade_days": int(len(trade_dates)),
        },
        "pool_summary": {
            "days": int(len(pool_returns)),
            "avg_target_return": float(sum(pool_returns) / len(pool_returns)) if pool_returns else 0.0,
            "hit_rates": {
                key: float(sum(values) / len(values)) if values else 0.0
                for key, values in pool_hits.items()
            },
        },
        "rerank_summary": {
            "days": int(len(rerank_returns)),
            "avg_target_return": float(sum(rerank_returns) / len(rerank_returns)) if rerank_returns else 0.0,
            "hit_rates": {
                key: float(sum(values) / len(values)) if values else 0.0
                for key, values in rerank_hits.items()
            },
        }
        if artifact is not None
        else None,
        "daily_results": daily_results[: args.output_daily_limit],
        "daily_result_count": int(len(daily_results)),
    }
    print_json(output)


if __name__ == "__main__":
    main()

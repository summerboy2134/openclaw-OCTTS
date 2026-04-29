from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from octts.config import get_settings
from octts.services.enhanced_screening_scheduler import BACKFILL_TRAINING_CANDIDATE_LIMIT, TODAY_TOP_LIMIT, TOP_RECOMMENDATION_LIMIT
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.tools.common import configure_tool_logging, print_json

HORIZONS = [1, 3, 5]


COMPARISON_KEYS = ["current_flow_top3", "current_flow_top3_soft_score", "stage2_top3_without_moneyflow"]
SCORE_BINS: List[Tuple[float, float]] = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),
]


_SCREENING_RESULTS_CACHE: Dict[str, Dict[str, Any]] = {}
_RERANK_RESULT_CACHE: Dict[str, Any] = {}
_STAGE_PIPELINE_CACHE: Dict[str, Dict[str, Any]] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stage2/top3 performance with local-first data.")
    parser.add_argument("--trade-dates", required=True, help="Comma-separated YYYY-MM-DD dates")
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--rule-weight", type=float, default=0.3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--sleep-every", type=int, default=20)
    parser.add_argument("--batch-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-moneyflow-backfill", action="store_true", help="Do not fetch candidate moneyflow during evaluation")
    parser.add_argument("--output-file", help="Optional path to save full JSON payload")
    parser.add_argument("--compact", action="store_true", help="Print compact summary payload only")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "evaluate_targeted_moneyflow_stage3")
    engineer = ShortTermFeatureEngineer(settings)
    repo = MarketRawDataRepository(settings.database_url)
    trade_dates = _parse_trade_dates(args.trade_dates)

    results = []
    total_days = len(trade_dates)
    for index, trade_day in enumerate(trade_dates, start=1):
        day_started_at = time.time()
        logger.info("Evaluation start: trade_date=%s (%s/%s)", trade_day.isoformat(), index, total_days)
        try:
            result = _run_one_day(
                trade_day=trade_day,
                engineer=engineer,
                repo=repo,
                candidate_limit=max(1, int(args.candidate_limit)),
                exclude_bj=bool(args.exclude_bj),
                rule_weight=float(args.rule_weight),
                sleep_seconds=max(0.0, float(args.sleep_seconds)),
                sleep_every=max(0, int(args.sleep_every)),
                batch_sleep_seconds=max(0.0, float(args.batch_sleep_seconds)),
                force_refresh=bool(args.force_refresh),
                skip_moneyflow_backfill=bool(args.skip_moneyflow_backfill),
                logger=logger,
            )
            result["elapsed_seconds"] = round(time.time() - day_started_at, 3)
            logger.info(
                "Evaluation complete: trade_date=%s (%s/%s), elapsed_seconds=%.3f, stage1=%s, stage2=%s, top3=%s",
                trade_day.isoformat(),
                index,
                total_days,
                result["elapsed_seconds"],
                result.get("candidate_pool", {}).get("stage1_candidate_count"),
                result.get("candidate_pool", {}).get("stage2_top20_count"),
                result.get("candidate_pool", {}).get("stage2_top3_count"),
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Evaluation failed: trade_date=%s (%s/%s)", trade_day.isoformat(), index, total_days)
            results.append({
                "trade_date": trade_day.isoformat(),
                "evaluated": False,
                "error": str(exc),
                "elapsed_seconds": round(time.time() - day_started_at, 3),
            })

    payload = {
        "evaluated": any(item.get("evaluated") for item in results),
        "trade_dates": [item.isoformat() for item in trade_dates],
        "summary": _build_summary(results),
        "results": results,
    }
    compact_payload = _build_compact_payload(payload)
    logger.info("Evaluation complete: %s", payload["summary"])
    print_json(compact_payload if args.compact else payload, output_file=args.output_file)


def _run_one_day(*, trade_day: date, engineer: ShortTermFeatureEngineer, repo: MarketRawDataRepository, candidate_limit: int, exclude_bj: bool, rule_weight: float, sleep_seconds: float, sleep_every: int, batch_sleep_seconds: float, force_refresh: bool, skip_moneyflow_backfill: bool, logger) -> Dict[str, Any]:
    scheduler = engineer.scheduler
    trade_date_text = trade_day.strftime("%Y%m%d")

    snapshot_started_at = time.time()
    snapshot_path = engineer.screener.client._screening_snapshot_path(trade_date_text)
    snapshot_state = "cache_hit" if snapshot_path.exists() else "cache_miss"
    logger.info(
        "Trade date %s snapshot load start: state=%s, path=%s",
        trade_day.isoformat(),
        snapshot_state,
        snapshot_path,
    )
    market_snapshot = engineer.screener.client.get_or_build_screening_snapshot(trade_date_text)
    snapshot_reason = engineer.screener.client._screening_snapshot_invalid_reason(market_snapshot)
    snapshot_coverage = engineer.screener.client._summarize_screening_daily_coverage((market_snapshot.get("daily") or {}).values())
    logger.info(
        "Trade date %s snapshot ready: elapsed_seconds=%.3f, validity=%s, stocks=%s, basic=%s, daily=%s, ge14=%s, ge20=%s, ge60=%s",
        trade_day.isoformat(),
        time.time() - snapshot_started_at,
        "valid" if snapshot_reason is None else snapshot_reason,
        len(market_snapshot.get("stocks") or []),
        len(market_snapshot.get("daily_basic") or {}),
        len(market_snapshot.get("daily") or {}),
        snapshot_coverage.get("ge_14", 0),
        snapshot_coverage.get("ge_20", 0),
        snapshot_coverage.get("ge_60", 0),
    )

    screening_started_at = time.time()
    screening_cache_key = trade_date_text
    cached_screening_results = _SCREENING_RESULTS_CACHE.get(screening_cache_key)
    if cached_screening_results is not None:
        screening_results = cached_screening_results
        logger.info(
            "Trade date %s screening cache hit: strategies=%s, elapsed_seconds=%.3f",
            trade_day.isoformat(),
            len(screening_results),
            time.time() - screening_started_at,
        )
    else:
        logger.info("Trade date %s screening start", trade_day.isoformat())
        screening_results = scheduler._run_screening_strategies_sync_for_backfill(trade_date_text, market_snapshot=market_snapshot)
        _SCREENING_RESULTS_CACHE[screening_cache_key] = screening_results
        logger.info(
            "Trade date %s screening complete: strategies=%s, elapsed_seconds=%.3f",
            trade_day.isoformat(),
            len(screening_results),
            time.time() - screening_started_at,
        )

    candidate_started_at = time.time()
    candidate_codes = scheduler._get_top_stocks(screening_results, limit=max(candidate_limit, int(engineer.settings.screening_top_n or 20)))
    eligible_candidate_codes = scheduler._filter_out_tracked_and_holding_codes(candidate_codes)
    logger.info(
        "Trade date %s candidate pool prepared: raw=%s, eligible=%s, elapsed_seconds=%.3f",
        trade_day.isoformat(),
        len(candidate_codes),
        len(eligible_candidate_codes),
        time.time() - candidate_started_at,
    )

    stage1_candidate_codes = eligible_candidate_codes[:candidate_limit]
    if skip_moneyflow_backfill:
        backfill = {
            "candidate_codes": len(stage1_candidate_codes),
            "hit_codes": 0,
            "missing_codes": 0,
            "missing_code_samples": [],
            "fetched_rows": 0,
            "inserted_rows": 0,
            "force_refresh": force_refresh,
            "skipped": True,
        }
        logger.info(
            "Trade date %s moneyflow backfill skipped: candidate_codes=%s",
            trade_day.isoformat(),
            len(stage1_candidate_codes),
        )
    else:
        logger.info("Trade date %s moneyflow backfill start: candidate_codes=%s", trade_day.isoformat(), len(stage1_candidate_codes))
        backfill = _backfill_moneyflow(
            trade_day=trade_day,
            ts_codes=stage1_candidate_codes,
            engineer=engineer,
            repo=repo,
            sleep_seconds=sleep_seconds,
            sleep_every=sleep_every,
            batch_sleep_seconds=batch_sleep_seconds,
            force_refresh=force_refresh,
            logger=logger,
        )
        logger.info(
            "Trade date %s moneyflow backfill complete: hit_codes=%s, fetched_rows=%s, inserted_rows=%s",
            trade_day.isoformat(),
            backfill.get("hit_codes"),
            backfill.get("fetched_rows"),
            backfill.get("inserted_rows"),
        )

    rerank_started_at = time.time()
    rerank_cache_key = f"{trade_date_text}|{candidate_limit}|{exclude_bj}|{rule_weight:.6f}|skip_backfill={int(skip_moneyflow_backfill)}|force_refresh={int(force_refresh)}"
    cached_rerank_result = _RERANK_RESULT_CACHE.get(rerank_cache_key)
    if cached_rerank_result is not None:
        rerank_result = cached_rerank_result
        logger.info(
            "Trade date %s rerank cache hit: candidate_codes=%s, analysis_codes=%s, fallback_reason=%s, elapsed_seconds=%.3f",
            trade_day.isoformat(),
            len(rerank_result.candidate_codes or []),
            len(rerank_result.analysis_codes or []),
            rerank_result.fallback_reason,
            time.time() - rerank_started_at,
        )
    else:
        logger.info("Trade date %s rerank start", trade_day.isoformat())
        rerank_result = scheduler.regression_rerank_service.rank_candidates(
            screening_results,
            trade_date=trade_day,
            coarse_limit=max(candidate_limit, BACKFILL_TRAINING_CANDIDATE_LIMIT),
            analysis_limit=TOP_RECOMMENDATION_LIMIT,
            exclude_bj=exclude_bj,
            rule_weight=rule_weight,
        )
        _RERANK_RESULT_CACHE[rerank_cache_key] = rerank_result
        logger.info(
            "Trade date %s rerank complete: candidate_codes=%s, analysis_codes=%s, fallback_reason=%s, elapsed_seconds=%.3f",
            trade_day.isoformat(),
            len(rerank_result.candidate_codes or []),
            len(rerank_result.analysis_codes or []),
            rerank_result.fallback_reason,
            time.time() - rerank_started_at,
        )

    pipeline_started_at = time.time()
    pipeline_cache_key = rerank_cache_key
    cached_stage_pipeline = _STAGE_PIPELINE_CACHE.get(pipeline_cache_key)
    if cached_stage_pipeline is not None:
        stage_pipeline = cached_stage_pipeline
        logger.info(
            "Trade date %s stage pipeline cache hit: stage1=%s, stage2=%s, top3=%s, elapsed_seconds=%.3f",
            trade_day.isoformat(),
            len(stage_pipeline.get("stage1_candidate_codes") or []),
            len(stage_pipeline.get("stage2_top20_codes") or []),
            len((stage_pipeline.get("stage2_top20_codes") or [])[:TODAY_TOP_LIMIT]),
            time.time() - pipeline_started_at,
        )
    else:
        logger.info("Trade date %s stage pipeline start", trade_day.isoformat())
        stage_pipeline = scheduler._build_stage_pipeline_result(
            trade_date=trade_day,
            screening_results=screening_results,
            market_snapshot=market_snapshot,
            rerank_result=rerank_result,
            baseline_candidate_codes=eligible_candidate_codes,
        )
        _STAGE_PIPELINE_CACHE[pipeline_cache_key] = stage_pipeline
        logger.info(
            "Trade date %s stage pipeline complete: stage1=%s, stage2=%s, top3=%s, elapsed_seconds=%.3f",
            trade_day.isoformat(),
            len(stage_pipeline.get("stage1_candidate_codes") or []),
            len(stage_pipeline.get("stage2_top20_codes") or []),
            len((stage_pipeline.get("stage2_top20_codes") or [])[:TODAY_TOP_LIMIT]),
            time.time() - pipeline_started_at,
        )

    stage2_top3_codes = list((stage_pipeline.get("stage2_top20_codes") or [])[:TODAY_TOP_LIMIT])
    soft_score_top3_codes = _select_soft_score_top3_codes(
        rerank_result.metadata_by_code,
        limit=TODAY_TOP_LIMIT,
    )
    names = scheduler._build_stock_name_map(screening_results)
    compare = {
        "current_flow_top3": _pick_details(
            list(rerank_result.analysis_codes[:TODAY_TOP_LIMIT]),
            stage_pipeline.get("stage2_recommendations") or {},
            names,
            rerank_payloads=rerank_result.metadata_by_code,
        ),
        "current_flow_top3_soft_score": _pick_details(
            soft_score_top3_codes,
            stage_pipeline.get("stage2_recommendations") or {},
            names,
            rerank_payloads=rerank_result.metadata_by_code,
        ),
        "stage2_top3_without_moneyflow": _pick_details(
            stage2_top3_codes,
            stage_pipeline.get("stage2_recommendations") or {},
            names,
            rerank_payloads=rerank_result.metadata_by_code,
        ),
    }
    rerank_veto_summary = _build_rerank_veto_summary(
        rerank_result.metadata_by_code,
        names,
        analysis_limit=TODAY_TOP_LIMIT,
    )
    evaluate_started_at = time.time()
    perf = {key: _evaluate_group(repo, trade_day, value) for key, value in compare.items()}
    logger.info(
        "Trade date %s performance evaluation complete: groups=%s, elapsed_seconds=%.3f",
        trade_day.isoformat(),
        list(compare.keys()),
        time.time() - evaluate_started_at,
    )
    return {
        "trade_date": trade_day.isoformat(),
        "evaluated": True,
        "candidate_pool": {
            "raw_candidate_count": len(candidate_codes),
            "eligible_candidate_count": len(eligible_candidate_codes),
            "stage1_candidate_count": len(stage_pipeline.get("stage1_candidate_codes") or []),
            "stage2_top20_count": len(stage_pipeline.get("stage2_top20_codes") or []),
            "stage2_top3_count": len(stage2_top3_codes),
        },
        "snapshot_status": {
            "state": snapshot_state,
            "validity": "valid" if snapshot_reason is None else snapshot_reason,
            "stocks": len(market_snapshot.get("stocks") or []),
            "daily_basic": len(market_snapshot.get("daily_basic") or {}),
            "daily": len(market_snapshot.get("daily") or {}),
            "ge_14": snapshot_coverage.get("ge_14", 0),
            "ge_20": snapshot_coverage.get("ge_20", 0),
            "ge_60": snapshot_coverage.get("ge_60", 0),
        },
        "cache_hits": {
            "screening_results": cached_screening_results is not None,
            "rerank_result": cached_rerank_result is not None,
            "stage_pipeline": cached_stage_pipeline is not None,
        },
        "moneyflow_backfill": backfill,
        "rerank_veto_summary": rerank_veto_summary,
        "top3_compare": compare,
        "top3_performance": perf,
    }


def _backfill_moneyflow(*, trade_day: date, ts_codes: List[str], engineer: ShortTermFeatureEngineer, repo: MarketRawDataRepository, sleep_seconds: float, sleep_every: int, batch_sleep_seconds: float, force_refresh: bool, logger) -> Dict[str, Any]:
    client = engineer.screener.client
    trade_date_text = trade_day.strftime("%Y%m%d")
    fetched_rows = 0
    inserted_rows = 0
    hit_codes = 0
    missing = []
    for index, ts_code in enumerate(ts_codes, start=1):
        rows = client.fetch_moneyflow(ts_code, trade_date=trade_date_text)
        fetched_rows += len(rows)
        if rows:
            hit_codes += 1
            inserted_rows += repo._db.upsert_market_moneyflow_daily(rows, force_refresh=force_refresh)
        else:
            missing.append(ts_code)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if sleep_every > 0 and index % sleep_every == 0 and batch_sleep_seconds > 0:
            logger.info("Moneyflow throttle: trade_date=%s processed=%s/%s", trade_day.isoformat(), index, len(ts_codes))
            time.sleep(batch_sleep_seconds)
    return {
        "candidate_codes": len(ts_codes),
        "hit_codes": hit_codes,
        "missing_codes": len(missing),
        "missing_code_samples": missing[:20],
        "fetched_rows": fetched_rows,
        "inserted_rows": inserted_rows,
        "force_refresh": force_refresh,
    }


def _pick_details(codes: List[str], payloads: Dict[str, Dict[str, Any]], names: Dict[str, str], rerank_payloads: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    results = []
    rerank_payloads = rerank_payloads or {}
    for rank, code in enumerate(codes, start=1):
        payload = payloads.get(code) or {}
        rerank_payload = rerank_payloads.get(code) or {}
        components = payload.get("selection_reason_components") or {}
        score = _to_float(payload.get("score"))
        blend_score = _first_number(rerank_payload, components, ["blend_score"])
        model_score = _first_number(rerank_payload, components, ["model_score"])
        model_score_norm = _first_number(rerank_payload, components, ["model_score_norm"])
        rule_score = _first_number(rerank_payload, components, ["rule_score"])
        rule_score_norm = _first_number(rerank_payload, components, ["rule_score_norm"])
        soft_filter_score = _first_number(rerank_payload, components, ["soft_filter_score"])
        soft_filter_penalty = _first_number(rerank_payload, components, ["soft_filter_penalty"])
        stable_score = model_score_norm if model_score_norm is not None else blend_score if blend_score is not None else score
        results.append({
            "rank": rank,
            "ts_code": code,
            "name": payload.get("name") or rerank_payload.get("name") or names.get(code) or code,
            "selection_stage": payload.get("selection_stage"),
            "score": score,
            "blend_score": blend_score,
            "model_score": model_score,
            "model_score_norm": model_score_norm,
            "rule_score": rule_score,
            "rule_score_norm": rule_score_norm,
            "soft_filter_score": soft_filter_score,
            "soft_filter_penalty": soft_filter_penalty,
            "stable_score": stable_score,
            "structured_rank_position": payload.get("structured_rank_position"),
            "moneyflow_3d_value": _first_number(rerank_payload, payload, ["recent_3d_net_inflow", "moneyflow_3d_value"]),
            "recent_large_order_net_inflow": _first_number(rerank_payload, payload, ["recent_large_order_net_inflow"]),
            "recent_super_large_order_net_inflow": _first_number(rerank_payload, payload, ["recent_super_large_order_net_inflow"]),
            "moneyflow_summary_rows": rerank_payload.get("moneyflow_summary_rows"),
            "moneyflow_missing_for_top3": rerank_payload.get("moneyflow_missing_for_top3"),
            "moneyflow_signal_score": rerank_payload.get("moneyflow_signal_score"),
            "moneyflow_model_combo_bucket": rerank_payload.get("moneyflow_model_combo_bucket"),
            "selection_reason": payload.get("selection_reason"),
            "selection_reason_components": components,
            "selected_for_analysis": rerank_payload.get("selected_for_analysis"),
            "moneyflow_vetoed_for_top3": rerank_payload.get("moneyflow_vetoed_for_top3"),
            "model_score_vetoed_for_top3": rerank_payload.get("model_score_vetoed_for_top3"),
            "veto_reason_for_top3": rerank_payload.get("veto_reason_for_top3"),
        })
    return results


def _select_soft_score_top3_codes(metadata_by_code: Dict[str, Dict[str, Any]], *, limit: int) -> List[str]:
    ordered_items = sorted(
        (metadata_by_code or {}).values(),
        key=lambda item: (
            -float(item.get("soft_filter_score") or -10**9),
            -float(item.get("blend_score") or -10**9),
            float(item.get("rerank_pool_rank") or 10**9),
            str(item.get("ts_code") or ""),
        ),
    )
    selected: List[str] = []
    for item in ordered_items:
        ts_code = str(item.get("ts_code") or "").strip().upper()
        if not ts_code:
            continue
        selected.append(ts_code)
        if len(selected) >= limit:
            break
    return selected


def _build_rerank_veto_summary(metadata_by_code: Dict[str, Dict[str, Any]], names: Dict[str, str], *, analysis_limit: int) -> Dict[str, Any]:
    ordered_items = sorted(
        (metadata_by_code or {}).values(),
        key=lambda item: (
            float(item.get("rerank_pool_rank") or 10**9),
            str(item.get("ts_code") or ""),
        ),
    )
    counts = {
        "candidate_total": len(ordered_items),
        "selected_for_analysis": 0,
        "moneyflow": 0,
        "moneyflow_missing": 0,
        "model_score": 0,
        "other": 0,
    }
    blocked_candidates: List[Dict[str, Any]] = []
    for item in ordered_items:
        veto_reason = item.get("veto_reason_for_top3")
        if item.get("selected_for_analysis"):
            counts["selected_for_analysis"] += 1
        elif veto_reason == "moneyflow":
            counts["moneyflow"] += 1
        elif veto_reason == "model_score":
            counts["model_score"] += 1
        elif veto_reason:
            counts["other"] += 1
        elif item.get("moneyflow_missing_for_top3"):
            counts["moneyflow_missing"] += 1
        if veto_reason:
            blocked_candidates.append(
                {
                    "ts_code": item.get("ts_code"),
                    "name": item.get("name") or names.get(str(item.get("ts_code") or "")) or item.get("ts_code"),
                    "rerank_pool_rank": item.get("rerank_pool_rank"),
                    "blend_score": item.get("blend_score"),
                    "model_score": item.get("model_score"),
                    "model_score_norm": item.get("model_score_norm"),
                    "rule_score": item.get("rule_score"),
                    "rule_score_norm": item.get("rule_score_norm"),
                    "recent_3d_net_inflow": item.get("recent_3d_net_inflow"),
                    "recent_large_order_net_inflow": item.get("recent_large_order_net_inflow"),
                    "recent_super_large_order_net_inflow": item.get("recent_super_large_order_net_inflow"),
                    "moneyflow_summary_rows": item.get("moneyflow_summary_rows"),
                    "moneyflow_missing_for_top3": item.get("moneyflow_missing_for_top3"),
                    "moneyflow_signal_score": item.get("moneyflow_signal_score"),
                    "moneyflow_model_combo_bucket": item.get("moneyflow_model_combo_bucket"),
                    "veto_reason_for_top3": veto_reason,
                }
            )
    return {
        "analysis_limit": analysis_limit,
        "counts": counts,
        "blocked_candidates": blocked_candidates,
    }


def _evaluate_group(repo: MarketRawDataRepository, trade_day: date, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_stock = []
    horizon_values = {str(h): [] for h in HORIZONS}
    for item in items:
        returns = {}
        for horizon in HORIZONS:
            value = _forward_return(repo, item.get("ts_code"), trade_day, horizon)
            returns[str(horizon)] = value
            if value is not None:
                horizon_values[str(horizon)].append(value)
        per_stock.append({
            "ts_code": item.get("ts_code"),
            "name": item.get("name"),
            "score": item.get("score"),
            "blend_score": item.get("blend_score"),
            "model_score": item.get("model_score"),
            "model_score_norm": item.get("model_score_norm"),
            "rule_score": item.get("rule_score"),
            "rule_score_norm": item.get("rule_score_norm"),
            "stable_score": item.get("stable_score"),
            "returns": returns,
        })
    summary = {}
    accuracy = {}
    for horizon in HORIZONS:
        values = horizon_values[str(horizon)]
        summary[str(horizon)] = {
            "sample_size": len(values),
            "avg_return": round(mean(values), 6) if values else None,
            "positive_rate": round(sum(1 for v in values if v > 0) / len(values), 4) if values else None,
        }
        accuracy[str(horizon)] = {
            ">0": round(sum(1 for v in values if v > 0) / len(values), 4) if values else None,
            ">=1%": round(sum(1 for v in values if v >= 0.01) / len(values), 4) if values else None,
            ">=2%": round(sum(1 for v in values if v >= 0.02) / len(values), 4) if values else None,
            ">=3%": round(sum(1 for v in values if v >= 0.03) / len(values), 4) if values else None,
        }
    return {
        "summary": summary,
        "accuracy": accuracy,
        "per_stock": per_stock,
        "draggers": _build_draggers(per_stock),
        "score_stability": _build_score_stability(per_stock),
    }


def _forward_return(repo: MarketRawDataRepository, ts_code: Optional[str], trade_day: date, horizon: int) -> Optional[float]:
    if not ts_code:
        return None
    start = trade_day.strftime("%Y%m%d")
    dates = repo.list_trading_dates(start_date=start, end_date=(trade_day + timedelta(days=max(20, horizon * 4))).strftime("%Y%m%d"))
    if len(dates) <= horizon:
        return None
    entry = repo.get_daily(ts_code=ts_code, trade_date=start)
    exit_row = repo.get_daily(ts_code=ts_code, trade_date=dates[horizon])
    if not entry or not exit_row:
        return None
    try:
        entry_close = float(entry.get("close") or 0.0)
        exit_close = float(exit_row.get("close") or 0.0)
    except (TypeError, ValueError):
        return None
    if entry_close <= 0:
        return None
    return round(exit_close / entry_close - 1.0, 6)


def _build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in results if item.get("evaluated")]
    if not valid:
        return {"evaluated_days": 0, "failed_days": len(results)}
    out = {"evaluated_days": len(valid), "failed_days": len(results) - len(valid), "comparisons": {}, "accuracy": {}, "draggers": {}, "score_stability": {}}
    for key in COMPARISON_KEYS:
        out["comparisons"][key] = {}
        out["accuracy"][key] = {}
        for horizon in HORIZONS:
            values = [item.get("top3_performance", {}).get(key, {}).get("summary", {}).get(str(horizon), {}).get("avg_return") for item in valid]
            values = [float(v) for v in values if v is not None]
            out["comparisons"][key][str(horizon)] = round(mean(values), 6) if values else None
            accuracy_payloads = [item.get("top3_performance", {}).get(key, {}).get("accuracy", {}).get(str(horizon), {}) for item in valid]
            out["accuracy"][key][str(horizon)] = _mean_accuracy_payloads(accuracy_payloads)
        out["draggers"][key] = _merge_draggers(valid, key)
        out["score_stability"][key] = _merge_score_stability(valid, key)
    cache_hit_counts = {
        "screening_results": sum(1 for item in valid if item.get("cache_hits", {}).get("screening_results")),
        "rerank_result": sum(1 for item in valid if item.get("cache_hits", {}).get("rerank_result")),
        "stage_pipeline": sum(1 for item in valid if item.get("cache_hits", {}).get("stage_pipeline")),
    }
    snapshot_states = {
        "cache_hit": sum(1 for item in valid if item.get("snapshot_status", {}).get("state") == "cache_hit"),
        "cache_miss": sum(1 for item in valid if item.get("snapshot_status", {}).get("state") == "cache_miss"),
    }
    elapsed_values = [float(item.get("elapsed_seconds")) for item in valid if item.get("elapsed_seconds") is not None]
    out["cache_hit_counts"] = cache_hit_counts
    out["snapshot_states"] = snapshot_states
    out["avg_elapsed_seconds"] = round(mean(elapsed_values), 3) if elapsed_values else None
    return out


def _parse_trade_dates(raw: str) -> List[date]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("--trade-dates cannot be empty")
    return [datetime.strptime(item, "%Y-%m-%d").date() for item in values]


def _build_compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload.get("results") or []
    compact_results = []
    for item in results:
        if not item.get("evaluated"):
            compact_results.append({
                "trade_date": item.get("trade_date"),
                "evaluated": False,
                "error": item.get("error"),
            })
            continue
        compact_compare = {}
        compact_perf = {}
        for key in COMPARISON_KEYS:
            picks = item.get("top3_compare", {}).get(key, [])
            compact_compare[key] = [
                {
                    "ts_code": pick.get("ts_code"),
                    "score": pick.get("score"),
                    "blend_score": pick.get("blend_score"),
                    "model_score_norm": pick.get("model_score_norm"),
                    "rule_score_norm": pick.get("rule_score_norm"),
                    "stable_score": pick.get("stable_score"),
                }
                for pick in picks
            ]
            compact_perf[key] = {
                "summary": item.get("top3_performance", {}).get(key, {}).get("summary"),
                "accuracy": item.get("top3_performance", {}).get(key, {}).get("accuracy"),
                "draggers": item.get("top3_performance", {}).get(key, {}).get("draggers"),
                "score_stability": item.get("top3_performance", {}).get(key, {}).get("score_stability", {}).get("best_positive_rate_band_by_horizon"),
            }
        compact_results.append({
            "trade_date": item.get("trade_date"),
            "evaluated": True,
            "elapsed_seconds": item.get("elapsed_seconds"),
            "snapshot_status": item.get("snapshot_status"),
            "cache_hits": item.get("cache_hits"),
            "moneyflow_backfill": item.get("moneyflow_backfill"),
            "rerank_veto_summary": {
                "counts": (item.get("rerank_veto_summary") or {}).get("counts"),
                "blocked_candidates": ((item.get("rerank_veto_summary") or {}).get("blocked_candidates") or [])[:10],
            },
            "top3_compare": compact_compare,
            "performance_summary": compact_perf,
        })
    return {
        "evaluated": payload.get("evaluated"),
        "trade_dates": payload.get("trade_dates"),
        "summary": payload.get("summary"),
        "results": compact_results,
    }


def _build_draggers(per_stock: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        items = []
        for item in per_stock:
            value = ((item.get("returns") or {}).get(key))
            if value is None:
                continue
            items.append({
                "ts_code": item.get("ts_code"),
                "name": item.get("name"),
                "return": value,
                "stable_score": item.get("stable_score"),
                "score": item.get("score"),
                "blend_score": item.get("blend_score"),
                "model_score_norm": item.get("model_score_norm"),
                "rule_score_norm": item.get("rule_score_norm"),
            })
        items.sort(key=lambda row: float(row.get("return") or 0.0))
        out[key] = items[:5]
    return out


def _build_score_stability(per_stock: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for low, high in SCORE_BINS:
        label = f"[{low:.1f},{min(high, 1.0):.1f}{')' if high < 1.01 else ']'}"
        bucket_items = []
        for item in per_stock:
            stable_score = _to_float(item.get("stable_score"))
            if stable_score is None:
                continue
            if stable_score < low:
                continue
            if high < 1.01 and stable_score >= high:
                continue
            if high >= 1.01 and stable_score > 1.0 + 1e-9:
                continue
            bucket_items.append(item)
        buckets[label] = {
            "sample_size": len(bucket_items),
            "horizons": {
                str(h): _score_bucket_metrics(bucket_items, str(h))
                for h in HORIZONS
            },
        }
    stable_by_horizon = {}
    for horizon in HORIZONS:
        best_label = None
        best_value = None
        for label, payload in buckets.items():
            metrics = (payload.get("horizons") or {}).get(str(horizon), {})
            if int(metrics.get("sample_size") or 0) < 3:
                continue
            value = metrics.get(">0")
            if value is None:
                continue
            if best_value is None or float(value) > float(best_value):
                best_value = float(value)
                best_label = label
        stable_by_horizon[str(horizon)] = {
            "score_band": best_label,
            "positive_rate": best_value,
        }
    return {
        "buckets": buckets,
        "best_positive_rate_band_by_horizon": stable_by_horizon,
    }


def _score_bucket_metrics(items: List[Dict[str, Any]], horizon: str) -> Dict[str, Any]:
    values = [((item.get("returns") or {}).get(horizon)) for item in items]
    values = [float(v) for v in values if v is not None]
    if not values:
        return {
            "sample_size": 0,
            "avg_return": None,
            ">0": None,
            ">=1%": None,
            ">=2%": None,
            ">=3%": None,
        }
    return {
        "sample_size": len(values),
        "avg_return": round(mean(values), 6),
        ">0": round(sum(1 for v in values if v > 0) / len(values), 4),
        ">=1%": round(sum(1 for v in values if v >= 0.01) / len(values), 4),
        ">=2%": round(sum(1 for v in values if v >= 0.02) / len(values), 4),
        ">=3%": round(sum(1 for v in values if v >= 0.03) / len(values), 4),
    }


def _merge_draggers(valid: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for horizon in HORIZONS:
        rows = []
        for item in valid:
            trade_date = item.get("trade_date")
            for row in item.get("top3_performance", {}).get(key, {}).get("draggers", {}).get(str(horizon), []):
                rows.append({**row, "trade_date": trade_date})
        rows.sort(key=lambda row: float(row.get("return") or 0.0))
        out[str(horizon)] = rows[:10]
    return out


def _merge_score_stability(valid: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    bucket_values: Dict[str, Dict[str, List[float]]] = {}
    bucket_returns: Dict[str, Dict[str, List[float]]] = {}
    for item in valid:
        buckets = item.get("top3_performance", {}).get(key, {}).get("score_stability", {}).get("buckets", {})
        for label, payload in buckets.items():
            bucket_values.setdefault(label, {str(h): [] for h in HORIZONS})
            bucket_returns.setdefault(label, {str(h): [] for h in HORIZONS})
            for horizon in HORIZONS:
                metrics = (payload.get("horizons") or {}).get(str(horizon), {})
                pos = metrics.get(">0")
                avg_ret = metrics.get("avg_return")
                if pos is not None:
                    bucket_values[label][str(horizon)].append(float(pos))
                if avg_ret is not None:
                    bucket_returns[label][str(horizon)].append(float(avg_ret))
    merged_buckets = {}
    for label in sorted(bucket_values.keys()):
        merged_buckets[label] = {
            "horizons": {
                str(h): {
                    "avg_positive_rate": round(mean(bucket_values[label][str(h)]), 4) if bucket_values[label][str(h)] else None,
                    "avg_return": round(mean(bucket_returns[label][str(h)]), 6) if bucket_returns[label][str(h)] else None,
                    "sample_days": len(bucket_values[label][str(h)]),
                }
                for h in HORIZONS
            }
        }
    best_by_horizon = {}
    for horizon in HORIZONS:
        best_label = None
        best_value = None
        for label, payload in merged_buckets.items():
            metrics = (payload.get("horizons") or {}).get(str(horizon), {})
            value = metrics.get("avg_positive_rate")
            if value is None:
                continue
            if best_value is None or float(value) > float(best_value):
                best_value = float(value)
                best_label = label
        best_by_horizon[str(horizon)] = {
            "score_band": best_label,
            "avg_positive_rate": best_value,
        }
    return {
        "buckets": merged_buckets,
        "best_positive_rate_band_by_horizon": best_by_horizon,
    }


def _mean_accuracy_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [">0", ">=1%", ">=2%", ">=3%"]
    out = {}
    for key in keys:
        values = [payload.get(key) for payload in payloads if payload.get(key) is not None]
        out[key] = round(mean(float(v) for v in values), 4) if values else None
    return out


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(payload: Dict[str, Any], components: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        direct = _to_float(payload.get(key))
        if direct is not None:
            return direct
        nested = _to_float(components.get(key))
        if nested is not None:
            return nested
    return None


if __name__ == "__main__":
    main()

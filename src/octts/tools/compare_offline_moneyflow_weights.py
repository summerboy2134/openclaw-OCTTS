from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

from octts.config import get_settings
from octts.services.enhanced_screening_scheduler import BACKFILL_TRAINING_CANDIDATE_LIMIT, TOP_RECOMMENDATION_LIMIT
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.tools.common import configure_tool_logging, print_json

HORIZONS = [1, 3, 5]
TOP3 = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline stage3 moneyflow weights using only local DB data.")
    parser.add_argument("--trade-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--moneyflow-weight-grid", default="0,0.5,1.0,1.5,2.0")
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--rule-weight", type=float, default=0.3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--sleep-every", type=int, default=20)
    parser.add_argument("--batch-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--force-refresh-moneyflow", action="store_true")
    parser.add_argument("--output-file", help="Optional path to save full JSON payload")
    parser.add_argument("--compact", action="store_true", help="Print compact summary payload only")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "compare_offline_moneyflow_weights")
    engineer = ShortTermFeatureEngineer(settings)
    repo = MarketRawDataRepository(settings.database_url)
    trade_day = datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    weight_grid = _parse_weight_grid(args.moneyflow_weight_grid)

    payload = _build_base_payload(
        trade_day=trade_day,
        engineer=engineer,
        candidate_limit=max(1, int(args.candidate_limit)),
        exclude_bj=bool(args.exclude_bj),
        rule_weight=float(args.rule_weight),
    )
    stage2_recommendations = payload["stage2_recommendations"]
    stage2_top20_codes = payload["stage2_top20_codes"]
    moneyflow_backfill = _ensure_moneyflow_for_codes(
        trade_day=trade_day,
        ts_codes=stage2_top20_codes,
        engineer=engineer,
        repo=repo,
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        sleep_every=max(0, int(args.sleep_every)),
        batch_sleep_seconds=max(0.0, float(args.batch_sleep_seconds)),
        force_refresh=bool(args.force_refresh_moneyflow),
        logger=logger,
    )
    moneyflow_summary_map = repo.get_moneyflow_summaries_by_trade_date(
        ts_codes=stage2_top20_codes,
        trade_date=trade_day.strftime("%Y%m%d"),
        lookback_days=3,
    )

    weight_results = []
    for weight in weight_grid:
        recommendations = _apply_offline_stage3_weight(
            stage2_recommendations=stage2_recommendations,
            stage2_top20_codes=stage2_top20_codes,
            moneyflow_summary_map=moneyflow_summary_map,
            moneyflow_weight=weight,
        )
        top3_codes = [code for code, item in recommendations.items() if item.get("selection_stage") == "stage3_final_top3"]
        picks = _pick_details(top3_codes, recommendations)
        performance = _evaluate_group(repo, trade_day, picks)
        weight_results.append(
            {
                "moneyflow_weight": weight,
                "top3_codes": top3_codes,
                "top3": picks,
                "performance": performance,
            }
        )

    stage2_top3_codes = stage2_top20_codes[:TOP3]
    stage2_top3 = _pick_details(stage2_top3_codes, stage2_recommendations)
    stage2_performance = _evaluate_group(repo, trade_day, stage2_top3)

    result = {
        "evaluated": True,
        "trade_date": trade_day.isoformat(),
        "candidate_pool": {
            "stage1_candidate_count": len(payload["stage1_candidate_codes"]),
            "stage2_top20_count": len(stage2_top20_codes),
            "stage2_top20_codes": stage2_top20_codes,
        },
        "moneyflow_data_coverage": {
            "top20_codes": len(stage2_top20_codes),
            "covered_codes": len(moneyflow_summary_map),
            "missing_codes": [code for code in stage2_top20_codes if code not in moneyflow_summary_map],
            "backfill": moneyflow_backfill,
        },
        "stage2_baseline": {
            "top3_codes": stage2_top3_codes,
            "top3": stage2_top3,
            "performance": stage2_performance,
        },
        "weight_results": weight_results,
        "best_by_horizon": _build_best_by_horizon(weight_results),
    }
    compact_result = _build_compact_payload(result)
    logger.info("Offline moneyflow weight comparison complete: trade_date=%s", trade_day.isoformat())
    print_json(compact_result if args.compact else result, output_file=args.output_file)


def _build_base_payload(*, trade_day: date, engineer: ShortTermFeatureEngineer, candidate_limit: int, exclude_bj: bool, rule_weight: float) -> Dict[str, Any]:
    scheduler = engineer.scheduler
    trade_date_text = trade_day.strftime("%Y%m%d")
    market_snapshot = engineer.screener.client.get_or_build_screening_snapshot(trade_date_text)
    screening_results = scheduler._run_screening_strategies_sync_for_backfill(trade_date_text, market_snapshot=market_snapshot)
    candidate_codes = scheduler._get_top_stocks(screening_results, limit=max(candidate_limit, int(engineer.settings.screening_top_n or 20)))
    eligible_candidate_codes = scheduler._filter_out_tracked_and_holding_codes(candidate_codes)
    rerank_result = scheduler.regression_rerank_service.rank_candidates(
        screening_results,
        trade_date=trade_day,
        coarse_limit=max(candidate_limit, BACKFILL_TRAINING_CANDIDATE_LIMIT),
        analysis_limit=TOP_RECOMMENDATION_LIMIT,
        exclude_bj=exclude_bj,
        rule_weight=rule_weight,
    )
    stage_pipeline = scheduler._build_stage_pipeline_result(
        trade_date=trade_day,
        screening_results=screening_results,
        market_snapshot=market_snapshot,
        rerank_result=rerank_result,
        baseline_candidate_codes=eligible_candidate_codes,
    )
    return {
        "stage1_candidate_codes": list(stage_pipeline["stage1_candidate_codes"]),
        "stage2_top20_codes": list(stage_pipeline["stage2_top20_codes"]),
        "stage2_recommendations": {k: dict(v) for k, v in (stage_pipeline["stage2_recommendations"] or {}).items()},
    }


def _ensure_moneyflow_for_codes(*, trade_day: date, ts_codes: List[str], engineer: ShortTermFeatureEngineer, repo: MarketRawDataRepository, sleep_seconds: float, sleep_every: int, batch_sleep_seconds: float, force_refresh: bool, logger) -> Dict[str, Any]:
    trade_date_text = trade_day.strftime("%Y%m%d")
    target_codes = [str(code).strip() for code in ts_codes if str(code).strip()]
    if not target_codes:
        return {
            "requested_codes": 0,
            "existing_codes": 0,
            "fetched_codes": 0,
            "missing_code_samples": [],
            "fetched_rows": 0,
            "inserted_rows": 0,
            "force_refresh": force_refresh,
        }

    existing_summaries = {} if force_refresh else repo.get_moneyflow_summaries_by_trade_date(
        ts_codes=target_codes,
        trade_date=trade_date_text,
        lookback_days=3,
    )
    pending_codes = list(target_codes if force_refresh else [code for code in target_codes if code not in existing_summaries])
    client = engineer.screener.client
    fetched_rows = 0
    inserted_rows = 0
    fetched_codes = 0
    missing_codes: List[str] = []
    for index, ts_code in enumerate(pending_codes, start=1):
        rows = client.fetch_moneyflow(ts_code, trade_date=trade_date_text)
        fetched_rows += len(rows)
        if rows:
            fetched_codes += 1
            inserted_rows += repo._db.upsert_market_moneyflow_daily(rows, force_refresh=force_refresh)
        else:
            missing_codes.append(ts_code)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if sleep_every > 0 and index % sleep_every == 0 and batch_sleep_seconds > 0:
            logger.info("Moneyflow on-demand throttle: trade_date=%s processed=%s/%s", trade_day.isoformat(), index, len(pending_codes))
            time.sleep(batch_sleep_seconds)
    return {
        "requested_codes": len(target_codes),
        "existing_codes": len(existing_summaries),
        "fetched_codes": fetched_codes,
        "missing_code_samples": missing_codes[:20],
        "fetched_rows": fetched_rows,
        "inserted_rows": inserted_rows,
        "force_refresh": force_refresh,
    }


def _apply_offline_stage3_weight(*, stage2_recommendations: Dict[str, Dict[str, Any]], stage2_top20_codes: List[str], moneyflow_summary_map: Dict[str, Dict[str, Any]], moneyflow_weight: float) -> Dict[str, Dict[str, Any]]:
    recommendations = {code: dict(item) for code, item in stage2_recommendations.items()}
    scored = []
    for code in stage2_top20_codes:
        payload = recommendations.get(code)
        if not payload:
            continue
        summary = moneyflow_summary_map.get(code) or {}
        recent_3d = float(summary.get("recent_3d_net_inflow") or payload.get("moneyflow_3d_value") or 0.0)
        large = float(summary.get("recent_large_order_net_inflow") or payload.get("recent_large_order_net_inflow") or 0.0)
        super_large = float(summary.get("recent_super_large_order_net_inflow") or payload.get("recent_super_large_order_net_inflow") or 0.0)
        mf_score = 0.0
        if recent_3d > 0:
            mf_score += 1.0
        elif recent_3d < 0:
            mf_score -= 1.0
        if large > 0:
            mf_score += 0.8
        elif large < 0:
            mf_score -= 0.8
        if super_large > 0:
            mf_score += 1.0
        elif super_large < 0:
            mf_score -= 1.0
        veto = bool(payload.get("unsupported_high_position_flag", False)) and (bool(payload.get("relay_candidate_veto", False)) or large < 0 or super_large < 0)
        final_score = round(float(payload.get("score") or 0.0) + mf_score * moneyflow_weight - (1.5 if veto else 0.0), 4)
        payload.update({
            "moneyflow_3d_value": round(recent_3d, 2),
            "recent_large_order_net_inflow": round(large, 2),
            "recent_super_large_order_net_inflow": round(super_large, 2),
            "stage3_moneyflow_score": round(mf_score, 4),
            "stage3_moneyflow_veto": veto,
            "stage3_final_score": final_score,
            "score": final_score,
            "weighted_score": final_score,
        })
        scored.append((code, final_score))
    ranked = [code for code, _ in sorted(scored, key=lambda item: item[1], reverse=True)]
    pos_map = {code: idx for idx, code in enumerate(ranked, start=1)}
    top3_codes = ranked[:TOP3]
    for code in stage2_top20_codes:
        payload = recommendations.get(code)
        if not payload:
            continue
        payload["structured_rank_score"] = payload.get("stage3_final_score", payload.get("structured_rank_score"))
        payload["structured_rank_position"] = pos_map.get(code, payload.get("structured_rank_position"))
        payload["selection_stage"] = "stage3_final_top3" if code in top3_codes else "stage2_top20_pre_moneyflow"
        payload["selection_reason"] = f"stage3_final_score={float(payload.get('stage3_final_score') or 0.0):.2f}; moneyflow_score={float(payload.get('stage3_moneyflow_score') or 0.0):.2f}; moneyflow_weight={moneyflow_weight:.2f}; moneyflow_veto={bool(payload.get('stage3_moneyflow_veto', False))}"
    return dict(sorted(recommendations.items(), key=lambda item: float(item[1].get("score") or 0.0), reverse=True))


def _pick_details(codes: List[str], recommendations: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "rank": idx,
        "ts_code": code,
        "name": (recommendations.get(code) or {}).get("name") or code,
        "selection_stage": (recommendations.get(code) or {}).get("selection_stage"),
        "score": (recommendations.get(code) or {}).get("score"),
        "moneyflow_3d_value": (recommendations.get(code) or {}).get("moneyflow_3d_value"),
        "recent_large_order_net_inflow": (recommendations.get(code) or {}).get("recent_large_order_net_inflow"),
        "recent_super_large_order_net_inflow": (recommendations.get(code) or {}).get("recent_super_large_order_net_inflow"),
        "selection_reason": (recommendations.get(code) or {}).get("selection_reason"),
    } for idx, code in enumerate(codes, start=1)]


def _evaluate_group(repo: MarketRawDataRepository, trade_day: date, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for horizon in HORIZONS:
        values = [_forward_return(repo, str(item.get("ts_code") or ""), trade_day, horizon) for item in items]
        values = [value for value in values if value is not None]
        summary[str(horizon)] = {
            "sample_size": len(values),
            "avg_return": round(mean(values), 6) if values else None,
            "positive_rate": round(sum(1 for value in values if value > 0) / len(values), 4) if values else None,
        }
    return {"summary": summary}


def _forward_return(repo: MarketRawDataRepository, ts_code: str, trade_day: date, horizon: int) -> Optional[float]:
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


def _build_best_by_horizon(weight_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for horizon in HORIZONS:
        candidates = []
        for item in weight_results:
            avg_return = item.get("performance", {}).get("summary", {}).get(str(horizon), {}).get("avg_return")
            if avg_return is not None:
                candidates.append((float(avg_return), item.get("moneyflow_weight"), item.get("top3_codes")))
        if candidates:
            best = max(candidates, key=lambda x: x[0])
            result[str(horizon)] = {"avg_return": round(best[0], 6), "moneyflow_weight": best[1], "top3_codes": best[2]}
        else:
            result[str(horizon)] = None
    return result


def _parse_weight_grid(raw: str) -> List[float]:
    values = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("--moneyflow-weight-grid cannot be empty")
    return values


def _build_compact_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "evaluated": result.get("evaluated"),
        "trade_date": result.get("trade_date"),
        "moneyflow_data_coverage": result.get("moneyflow_data_coverage"),
        "stage2_baseline": {
            "top3_codes": result.get("stage2_baseline", {}).get("top3_codes"),
            "performance_summary": result.get("stage2_baseline", {}).get("performance", {}).get("summary"),
        },
        "weight_results": [
            {
                "moneyflow_weight": item.get("moneyflow_weight"),
                "top3_codes": item.get("top3_codes"),
                "performance_summary": item.get("performance", {}).get("summary"),
            }
            for item in (result.get("weight_results") or [])
        ],
        "best_by_horizon": result.get("best_by_horizon"),
    }


if __name__ == "__main__":
    main()

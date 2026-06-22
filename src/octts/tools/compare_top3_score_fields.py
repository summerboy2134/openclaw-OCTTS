from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

from octts.config import get_settings
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.enhanced_screening_scheduler import (
    BACKFILL_TRAINING_CANDIDATE_LIMIT,
    TOP_RECOMMENDATION_LIMIT,
)
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.tools.common import configure_tool_logging, print_json

HORIZONS = [1, 3, 5]
DEFAULT_SCORE_FIELDS = [
    "model_rank",
    "rerank_model_score",
    "rerank_blend_score",
    "overall_score",
    "fusion_70_30",
    "fusion_capped",
    "weighted_score",
    "structured_rank_score",
    "stage3_final_score",
]
ASCENDING_SCORE_FIELDS = {"model_rank", "rerank_pool_rank"}
CACHE_VERSION = "compare_top3_score_fields_v1"


def _is_st_name(name: Optional[str]) -> bool:
    normalized = str(name or "").upper().replace(" ", "")
    return "ST" in normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare which score field ranks Top3 better by future return and hit rate."
    )
    parser.add_argument("--trade-dates", help="Comma-separated YYYY-MM-DD dates")
    parser.add_argument("--start-date", help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", help="YYYY-MM-DD, inclusive")
    parser.add_argument("--score-fields", default=",".join(DEFAULT_SCORE_FIELDS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--exclude-st", action="store_true")
    parser.add_argument("--pool", choices=["stage2_top20", "stage1_candidate"], default="stage2_top20")
    parser.add_argument("--entry-mode", choices=["close", "next_open"], default="close")
    parser.add_argument(
        "--require-fillable-entry",
        action="store_true",
        help="When using next_open entry, exclude stocks that appear unfillable at next open (e.g. limit-up open).",
    )
    parser.add_argument(
        "--refill-unfillable",
        action="store_true",
        help="When enabled, keep scanning lower-ranked candidates to refill TopK if next-open entry appears unfillable.",
    )
    parser.add_argument(
        "--cache-dir",
        default="tmp/compare_top3_score_fields_cache",
        help="Directory for per-trade-date cached base results.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable local cache reuse for this run.")
    parser.add_argument("--force-refresh-cache", action="store_true", help="Rebuild and overwrite cache entries.")
    parser.add_argument("--output-file", help="Optional path to save full JSON payload")
    parser.add_argument("--compact", action="store_true", help="Print compact summary only")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "compare_top3_score_fields")
    repo = MarketRawDataRepository(settings.database_url)
    trade_days = _resolve_trade_days(args, repo)
    score_fields = _parse_score_fields(args.score_fields)
    top_k = max(1, int(args.top_k))

    engineer = ShortTermFeatureEngineer(settings)
    stock_universe = _load_local_stock_universe(settings.database_url)
    results = []
    for index, trade_day in enumerate(trade_days, start=1):
        logger.info(
            "Compare score fields start: trade_date=%s (%s/%s)",
            trade_day.isoformat(),
            index,
            len(trade_days),
        )
        try:
            result = _run_one_day(
                trade_day=trade_day,
                engineer=engineer,
                repo=repo,
                stock_universe=stock_universe,
                score_fields=score_fields,
                top_k=top_k,
                candidate_limit=max(1, int(args.candidate_limit)),
                exclude_bj=bool(args.exclude_bj),
                exclude_st=bool(args.exclude_st),
                pool_name=str(args.pool),
                entry_mode=str(args.entry_mode),
                require_fillable_entry=bool(args.require_fillable_entry),
                refill_unfillable=bool(args.refill_unfillable),
                cache_dir=None if bool(args.no_cache) else str(args.cache_dir),
                force_refresh_cache=bool(args.force_refresh_cache),
            )
            logger.info(
                "Compare score fields complete: trade_date=%s, pool=%s, fields=%s",
                trade_day.isoformat(),
                result.get("pool_size"),
                len(score_fields),
            )
            results.append(result)
        except Exception as exc:
            logger.exception("Compare score fields failed: trade_date=%s", trade_day.isoformat())
            results.append(
                {
                    "trade_date": trade_day.isoformat(),
                    "evaluated": False,
                    "error": str(exc),
                }
            )

    payload = {
        "evaluated": any(item.get("evaluated") for item in results),
        "trade_dates": [item.isoformat() for item in trade_days],
        "score_fields": score_fields,
        "entry_mode": str(args.entry_mode),
        "require_fillable_entry": bool(args.require_fillable_entry),
        "refill_unfillable": bool(args.refill_unfillable),
        "cache_dir": None if bool(args.no_cache) else str(args.cache_dir),
        "summary": _build_summary(results, score_fields),
        "results": results,
    }
    compact_payload = _build_compact_payload(payload)
    print_json(compact_payload if args.compact else payload, output_file=args.output_file)


def _resolve_trade_days(args: argparse.Namespace, repo: MarketRawDataRepository) -> List[date]:
    if args.trade_dates:
        return [
            datetime.strptime(item.strip(), "%Y-%m-%d").date()
            for item in str(args.trade_dates).split(",")
            if item.strip()
        ]
    if not args.start_date or not args.end_date:
        raise ValueError("Either --trade-dates or both --start-date/--end-date are required.")
    trading_dates = repo.list_trading_dates(
        start_date=args.start_date.replace("-", ""),
        end_date=args.end_date.replace("-", ""),
    )
    if not trading_dates:
        raise ValueError("No trading dates found in the requested range.")
    return [datetime.strptime(item, "%Y%m%d").date() for item in trading_dates]


def _parse_score_fields(raw: str) -> List[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or list(DEFAULT_SCORE_FIELDS)


def _build_day_cache_path(
    cache_dir: str,
    *,
    trade_day: date,
    candidate_limit: int,
    exclude_bj: bool,
) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key_payload = {
        "version": CACHE_VERSION,
        "trade_date": trade_day.isoformat(),
        "candidate_limit": int(candidate_limit),
        "exclude_bj": bool(exclude_bj),
    }
    digest = hashlib.sha1(json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{trade_day.strftime('%Y%m%d')}_{digest}.json"


def _load_day_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("cache_version") != CACHE_VERSION:
        return None
    base_payload = payload.get("base_payload")
    return base_payload if isinstance(base_payload, dict) else None


def _write_day_cache(path: Path, *, payload: Dict[str, Any]) -> None:
    cache_payload = {
        "cache_version": CACHE_VERSION,
        "saved_at": datetime.now().isoformat(),
        "base_payload": payload,
    }
    path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_result_from_cached_base(
    cached_base: Dict[str, Any],
    *,
    repo: MarketRawDataRepository,
    trade_day: date,
    score_fields: Sequence[str],
    top_k: int,
    exclude_st: bool,
    pool_name: str,
    entry_mode: str,
    require_fillable_entry: bool,
    refill_unfillable: bool,
) -> Dict[str, Any]:
    diagnostics = dict(cached_base.get("diagnostics") or {})
    stage1_candidate_codes = list(cached_base.get("stage1_candidate_codes") or [])
    stage2_top20_codes = list(cached_base.get("stage2_top20_codes") or [])
    stage1_records = {
        str(code): dict(item)
        for code, item in (cached_base.get("stage1_records") or {}).items()
        if isinstance(item, dict)
    }
    pool_codes = stage2_top20_codes if pool_name == "stage2_top20" else stage1_candidate_codes
    records = {
        code: dict(stage1_records[code])
        for code in pool_codes
        if code in stage1_records
    }
    field_results: Dict[str, Any] = {}
    for score_field in score_fields:
        ranked_codes = _rank_codes(records, score_field=score_field)
        if exclude_st:
            ranked_codes = [code for code in ranked_codes if not _is_st_name((records.get(code) or {}).get("name"))]
        picked_codes = _pick_tradeable_codes(
            repo,
            trade_day,
            ranked_codes,
            records=records,
            top_k=top_k,
            entry_mode=entry_mode,
            require_fillable_entry=require_fillable_entry,
            refill_unfillable=refill_unfillable,
        )
        picked_items = [_build_pick_item(rank + 1, records[code], score_field) for rank, code in enumerate(picked_codes)]
        field_results[score_field] = {
            "top_codes": picked_codes,
            "top_picks": picked_items,
            "performance": _evaluate_group(
                repo,
                trade_day,
                picked_codes,
                records=records,
                entry_mode=entry_mode,
                require_fillable_entry=require_fillable_entry,
            ),
        }
    return {
        "trade_date": trade_day.isoformat(),
        "evaluated": True,
        "pool_name": pool_name,
        "pool_size": len(pool_codes),
        "stage1_candidate_count": len(stage1_candidate_codes),
        "stage2_top20_count": len(stage2_top20_codes),
        "diagnostics": diagnostics,
        "field_results": field_results,
        "cache_hit": True,
    }


def _run_one_day(
    *,
    trade_day: date,
    engineer: ShortTermFeatureEngineer,
    repo: MarketRawDataRepository,
    stock_universe: Sequence[Dict[str, Any]],
    score_fields: Sequence[str],
    top_k: int,
    candidate_limit: int,
    exclude_bj: bool,
    exclude_st: bool,
    pool_name: str,
    entry_mode: str,
    require_fillable_entry: bool,
    refill_unfillable: bool,
    cache_dir: Optional[str],
    force_refresh_cache: bool,
) -> Dict[str, Any]:
    cached_base = None
    cache_path = (
        _build_day_cache_path(
            cache_dir,
            trade_day=trade_day,
            candidate_limit=candidate_limit,
            exclude_bj=exclude_bj,
        )
        if cache_dir
        else None
    )
    if cache_path and not force_refresh_cache:
        cached_base = _load_day_cache(cache_path)
    if cached_base is not None:
        return _build_result_from_cached_base(
            cached_base,
            repo=repo,
            trade_day=trade_day,
            score_fields=score_fields,
            top_k=top_k,
            exclude_st=exclude_st,
            pool_name=pool_name,
            entry_mode=entry_mode,
            require_fillable_entry=require_fillable_entry,
            refill_unfillable=refill_unfillable,
        )

    scheduler = engineer.scheduler
    trade_date_text = trade_day.strftime("%Y%m%d")
    rerank_result = scheduler.regression_rerank_service.rank_market_universe(
        trade_date=trade_day,
        candidate_limit=max(candidate_limit, BACKFILL_TRAINING_CANDIDATE_LIMIT),
        analysis_limit=TOP_RECOMMENDATION_LIMIT,
        exclude_bj=exclude_bj,
    )
    market_snapshot = _build_local_market_snapshot(
        scheduler=scheduler,
        repo=repo,
        stock_universe=stock_universe,
        trade_date=trade_date_text,
        stock_codes=rerank_result.candidate_codes,
    )
    screening_results = scheduler._build_model_candidate_screening_results(
        trade_date=trade_day,
        rerank_result=rerank_result,
        market_snapshot=market_snapshot,
    )
    baseline_candidate_codes = list(rerank_result.candidate_codes)
    eligible_candidate_codes = scheduler._filter_out_tracked_and_holding_codes(baseline_candidate_codes)
    stage_pipeline = scheduler._build_stage_pipeline_result(
        trade_date=trade_day,
        screening_results=screening_results,
        market_snapshot=market_snapshot,
        rerank_result=rerank_result,
        baseline_candidate_codes=baseline_candidate_codes,
    )

    final_recommendations = {
        code: dict(item)
        for code, item in (stage_pipeline.get("final_recommendations") or {}).items()
    }
    stage1_candidate_codes = list(stage_pipeline.get("stage1_candidate_codes") or [])
    stage2_top20_codes = list(stage_pipeline.get("stage2_top20_codes") or [])
    pool_codes = stage2_top20_codes if pool_name == "stage2_top20" else stage1_candidate_codes
    strategy_counts = {
        strategy_id: len((result.stocks or []))
        for strategy_id, result in (screening_results or {}).items()
        if result is not None
    }
    diagnostics = {
        "trade_date": trade_day.isoformat(),
        "stock_universe_size": len(stock_universe),
        "snapshot_stock_count": len(market_snapshot.get("stocks") or []),
        "snapshot_daily_basic_count": len(market_snapshot.get("daily_basic") or {}),
        "snapshot_daily_history_count": len(market_snapshot.get("daily") or {}),
        "screening_strategy_count": len(screening_results or {}),
        "screening_total_picks": sum(strategy_counts.values()),
        "screening_strategy_counts": strategy_counts,
        "candidate_count": len(baseline_candidate_codes),
        "eligible_candidate_count": len(eligible_candidate_codes),
        "rerank_candidate_count": len(rerank_result.candidate_codes or []),
        "rerank_analysis_count": len(rerank_result.analysis_codes or []),
        "rerank_fallback_reason": rerank_result.fallback_reason,
        "rerank_error_message": rerank_result.error_message,
        "stage1_candidate_count": len(stage1_candidate_codes),
        "stage2_top20_count": len(stage2_top20_codes),
        "final_recommendation_count": len(final_recommendations),
        "pool_name": pool_name,
        "pool_size": len(pool_codes),
    }

    stock_map = scheduler._build_screened_stock_map(screening_results)
    stage1_records = _build_candidate_records(
        pool_codes=stage1_candidate_codes,
        final_recommendations=final_recommendations,
        stock_map=stock_map,
        rerank_metadata=rerank_result.metadata_by_code or {},
    )
    if cache_path:
        _write_day_cache(
            cache_path,
            payload={
                "trade_date": trade_day.isoformat(),
                "diagnostics": diagnostics,
                "stage1_candidate_codes": stage1_candidate_codes,
                "stage2_top20_codes": stage2_top20_codes,
                "final_recommendations": final_recommendations,
                "stage1_records": stage1_records,
            },
        )
    records = {
        code: dict(stage1_records[code])
        for code in pool_codes
        if code in stage1_records
    }

    field_results: Dict[str, Any] = {}
    for score_field in score_fields:
        ranked_codes = _rank_codes(records, score_field=score_field)
        if exclude_st:
            ranked_codes = [code for code in ranked_codes if not _is_st_name((records.get(code) or {}).get("name"))]
        picked_codes = _pick_tradeable_codes(
            repo,
            trade_day,
            ranked_codes,
            records=records,
            top_k=top_k,
            entry_mode=entry_mode,
            require_fillable_entry=require_fillable_entry,
            refill_unfillable=refill_unfillable,
        )
        picked_items = [_build_pick_item(rank + 1, records[code], score_field) for rank, code in enumerate(picked_codes)]
        field_results[score_field] = {
            "top_codes": picked_codes,
            "top_picks": picked_items,
            "performance": _evaluate_group(
                repo,
                trade_day,
                picked_codes,
                records=records,
                entry_mode=entry_mode,
                require_fillable_entry=require_fillable_entry,
            ),
        }

    return {
        "trade_date": trade_day.isoformat(),
        "evaluated": True,
        "pool_name": pool_name,
        "pool_size": len(pool_codes),
        "stage1_candidate_count": len(stage1_candidate_codes),
        "stage2_top20_count": len(stage2_top20_codes),
        "diagnostics": diagnostics,
        "field_results": field_results,
        "cache_hit": False,
    }


def _load_local_stock_universe(database_url: str) -> List[Dict[str, Any]]:
    db = DatabaseManager(database_url)
    session = db.get_session()
    try:
        rows = session.query(MarketStockBasic).all()
        return [
            {
                "ts_code": row.ts_code,
                "symbol": row.symbol,
                "name": row.name,
                "area": row.area,
                "industry": row.industry,
                "market": row.market,
                "list_date": row.list_date.strftime("%Y%m%d") if row.list_date else None,
                "delist_date": row.delist_date.strftime("%Y%m%d") if row.delist_date else None,
                "is_hs": row.is_hs,
            }
            for row in rows
            if row.ts_code
        ]
    finally:
        session.close()



def _build_local_market_snapshot(
    *,
    scheduler: Any,
    repo: MarketRawDataRepository,
    stock_universe: Sequence[Dict[str, Any]],
    trade_date: str,
    stock_codes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    stocks = [dict(item) for item in stock_universe if isinstance(item, dict) and item.get("ts_code")]
    ts_codes = [str(stock.get("ts_code")).strip().upper() for stock in stocks if stock.get("ts_code")]
    daily_basic = repo.get_daily_basic_batch_for_trade_date(ts_codes=ts_codes, trade_date=trade_date)
    snapshot = {
        "snapshot_version": "local_db_compare_v1",
        "snapshot_type": "local_db_compare",
        "trade_date": trade_date,
        "created_at": datetime.now().isoformat(),
        "stocks": stocks,
        "daily_basic": daily_basic,
        "daily": {},
    }
    hydrate_codes = [str(code).strip().upper() for code in (stock_codes or ts_codes) if str(code).strip()]
    return scheduler._hydrate_snapshot_daily_history_for_codes(
        snapshot,
        trade_date=trade_date,
        stock_codes=hydrate_codes,
    )



def _build_candidate_records(
    *,
    pool_codes: Sequence[str],
    final_recommendations: Dict[str, Dict[str, Any]],
    stock_map: Dict[str, Any],
    rerank_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for code in pool_codes:
        recommendation = dict(final_recommendations.get(code) or {})
        stock = stock_map.get(code)
        metadata = rerank_metadata.get(code) or {}
        record = {
            "ts_code": code,
            "name": recommendation.get("name") or getattr(stock, "name", code),
            "market": recommendation.get("market") or getattr(stock, "market", None),
            "model_rank": metadata.get("rerank_pool_rank") or recommendation.get("rerank_pool_rank"),
            "rerank_pool_rank": metadata.get("rerank_pool_rank") or recommendation.get("rerank_pool_rank"),
            "rerank_model_score": metadata.get("model_score") or recommendation.get("rerank_model_score"),
            "rerank_blend_score": metadata.get("blend_score") or recommendation.get("rerank_blend_score"),
            "overall_score": recommendation.get("overall_score"),
            "weighted_score": recommendation.get("weighted_score") or recommendation.get("score"),
            "structured_rank_score": recommendation.get("structured_rank_score"),
            "stage3_final_score": recommendation.get("stage3_final_score"),
            "selection_stage": recommendation.get("selection_stage"),
            "distribution_risk_score": recommendation.get("distribution_risk_score"),
            "moneyflow_3d_value": recommendation.get("moneyflow_3d_value"),
        }
        records[code] = record
    _attach_fusion_scores(records)
    return records


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_descending(values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    valid_values = [value for value in values.values() if value is not None]
    if not valid_values:
        return {code: None for code in values}
    min_value = min(valid_values)
    max_value = max(valid_values)
    if max_value == min_value:
        return {code: 100.0 if values.get(code) is not None else None for code in values}
    return {
        code: round((float(value) - min_value) / (max_value - min_value) * 100.0, 6) if value is not None else None
        for code, value in values.items()
    }


def _rank_to_percentile(records: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    ranks = {code: _safe_float(record.get("model_rank")) for code, record in records.items()}
    valid_ranks = [rank for rank in ranks.values() if rank is not None]
    if not valid_ranks:
        return {code: None for code in records}
    min_rank = min(valid_ranks)
    max_rank = max(valid_ranks)
    if max_rank == min_rank:
        return {code: 100.0 if ranks.get(code) is not None else None for code in records}
    return {
        code: round((max_rank - float(rank)) / (max_rank - min_rank) * 100.0, 6) if rank is not None else None
        for code, rank in ranks.items()
    }


def _attach_fusion_scores(records: Dict[str, Dict[str, Any]]) -> None:
    model_values = {
        code: _safe_float(record.get("rerank_blend_score"))
        if _safe_float(record.get("rerank_blend_score")) is not None
        else _safe_float(record.get("rerank_model_score"))
        for code, record in records.items()
    }
    model_norm = _normalize_descending(model_values)
    if all(value is None for value in model_norm.values()):
        model_norm = _rank_to_percentile(records)
    overall_norm = _normalize_descending({code: _safe_float(record.get("overall_score")) for code, record in records.items()})

    for code, record in records.items():
        model_score = model_norm.get(code)
        overall_score = overall_norm.get(code)
        record["model_score_norm"] = model_score
        record["overall_score_norm"] = overall_score
        if model_score is not None and overall_score is not None:
            record["fusion_70_30"] = round(model_score * 0.7 + overall_score * 0.3, 6)
            overall_adjustment = max(-3.0, min(3.0, (overall_score - 70.0) * 0.15))
            record["fusion_capped"] = round(model_score + overall_adjustment, 6)
        else:
            record["fusion_70_30"] = None
            record["fusion_capped"] = None


def _rank_codes(records: Dict[str, Dict[str, Any]], *, score_field: str) -> List[str]:
    def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[Any, ...]:
        code, payload = item
        raw_value = payload.get(score_field)
        missing = raw_value in (None, "")
        if score_field in ASCENDING_SCORE_FIELDS:
            value = float(raw_value or 999999.0)
            return (missing, value, code)
        value = float(raw_value or float("-inf"))
        return (missing, -value, code)

    ranked = sorted(records.items(), key=sort_key)
    return [code for code, payload in ranked if payload.get(score_field) not in (None, "")]


def _build_pick_item(rank: int, record: Dict[str, Any], score_field: str) -> Dict[str, Any]:
    return {
        "rank": rank,
        "ts_code": record.get("ts_code"),
        "name": record.get("name"),
        "market": record.get("market"),
        "score_field": score_field,
        "score_value": record.get(score_field),
        "model_rank": record.get("model_rank"),
        "rerank_model_score": record.get("rerank_model_score"),
        "rerank_blend_score": record.get("rerank_blend_score"),
        "overall_score": record.get("overall_score"),
        "model_score_norm": record.get("model_score_norm"),
        "overall_score_norm": record.get("overall_score_norm"),
        "fusion_70_30": record.get("fusion_70_30"),
        "fusion_capped": record.get("fusion_capped"),
        "weighted_score": record.get("weighted_score"),
        "structured_rank_score": record.get("structured_rank_score"),
        "stage3_final_score": record.get("stage3_final_score"),
        "selection_stage": record.get("selection_stage"),
        "distribution_risk_score": record.get("distribution_risk_score"),
        "moneyflow_3d_value": record.get("moneyflow_3d_value"),
    }


def _entry_fillable_status(
    repo: MarketRawDataRepository,
    *,
    ts_code: str,
    trade_day: date,
    record: Dict[str, Any],
    entry_mode: str,
    require_fillable_entry: bool,
) -> Optional[bool]:
    if entry_mode != "next_open" or not require_fillable_entry:
        return None
    if not ts_code:
        return None
    start = trade_day.strftime("%Y%m%d")
    end = (trade_day + timedelta(days=10)).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=start, end_date=end)
    if len(trading_dates) <= 1:
        return None
    entry_bar = repo.get_daily(ts_code=ts_code, trade_date=trading_dates[1])
    if not entry_bar:
        return None
    return not _looks_unfillable_limit_up(
        entry_bar=entry_bar,
        ts_code=ts_code,
        name=record.get("name"),
        market=record.get("market"),
    )


def _pick_tradeable_codes(
    repo: MarketRawDataRepository,
    trade_day: date,
    ranked_codes: Sequence[str],
    *,
    records: Dict[str, Dict[str, Any]],
    top_k: int,
    entry_mode: str,
    require_fillable_entry: bool,
    refill_unfillable: bool,
) -> List[str]:
    if not refill_unfillable or entry_mode != "next_open" or not require_fillable_entry:
        return list(ranked_codes[:top_k])
    picked_codes: List[str] = []
    for code in ranked_codes:
        fillable = _entry_fillable_status(
            repo,
            ts_code=str(code or ""),
            trade_day=trade_day,
            record=records.get(str(code or ""), {}),
            entry_mode=entry_mode,
            require_fillable_entry=require_fillable_entry,
        )
        if fillable is False:
            continue
        picked_codes.append(code)
        if len(picked_codes) >= top_k:
            break
    return picked_codes


def _price_limit_ratio(*, ts_code: str, name: Optional[str], market: Optional[str]) -> float:
    normalized_code = str(ts_code or "").strip().upper()
    normalized_name = str(name or "").strip().upper().replace(" ", "")
    normalized_market = str(market or "").strip().upper()
    if "ST" in normalized_name:
        return 0.05
    if normalized_code.endswith(".BJ") or "BJ" in normalized_market:
        return 0.30
    if normalized_code.startswith(("300", "301")) or normalized_code.startswith("688"):
        return 0.20
    return 0.10


def _looks_unfillable_limit_up(
    *,
    entry_bar: Dict[str, Any],
    ts_code: str,
    name: Optional[str],
    market: Optional[str],
) -> bool:
    entry_open = _safe_float(entry_bar.get("open"))
    pre_close = _safe_float(entry_bar.get("pre_close"))
    high = _safe_float(entry_bar.get("high"))
    low = _safe_float(entry_bar.get("low"))
    close = _safe_float(entry_bar.get("close"))
    pct_chg = _safe_float(entry_bar.get("pct_chg"))
    if entry_open in (None, 0.0) or pre_close in (None, 0.0):
        return False
    limit_ratio = _price_limit_ratio(ts_code=ts_code, name=name, market=market)
    limit_price = float(pre_close) * (1.0 + limit_ratio)
    open_at_limit = float(entry_open) >= limit_price * 0.999
    board_locked = (
        high is not None
        and low is not None
        and close is not None
        and abs(float(high) - float(low)) < 1e-6
        and abs(float(close) - float(high)) < 1e-6
    )
    pct_near_limit = pct_chg is not None and float(pct_chg) >= limit_ratio * 100.0 - 0.2
    return bool(open_at_limit and (board_locked or pct_near_limit))


def _evaluate_group(
    repo: MarketRawDataRepository,
    trade_day: date,
    picked_codes: Sequence[str],
    *,
    records: Dict[str, Dict[str, Any]],
    entry_mode: str,
    require_fillable_entry: bool,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for horizon in HORIZONS:
        evaluations = [
            _forward_return(
                repo,
                ts_code=str(code or ""),
                trade_day=trade_day,
                horizon=horizon,
                record=records.get(str(code or ""), {}),
                entry_mode=entry_mode,
                require_fillable_entry=require_fillable_entry,
            )
            for code in picked_codes
        ]
        returns = [
            _safe_float(item.get("return"))
            for item in evaluations
            if _safe_float(item.get("return")) is not None
        ]
        fillable_values = [item.get("fillable") for item in evaluations if item.get("fillable") is not None]
        fillable_count = sum(1 for value in fillable_values if bool(value))
        summary[str(horizon)] = {
            "sample_size": len(returns),
            "avg_return": round(mean(returns), 6) if returns else None,
            "positive_rate": round(sum(1 for value in returns if value > 0) / len(returns), 4) if returns else None,
            "fillable_count": fillable_count,
            "fillable_rate": round(fillable_count / len(fillable_values), 4) if fillable_values else None,
        }
    return {"summary": summary}


def _forward_return(
    repo: MarketRawDataRepository,
    *,
    ts_code: str,
    trade_day: date,
    horizon: int,
    record: Dict[str, Any],
    entry_mode: str,
    require_fillable_entry: bool,
) -> Dict[str, Any]:
    if not ts_code:
        return {"return": None, "fillable": None}
    start = trade_day.strftime("%Y%m%d")
    end = (trade_day + timedelta(days=max(20, horizon * 4))).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=start, end_date=end)
    if entry_mode == "next_open":
        if len(trading_dates) <= horizon:
            return {"return": None, "fillable": None}
        entry_trade_date = trading_dates[1]
    else:
        if len(trading_dates) <= horizon:
            return {"return": None, "fillable": None}
        entry_trade_date = trading_dates[0]
    entry = repo.get_daily(ts_code=ts_code, trade_date=entry_trade_date)
    if not entry:
        return {"return": None, "fillable": None}
    if entry_mode == "next_open":
        fillable = not _looks_unfillable_limit_up(
            entry_bar=entry,
            ts_code=ts_code,
            name=record.get("name"),
            market=record.get("market"),
        )
        if require_fillable_entry and not fillable:
            return {"return": None, "fillable": False}
        entry_price = entry.get("open")
        entry_close = entry.get("close")
        if entry_price in (None, 0) or entry_close is None:
            return {"return": None, "fillable": fillable}
        compounded = float(entry_close) / float(entry_price)
        pct_dates = trading_dates[2:horizon + 1]
    else:
        fillable = None
        entry_price = entry.get("close")
        if entry_price in (None, 0):
            return {"return": None, "fillable": fillable}
        compounded = 1.0
        pct_dates = trading_dates[1:horizon + 1]
    for trade_date_text in pct_dates:
        row = repo.get_daily(ts_code=ts_code, trade_date=trade_date_text)
        if not row:
            return {"return": None, "fillable": fillable}
        try:
            pct_chg = float(row.get("pct_chg"))
        except (TypeError, ValueError):
            return {"return": None, "fillable": fillable}
        compounded *= 1.0 + pct_chg / 100.0
    return {
        "return": round(compounded - 1.0, 6),
        "fillable": fillable,
    }


def _build_summary(results: List[Dict[str, Any]], score_fields: Sequence[str]) -> Dict[str, Any]:
    summary_by_field: Dict[str, Any] = {}
    for field in score_fields:
        horizon_metrics: Dict[str, Dict[str, List[float]]] = {
            str(h): {"avg_return": [], "positive_rate": [], "fillable_rate": []} for h in HORIZONS
        }
        top_code_examples: List[Dict[str, Any]] = []
        evaluated_days = 0
        for item in results:
            if not item.get("evaluated"):
                continue
            field_result = (item.get("field_results") or {}).get(field) or {}
            if not field_result:
                continue
            evaluated_days += 1
            if len(top_code_examples) < 5:
                top_code_examples.append(
                    {
                        "trade_date": item.get("trade_date"),
                        "top_codes": field_result.get("top_codes") or [],
                    }
                )
            performance_summary = ((field_result.get("performance") or {}).get("summary") or {})
            for horizon in HORIZONS:
                metrics = performance_summary.get(str(horizon)) or {}
                avg_return = metrics.get("avg_return")
                positive_rate = metrics.get("positive_rate")
                fillable_rate = metrics.get("fillable_rate")
                if avg_return is not None:
                    horizon_metrics[str(horizon)]["avg_return"].append(float(avg_return))
                if positive_rate is not None:
                    horizon_metrics[str(horizon)]["positive_rate"].append(float(positive_rate))
                if fillable_rate is not None:
                    horizon_metrics[str(horizon)]["fillable_rate"].append(float(fillable_rate))
        summary_by_field[field] = {
            "evaluated_days": evaluated_days,
            "horizons": {
                horizon: {
                    "avg_return": round(mean(values["avg_return"]), 6) if values["avg_return"] else None,
                    "positive_rate": round(mean(values["positive_rate"]), 4) if values["positive_rate"] else None,
                    "fillable_rate": round(mean(values["fillable_rate"]), 4) if values["fillable_rate"] else None,
                }
                for horizon, values in horizon_metrics.items()
            },
            "sample_top_codes": top_code_examples,
        }

    best_by_horizon: Dict[str, Any] = {}
    for horizon in HORIZONS:
        horizon_key = str(horizon)
        best_return_field = None
        best_return_value = None
        best_hit_field = None
        best_hit_value = None
        for field in score_fields:
            metrics = (summary_by_field.get(field) or {}).get("horizons", {}).get(horizon_key, {})
            avg_return = metrics.get("avg_return")
            positive_rate = metrics.get("positive_rate")
            if avg_return is not None and (best_return_value is None or avg_return > best_return_value):
                best_return_field = field
                best_return_value = avg_return
            if positive_rate is not None and (best_hit_value is None or positive_rate > best_hit_value):
                best_hit_field = field
                best_hit_value = positive_rate
        best_by_horizon[horizon_key] = {
            "best_avg_return_field": best_return_field,
            "best_avg_return": best_return_value,
            "best_positive_rate_field": best_hit_field,
            "best_positive_rate": best_hit_value,
        }
    return {
        "summary_by_field": summary_by_field,
        "best_by_horizon": best_by_horizon,
    }


def _build_compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary") or {}
    summary_by_field = (summary.get("summary_by_field") or {})
    compact_rows = []
    for field, metrics in summary_by_field.items():
        row = {
            "score_field": field,
            "evaluated_days": metrics.get("evaluated_days"),
        }
        for horizon in HORIZONS:
            horizon_metrics = (metrics.get("horizons") or {}).get(str(horizon), {})
            row[f"avg_return_{horizon}d"] = horizon_metrics.get("avg_return")
            row[f"positive_rate_{horizon}d"] = horizon_metrics.get("positive_rate")
            row[f"fillable_rate_{horizon}d"] = horizon_metrics.get("fillable_rate")
        compact_rows.append(row)
    compact_rows.sort(key=lambda item: item.get("avg_return_3d") or float("-inf"), reverse=True)
    compact_diagnostics = []
    for result in payload.get("results") or []:
        diagnostics = dict(result.get("diagnostics") or {})
        compact_diagnostics.append(
            {
                "trade_date": result.get("trade_date"),
                "evaluated": result.get("evaluated"),
                "cache_hit": result.get("cache_hit"),
                "pool_name": result.get("pool_name"),
                "pool_size": result.get("pool_size"),
                "candidate_count": diagnostics.get("candidate_count"),
                "eligible_candidate_count": diagnostics.get("eligible_candidate_count"),
                "rerank_candidate_count": diagnostics.get("rerank_candidate_count"),
                "stage1_candidate_count": diagnostics.get("stage1_candidate_count"),
                "stage2_top20_count": diagnostics.get("stage2_top20_count"),
                "final_recommendation_count": diagnostics.get("final_recommendation_count"),
                "rerank_fallback_reason": diagnostics.get("rerank_fallback_reason"),
                "diagnosis": _diagnose_empty_pool(diagnostics),
            }
        )
    return {
        "evaluated": payload.get("evaluated"),
        "trade_dates": payload.get("trade_dates"),
        "entry_mode": payload.get("entry_mode"),
        "require_fillable_entry": payload.get("require_fillable_entry"),
        "refill_unfillable": payload.get("refill_unfillable"),
        "best_by_horizon": summary.get("best_by_horizon"),
        "diagnostics": compact_diagnostics,
        "score_field_rows": compact_rows,
    }



def _diagnose_empty_pool(diagnostics: Dict[str, Any]) -> str:
    if not diagnostics:
        return "no_diagnostics"
    if int(diagnostics.get("screening_total_picks") or 0) <= 0:
        return "no_screening_picks"
    if int(diagnostics.get("candidate_count") or 0) <= 0:
        return "no_candidates_after_strategy_merge"
    if int(diagnostics.get("eligible_candidate_count") or 0) <= 0:
        return "all_candidates_filtered_by_tracking_or_holding"
    if int(diagnostics.get("rerank_candidate_count") or 0) <= 0:
        fallback_reason = diagnostics.get("rerank_fallback_reason")
        return f"rerank_empty:{fallback_reason}" if fallback_reason else "rerank_empty"
    if int(diagnostics.get("stage1_candidate_count") or 0) <= 0:
        return "stage1_empty"
    if int(diagnostics.get("stage2_top20_count") or 0) <= 0:
        return "stage2_empty"
    if int(diagnostics.get("final_recommendation_count") or 0) <= 0:
        return "final_recommendations_empty"
    if int(diagnostics.get("pool_size") or 0) <= 0:
        return "selected_pool_empty"
    return "ok"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

from octts.config import get_settings
from octts.services.enhanced_screening_scheduler import BACKFILL_TRAINING_CANDIDATE_LIMIT, TOP_RECOMMENDATION_LIMIT, TODAY_TOP_LIMIT
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.tools.common import configure_tool_logging, print_json

H = [1, 3, 5]
PRESETS = [
    {"name": "baseline", "risk": 0.0, "theme": 0.0, "cont": 0.0, "veto": "none", "penalty": 0.0},
    {"name": "risk_light", "risk": 0.5, "theme": 0.0, "cont": 0.0, "veto": "none", "penalty": 0.0},
    {"name": "risk_heavy", "risk": 1.0, "theme": 0.0, "cont": 0.0, "veto": "none", "penalty": 0.0},
    {"name": "support_light", "risk": 0.0, "theme": 0.5, "cont": 0.0, "veto": "none", "penalty": 0.0},
    {"name": "combo_light", "risk": 0.5, "theme": 0.5, "cont": 0.25, "veto": "none", "penalty": 0.0},
    {"name": "soft_veto_combo", "risk": 0.8, "theme": 0.8, "cont": 0.3, "veto": "soft", "penalty": 1.5},
    {"name": "strict_veto_combo", "risk": 1.0, "theme": 1.0, "cont": 0.5, "veto": "strict", "penalty": 2.0},
]
SR: Dict[str, Dict[str, Any]] = {}
RR: Dict[str, Any] = {}
SP: Dict[str, Dict[str, Any]] = {}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare stage2 Top20→Top3 cut variants")
    p.add_argument("--trade-dates", required=True)
    p.add_argument("--candidate-limit", type=int, default=200)
    p.add_argument("--exclude-bj", action="store_true")
    p.add_argument("--rule-weight", type=float, default=0.1)
    p.add_argument("--rule-weights", default="")
    p.add_argument("--scheme-names", default="")
    p.add_argument("--output-file", default="")
    p.add_argument("--compact", action="store_true")
    a = p.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "compare_stage2_top3_cut_variants")
    engineer = ShortTermFeatureEngineer(settings)
    repo = MarketRawDataRepository(settings.database_url)
    trade_dates = [datetime.strptime(x.strip(), "%Y-%m-%d").date() for x in a.trade_dates.split(",") if x.strip()]
    rule_weights = parse_rule_weights(a.rule_weights, default=float(a.rule_weight))
    schemes = [dict(x) for x in PRESETS if not a.scheme_names or x["name"] in {v.strip() for v in a.scheme_names.split(",") if v.strip()}]
    if len(rule_weights) > 1:
        payload = evaluate_rule_weights(
            trade_dates=trade_dates,
            engineer=engineer,
            repo=repo,
            candidate_limit=a.candidate_limit,
            exclude_bj=a.exclude_bj,
            rule_weights=rule_weights,
            compact=a.compact,
            logger=logger,
        )
        print_json(payload, output_file=a.output_file or None)
        return
    daily = []
    for i, d in enumerate(trade_dates, start=1):
        logger.info("Compare stage2 cut variants: trade_date=%s (%s/%s)", d.isoformat(), i, len(trade_dates))
        top20 = build_top20(d, engineer, a.candidate_limit, a.exclude_bj, a.rule_weight)
        scheme_results = []
        for s in schemes:
            top3 = apply_scheme(top20, s)
            scheme_results.append({"scheme": s, "top3": top3, "performance": eval_group(repo, d, top3)})
        daily.append({"trade_date": d.isoformat(), "top20": ([] if a.compact else top20), "scheme_results": scheme_results})
    payload = {"evaluated": bool(daily), "trade_dates": [d.isoformat() for d in trade_dates], "schemes": schemes, "summary": summary(daily, schemes), "daily_results": compact_days(daily) if a.compact else daily}
    print_json(payload, output_file=a.output_file or None)


def evaluate_rule_weights(*, trade_dates: List[date], engineer: ShortTermFeatureEngineer, repo: MarketRawDataRepository, candidate_limit: int, exclude_bj: bool, rule_weights: List[float], compact: bool, logger) -> Dict[str, Any]:
    runs = []
    total_weights = len(rule_weights)
    for weight_index, rule_weight in enumerate(rule_weights, start=1):
        logger.info("Rule-weight batch start: rule_weight=%.4f (%s/%s)", rule_weight, weight_index, total_weights)
        daily = []
        for i, d in enumerate(trade_dates, start=1):
            logger.info("Rule-weight batch day start: rule_weight=%.4f trade_date=%s (%s/%s)", rule_weight, d.isoformat(), i, len(trade_dates))
            top20 = build_top20(d, engineer, candidate_limit, exclude_bj, rule_weight, logger=logger)
            logger.info("Rule-weight batch top20 ready: rule_weight=%.4f trade_date=%s top20_count=%s", rule_weight, d.isoformat(), len(top20))
            top3 = [{
                "rank": idx,
                "ts_code": item.get("ts_code"),
                "name": item.get("name"),
                "base_score": item.get("base"),
                "blend_score": item.get("blend_score"),
                "model_score_norm": item.get("model_score_norm"),
                "rule_score_norm": item.get("rule_score_norm"),
            } for idx, item in enumerate(top20[:TODAY_TOP_LIMIT], start=1)]
            perf = eval_group(repo, d, top3)
            logger.info("Rule-weight batch performance ready: rule_weight=%.4f trade_date=%s top3_count=%s", rule_weight, d.isoformat(), len(top3))
            daily.append({"trade_date": d.isoformat(), "top3": top3, "performance": perf})
        runs.append({
            "rule_weight": rule_weight,
            "summary": summarize_rule_weight_run(daily),
            "daily_results": compact_rule_days(daily) if compact else daily,
        })
        logger.info("Rule-weight batch complete: rule_weight=%.4f summary=%s", rule_weight, runs[-1]["summary"])
    return {
        "evaluated": bool(runs),
        "trade_dates": [d.isoformat() for d in trade_dates],
        "rule_weights": rule_weights,
        "best_by_avg_return": pick_best_rule_weight(runs, "avg_return"),
        "best_by_positive_rate": pick_best_rule_weight(runs, "avg_positive_rate"),
        "runs": runs,
    }


def build_top20(trade_day: date, engineer: ShortTermFeatureEngineer, candidate_limit: int, exclude_bj: bool, rule_weight: float, logger=None) -> List[Dict[str, Any]]:
    scheduler = engineer.scheduler
    td = trade_day.strftime("%Y%m%d")
    if logger:
        logger.info("build_top20 start: trade_date=%s rule_weight=%.4f", trade_day.isoformat(), rule_weight)

    pool_top20 = _load_top20_from_recommendation_pool(trade_day, engineer, rule_weight, logger=logger)
    if pool_top20:
        if logger:
            logger.info("build_top20 recommendation_pool hit: trade_date=%s rule_weight=%.4f top20_count=%s", trade_day.isoformat(), rule_weight, len(pool_top20))
        return pool_top20

    snapshot = engineer.screener.client.get_or_build_screening_snapshot(td)
    if logger:
        logger.info("build_top20 snapshot ready: trade_date=%s rule_weight=%.4f", trade_day.isoformat(), rule_weight)
    sr = SR.get(td)
    if sr is None:
        if logger:
            logger.info("build_top20 screening cache miss: trade_date=%s", trade_day.isoformat())
        sr = scheduler._run_screening_strategies_sync_for_backfill(td, market_snapshot=snapshot)
        SR[td] = sr
    elif logger:
        logger.info("build_top20 screening cache hit: trade_date=%s", trade_day.isoformat())
    candidates = scheduler._get_top_stocks(sr, limit=max(candidate_limit, int(engineer.settings.screening_top_n or 20)))
    eligible = scheduler._filter_out_tracked_and_holding_codes(candidates)
    rk = f"{td}|{candidate_limit}|{exclude_bj}|{rule_weight:.6f}"
    rr = RR.get(rk)
    if rr is None:
        if logger:
            logger.info("build_top20 rerank cache miss: trade_date=%s rule_weight=%.4f", trade_day.isoformat(), rule_weight)
        rr = scheduler.regression_rerank_service.rank_candidates(sr, trade_date=trade_day, coarse_limit=max(candidate_limit, BACKFILL_TRAINING_CANDIDATE_LIMIT), analysis_limit=TOP_RECOMMENDATION_LIMIT, exclude_bj=exclude_bj, rule_weight=rule_weight)
        RR[rk] = rr
    elif logger:
        logger.info("build_top20 rerank cache hit: trade_date=%s rule_weight=%.4f", trade_day.isoformat(), rule_weight)
    sp = SP.get(rk)
    if sp is None:
        if logger:
            logger.info("build_top20 stage pipeline cache miss: trade_date=%s rule_weight=%.4f", trade_day.isoformat(), rule_weight)
        sp = scheduler._build_stage_pipeline_result(trade_date=trade_day, screening_results=sr, market_snapshot=snapshot, rerank_result=rr, baseline_candidate_codes=eligible)
        SP[rk] = sp
        _persist_recommendation_pool_from_stage_pipeline(
            trade_day=trade_day,
            engineer=engineer,
            screening_results=sr,
            stage_pipeline=sp,
            rerank_result=rr,
            candidate_codes=eligible,
            logger=logger,
        )
    elif logger:
        logger.info("build_top20 stage pipeline cache hit: trade_date=%s rule_weight=%.4f", trade_day.isoformat(), rule_weight)
    recs = sp.get("stage2_recommendations") or {}
    out = []
    for code in list(sp.get("stage2_top20_codes") or []):
        p = recs.get(code) or {}
        c = dict(p.get("selection_reason_components") or {})
        out.append({
            "ts_code": code,
            "name": p.get("name") or code,
            "base": num(p.get("score")),
            "blend_score": first_num(p, c, ["blend_score"]),
            "model_score_norm": first_num(p, c, ["model_score_norm"]),
            "rule_score_norm": first_num(p, c, ["rule_score_norm"]),
            "pos": intval(p.get("structured_rank_position")),
            "risk": first_num(p, c, ["ranking_risk_penalty", "risk_penalty"]),
            "theme": first_num(p, c, ["theme_support_score"]),
            "cont": first_num(p, c, ["continuation_bias_score", "continuation_bias"]),
            "high": bool(p.get("unsupported_high_position_flag", False)),
            "relay": bool(p.get("relay_candidate_veto", False)),
        })
    return out


def apply_scheme(top20: List[Dict[str, Any]], scheme: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranked = []
    for x in top20:
        veto = (x["high"] or x["relay"]) if scheme["veto"] == "soft" else (x["high"] and x["relay"]) if scheme["veto"] == "strict" else False
        score = float(x.get("base") or 0.0) - float(x.get("risk") or 0.0) * float(scheme["risk"]) + float(x.get("theme") or 0.0) * float(scheme["theme"]) + float(x.get("cont") or 0.0) * float(scheme["cont"]) - (float(scheme["penalty"]) if veto else 0.0)
        ranked.append({**x, "adjusted": round(score, 6), "vetoed": veto})
    ranked.sort(key=lambda x: (-float(x.get("adjusted") or 0.0), int(x.get("pos") or 9999), str(x.get("ts_code") or "")))
    return [{"rank": i, "ts_code": x.get("ts_code"), "name": x.get("name"), "base_score": x.get("base"), "adjusted_score": x.get("adjusted"), "risk": x.get("risk"), "theme": x.get("theme"), "cont": x.get("cont"), "high": x.get("high"), "relay": x.get("relay"), "veto_triggered": x.get("vetoed")} for i, x in enumerate(ranked[:TODAY_TOP_LIMIT], start=1)]


def eval_group(repo: MarketRawDataRepository, trade_day: date, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = {str(h): [] for h in H}
    per_stock = []
    for item in items:
        returns = {}
        for h in H:
            v = fwd(repo, item.get("ts_code"), trade_day, h)
            returns[str(h)] = v
            if v is not None:
                vals[str(h)].append(v)
        per_stock.append({"ts_code": item.get("ts_code"), "name": item.get("name"), "returns": returns})
    summary = {str(h): {"sample_size": len(vals[str(h)]), "avg_return": round(mean(vals[str(h)]), 6) if vals[str(h)] else None, "positive_rate": round(sum(1 for v in vals[str(h)] if v > 0) / len(vals[str(h)]), 4) if vals[str(h)] else None} for h in H}
    return {"summary": summary, "per_stock": per_stock}


def fwd(repo: MarketRawDataRepository, ts_code: Optional[str], trade_day: date, horizon: int) -> Optional[float]:
    if not ts_code:
        return None
    start = trade_day.strftime("%Y%m%d")
    dates = repo.list_trading_dates(start_date=start, end_date=(trade_day + timedelta(days=max(20, horizon * 4))).strftime("%Y%m%d"))
    if len(dates) <= horizon:
        return None
    entry, exit_row = repo.get_daily(ts_code=ts_code, trade_date=start), repo.get_daily(ts_code=ts_code, trade_date=dates[horizon])
    if not entry or not exit_row:
        return None
    try:
        a, b = float(entry.get("close") or 0.0), float(exit_row.get("close") or 0.0)
    except (TypeError, ValueError):
        return None
    return None if a <= 0 else round(b / a - 1.0, 6)


def summary(days: List[Dict[str, Any]], schemes: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"evaluated_days": len(days), "scheme_summaries": {}, "best_by_avg_return": {}, "best_by_positive_rate": {}, "win_days_vs_baseline": {}}
    for s in schemes:
        out["scheme_summaries"][s["name"]] = {}
        for h in H:
            rs, ps = [], []
            for day in days:
                perf = (((res(day, s["name"]) or {}).get("performance") or {}).get("summary") or {}).get(str(h), {})
                if perf.get("avg_return") is not None:
                    rs.append(float(perf["avg_return"]))
                if perf.get("positive_rate") is not None:
                    ps.append(float(perf["positive_rate"]))
            out["scheme_summaries"][s["name"]][str(h)] = {"avg_return": round(mean(rs), 6) if rs else None, "avg_positive_rate": round(mean(ps), 4) if ps else None, "sample_days": len(rs)}
    for h in H:
        br = max((s["name"] for s in schemes if out["scheme_summaries"][s["name"]][str(h)]["avg_return"] is not None), key=lambda n: out["scheme_summaries"][n][str(h)]["avg_return"], default=None)
        bp = max((s["name"] for s in schemes if out["scheme_summaries"][s["name"]][str(h)]["avg_positive_rate"] is not None), key=lambda n: out["scheme_summaries"][n][str(h)]["avg_positive_rate"], default=None)
        out["best_by_avg_return"][str(h)] = {"scheme": br, "value": None if br is None else out["scheme_summaries"][br][str(h)]["avg_return"]}
        out["best_by_positive_rate"][str(h)] = {"scheme": bp, "value": None if bp is None else out["scheme_summaries"][bp][str(h)]["avg_positive_rate"]}
        out["win_days_vs_baseline"][str(h)] = {s["name"]: sum(1 for day in days if better(day, "baseline", s["name"], str(h))) for s in schemes if s["name"] != "baseline"}
    return out


def better(day: Dict[str, Any], a: str, b: str, h: str) -> bool:
    va = ((((res(day, a) or {}).get("performance") or {}).get("summary") or {}).get(h, {}) or {}).get("avg_return")
    vb = ((((res(day, b) or {}).get("performance") or {}).get("summary") or {}).get(h, {}) or {}).get("avg_return")
    return va is not None and vb is not None and float(vb) > float(va)


def res(day: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for item in day.get("scheme_results") or []:
        if ((item.get("scheme") or {}).get("name")) == name:
            return item
    return None


def compact_days(days: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"trade_date": day.get("trade_date"), "scheme_results": [{"scheme": (r.get("scheme") or {}).get("name"), "top3_codes": [x.get("ts_code") for x in r.get("top3") or []], "performance_summary": ((r.get("performance") or {}).get("summary"))} for r in day.get("scheme_results") or []]} for day in days]


def _persist_recommendation_pool_from_stage_pipeline(*, trade_day: date, engineer: ShortTermFeatureEngineer, screening_results: Dict[str, Any], stage_pipeline: Dict[str, Any], rerank_result: Any, candidate_codes: List[str], logger=None) -> None:
    existing_states = engineer.store.load_recommendation_pool_state(trade_date=trade_day) or []
    if existing_states:
        if logger:
            logger.info("build_top20 recommendation_pool persist skipped: trade_date=%s reason=already_exists state_count=%s", trade_day.isoformat(), len(existing_states))
        return
    final_recommendations = dict(stage_pipeline.get("final_recommendations") or {})
    stage2_recommendations = dict(stage_pipeline.get("stage2_recommendations") or {})
    final_recommendations.update(stage2_recommendations)
    pool_states = engineer.scheduler._build_recommendation_pool_states(
        trade_date=trade_day,
        screening_results=screening_results,
        final_recommendations=final_recommendations,
        candidate_codes=list(stage_pipeline.get("stage1_candidate_codes") or candidate_codes),
        rerank_metadata=getattr(rerank_result, "metadata_by_code", {}) or {},
    )
    if not pool_states:
        if logger:
            logger.info("build_top20 recommendation_pool persist skipped: trade_date=%s reason=no_pool_states", trade_day.isoformat())
        return
    persisted = engineer.store.upsert_recommendation_pool_states(pool_states)
    if logger:
        logger.info("build_top20 recommendation_pool persisted: trade_date=%s state_count=%s persisted_count=%s", trade_day.isoformat(), len(pool_states), len(persisted or []))


def _load_top20_from_recommendation_pool(trade_day: date, engineer: ShortTermFeatureEngineer, rule_weight: float, logger=None) -> List[Dict[str, Any]]:
    states = engineer.store.load_recommendation_pool_state(trade_date=trade_day) or []
    if not states:
        if logger:
            logger.info("build_top20 recommendation_pool miss: trade_date=%s reason=no_state", trade_day.isoformat())
        return []
    stage2_items = []
    for item in states:
        if str(item.get("selection_stage") or "") not in {"stage2_top20_pre_moneyflow", "stage3_final_top3"}:
            continue
        components = dict(item.get("selection_reason_components") or {})
        model_score = num(item.get("rerank_model_score"))
        rule_score = num(item.get("rerank_rule_score"))
        blend_score = num(item.get("rerank_blend_score"))
        if model_score is None and rule_score is None and blend_score is None and not components:
            continue
        model_score_norm = first_num(item, components, ["model_score_norm"])
        rule_score_norm = first_num(item, components, ["rule_score_norm"])
        pool_blend_score = blend_score if blend_score is not None else first_num(item, components, ["blend_score"])
        if model_score_norm is None and rule_score_norm is None and pool_blend_score is None:
            continue
        bounded_rule_weight = max(0.0, min(1.0, float(rule_weight)))
        if model_score_norm is None:
            recomputed_blend = rule_score_norm
        elif rule_score_norm is None:
            recomputed_blend = model_score_norm
        else:
            recomputed_blend = (1.0 - bounded_rule_weight) * float(model_score_norm) + bounded_rule_weight * float(rule_score_norm)
        stage2_items.append({
            "ts_code": item.get("ts_code"),
            "name": item.get("name") or item.get("ts_code"),
            "base": num(item.get("recommendation_score")) or num(item.get("final_display_recommendation_score")) or num(item.get("overall_score")),
            "blend_score": round(float(recomputed_blend), 6) if recomputed_blend is not None else pool_blend_score,
            "model_score_norm": model_score_norm,
            "rule_score_norm": rule_score_norm,
            "pos": intval(item.get("structured_rank_position")) or intval(item.get("recommend_rank")),
            "risk": first_num(item, components, ["ranking_risk_penalty", "risk_penalty"]),
            "theme": first_num(item, components, ["theme_support_score"]),
            "cont": first_num(item, components, ["continuation_bias_score", "continuation_bias"]),
            "high": bool(item.get("unsupported_high_position_flag", False) or components.get("unsupported_high_position_flag", False)),
            "relay": bool(item.get("relay_candidate_veto", False) or components.get("relay_candidate_veto", False)),
        })
    if not stage2_items:
        if logger:
            logger.info("build_top20 recommendation_pool miss: trade_date=%s reason=no_stage2_items", trade_day.isoformat())
        return []
    stage2_items.sort(key=lambda x: (-float(x.get("blend_score") or -999.0), int(x.get("pos") or 9999), str(x.get("ts_code") or "")))
    return stage2_items[:20]


def summarize_rule_weight_run(days: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"evaluated_days": len(days), "horizons": {}}
    for h in H:
        returns = []
        positives = []
        for day in days:
            perf = ((day.get("performance") or {}).get("summary") or {}).get(str(h), {})
            if perf.get("avg_return") is not None:
                returns.append(float(perf.get("avg_return")))
            if perf.get("positive_rate") is not None:
                positives.append(float(perf.get("positive_rate")))
        out["horizons"][str(h)] = {
            "avg_return": round(mean(returns), 6) if returns else None,
            "avg_positive_rate": round(mean(positives), 4) if positives else None,
            "sample_days": len(returns),
        }
    return out


def compact_rule_days(days: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "trade_date": day.get("trade_date"),
        "top3": day.get("top3"),
        "performance_summary": (day.get("performance") or {}).get("summary"),
    } for day in days]


def pick_best_rule_weight(runs: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for h in H:
        best_weight = None
        best_value = None
        for run in runs:
            payload = (((run.get("summary") or {}).get("horizons") or {}).get(str(h)) or {})
            value = payload.get(metric)
            if value is None:
                continue
            if best_value is None or float(value) > float(best_value):
                best_value = float(value)
                best_weight = run.get("rule_weight")
        out[str(h)] = {"rule_weight": best_weight, "value": best_value}
    return out


def parse_rule_weights(raw: str, *, default: float) -> List[float]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        return [default]
    parsed = []
    for value in values:
        weight = float(value)
        parsed.append(max(0.0, min(1.0, weight)))
    return parsed


def first_num(payload: Dict[str, Any], comps: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if payload.get(k) is not None:
            return num(payload.get(k))
        if comps.get(k) is not None:
            return num(comps.get(k))
    return None


def num(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def intval(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()

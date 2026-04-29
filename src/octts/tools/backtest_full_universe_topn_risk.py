from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from octts.config import Settings
from octts.models.screening_models import DatabaseManager, MarketStockBasic
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.regression_rerank_service import RegressionRerankService

logger = logging.getLogger(__name__)


def dates(raw: str):
    return [datetime.strptime(x.strip(), "%Y-%m-%d").date() for x in raw.split(",") if x.strip()]


def ints(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def floats(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def strs(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def num(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def weighted_rank(score_maps: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    parts = [pd.Series(m, dtype=float).rank(method="average", pct=True) * weights[name] for name, m in score_maps.items()]
    combo = parts[0]
    for part in parts[1:]:
        combo = combo.add(part, fill_value=0.0)
    return {str(k): float(v) for k, v in combo.items()}


def pct_rank(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ranked = pd.Series(values, dtype=float).rank(method="average", pct=True)
    return {str(k): float(v) for k, v in ranked.items()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str):
        return [float(x[key]) for x in rows if x.get(key) is not None]
    r1, r3, r5 = vals("return_1d"), vals("return_3d"), vals("return_5d")
    a1 = [1.0 if float(x.get("return_1d") or 0) > 0 else 0.0 for x in rows if x.get("return_1d") is not None]
    a3 = [1.0 if float(x.get("return_3d") or 0) > 0 else 0.0 for x in rows if x.get("return_3d") is not None]
    return {
        "avg_return_1d": mean(r1) if r1 else None,
        "avg_return_3d": mean(r3) if r3 else None,
        "avg_return_5d": mean(r5) if r5 else None,
        "accuracy_1d": mean(a1) if a1 else None,
        "accuracy_3d": mean(a3) if a3 else None,
        "count": len(rows),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"days": len(items)}
    for key in ["avg_return_1d", "avg_return_3d", "avg_return_5d", "accuracy_1d", "accuracy_3d"]:
        values = [float(x[key]) for x in items if x.get(key) is not None]
        out[key] = mean(values) if values else None
    return out


def load_stock_name_map(database_url: str, ts_codes: list[str]) -> dict[str, str]:
    unique_codes = list(dict.fromkeys(str(code).strip().upper() for code in ts_codes if str(code).strip()))
    if not unique_codes:
        return {}
    db = DatabaseManager(database_url)
    session = db.get_session()
    try:
        rows = session.query(MarketStockBasic).filter(MarketStockBasic.ts_code.in_(unique_codes)).all()
        return {
            str(row.ts_code).strip().upper(): str(row.name or "").strip()
            for row in rows
            if row.ts_code
        }
    finally:
        session.close()


def is_st_stock_name(name: Any) -> bool:
    normalized = str(name or "").upper().replace(" ", "")
    return "ST" in normalized


def validate_exclude_st_modes(raw: str) -> list[str]:
    modes = strs(raw)
    allowed = {"none", "drop", "refill"}
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise ValueError(f"Unsupported exclude_st_mode value(s): {invalid}, allowed={sorted(allowed)}")
    if not modes:
        raise ValueError("exclude_st_mode cannot be empty")
    return modes


def select_topk_with_st_mode(
    adjusted: list[tuple[str, float, float]],
    *,
    top_k: int,
    stock_name_map: dict[str, str],
    exclude_st_mode: str,
) -> tuple[list[tuple[str, float, float]], int]:
    if exclude_st_mode == "none":
        return adjusted[:top_k], 0

    if exclude_st_mode == "drop":
        base_topk = adjusted[:top_k]
        filtered = [
            row for row in base_topk
            if not is_st_stock_name(stock_name_map.get(row[0], row[0]))
        ]
        return filtered, len(base_topk) - len(filtered)

    if exclude_st_mode == "refill":
        picked: list[tuple[str, float, float]] = []
        skipped_st = 0
        for row in adjusted:
            code = row[0]
            if is_st_stock_name(stock_name_map.get(code, code)):
                skipped_st += 1
                continue
            picked.append(row)
            if len(picked) >= top_k:
                break
        return picked, skipped_st

    raise ValueError(f"Unsupported exclude_st_mode: {exclude_st_mode}")


def score_samples(samples: list[Any], artifacts: list[dict[str, Any]], weights: dict[str, float]):
    sample_map = {s.ts_code.strip().upper(): s.model_dump(mode="python") for s in samples}
    score_maps: dict[str, dict[str, float]] = {}
    for spec in artifacts:
        name, artifact = str(spec["model_name"]), spec["artifact"]
        columns, model = list(artifact.get("feature_columns") or []), artifact.get("model")
        rows, codes = [], []
        for code, sample in sample_map.items():
            codes.append(code)
            rows.append({column: sample.get(column, 0.0) for column in columns})
        frame = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        preds = model.predict(frame)
        score_maps[name] = {code: float(pred) for code, pred in zip(codes, preds)}
    return weighted_rank(score_maps, weights), sample_map


def build_risk_map(repo: MarketRawDataRepository, codes: list[str], sample_map: dict[str, dict[str, Any]], trade_date: str):
    limit_map = repo.get_limit_list_by_trade_date(ts_codes=codes, trade_date=trade_date)
    top_map = repo.get_top_list_by_trade_date(ts_codes=codes, trade_date=trade_date)
    risks: dict[str, dict[str, Any]] = {}
    for code in codes:
        sample = sample_map.get(code, {})
        limit_row = limit_map.get(code) or {}
        top_rows = top_map.get(code) or []
        failure_score = 0.0
        flags: list[str] = []
        def add(score: float, flag: str):
            nonlocal failure_score
            failure_score += score
            flags.append(flag)
        price_position = num(sample.get("price_position_20d"))
        return_5d = num(sample.get("return_5d_past"))
        pct_change = num(sample.get("pct_change"))
        max_drawdown_10d = num(sample.get("max_drawdown_10d_past"))
        volume_ratio = num(sample.get("volume_ratio"))
        high_position = price_position >= 0.88
        recent_runup = return_5d >= 0.10
        if high_position and pct_change <= -1.0:
            add(1.0, "高位当日转弱")
        if recent_runup and max_drawdown_10d <= -0.08:
            add(0.8, "短期涨幅后回撤偏大")
        if high_position and recent_runup and volume_ratio >= 2.5 and pct_change <= 0:
            add(0.9, "高位放量但价格未延续")
        try:
            open_times = int(limit_row.get("open_times")) if limit_row.get("open_times") is not None else 0
        except (TypeError, ValueError):
            open_times = 0
        last_time = str(limit_row.get("last_time") or "").strip()
        top_net = sum(num(row.get("net_amount")) for row in top_rows)
        rates = [num(row.get("net_rate"), None) for row in top_rows]
        rates = [x for x in rates if x is not None]
        top_rate = mean(rates) if rates else None
        if open_times >= 3:
            add(1.4, "炸板次数过多")
        elif open_times >= 2 and last_time.isdigit() and int(last_time) >= 145000:
            add(1.2, "多次开板且尾盘封板")
        elif open_times >= 1 and pct_change <= 0:
            add(0.6, "开板分歧且价格未延续")
        if top_net < 0 and top_rate is not None and top_rate <= -3.0:
            add(1.2, "龙虎榜明显净卖出")
        elif top_net < 0 and high_position:
            add(0.7, "高位龙虎榜净卖出")
        relay_veto = open_times >= 3 or (open_times >= 2 and last_time.isdigit() and int(last_time) >= 145000) or (top_net < 0 and top_rate is not None and top_rate <= -3.0)
        failure_score = round(failure_score, 2)
        risks[code] = {
            "distribution_risk_score": failure_score,
            "risk_flags": flags,
            "candidate_risk_blocked": relay_veto,
            "relay_candidate_veto": relay_veto,
            "relay_top_net_amount": round(top_net, 2),
            "relay_open_times": open_times,
            "moneyflow_used": False,
        }
    return risks


def apply_risk(codes: list[str], scores: dict[str, float], risks: dict[str, dict[str, Any]], mode: str, weight: float, max_penalty: float):
    norms = pct_rank({code: num(risks.get(code, {}).get("distribution_risk_score")) for code in codes})
    rows = []
    for code in codes:
        risk = risks.get(code, {})
        extreme = bool(risk.get("candidate_risk_blocked")) or bool(risk.get("relay_candidate_veto"))
        if mode in {"extreme_veto", "failure_soft"} and extreme:
            continue
        penalty = min(max_penalty, weight * norms.get(code, 0.0)) if mode == "failure_soft" else 0.0
        rows.append((code, scores[code] - penalty, penalty))
    return sorted(rows, key=lambda x: (x[1], scores[x[0]]), reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Backtest full-universe model TopN with risk modes")
    parser.add_argument("--trade-dates", required=True)
    parser.add_argument("--model-top-n", default="100")
    parser.add_argument("--risk-mode", default="none,extreme_veto,failure_soft")
    parser.add_argument("--risk-weight", default="0.0001,0.0003,0.0005,0.001")
    parser.add_argument("--max-risk-penalty", type=float, default=0.001, help="Cap single-stock failure_soft penalty to avoid large reranking.")
    parser.add_argument("--top-k", default="3,5,10")
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument(
        "--exclude-st-mode",
        default="none",
        help="Comma-separated ST handling modes: none=keep, drop=remove without refill, refill=skip ST and backfill with next non-ST candidate.",
    )
    parser.add_argument("--output", default="tmp/full_universe_topn_risk_backtest.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    settings = Settings()
    repo = MarketRawDataRepository(settings.database_url)
    rerank = RegressionRerankService(settings)
    artifacts = rerank._load_default_artifacts(rerank._resolve_default_artifact_paths())
    weights = {str(spec["model_name"]): float(spec["weight"]) for spec in artifacts}
    trade_days, top_ns, modes = dates(args.trade_dates), ints(args.model_top_n), strs(args.risk_mode)
    risk_weights, top_ks = floats(args.risk_weight), ints(args.top_k)
    exclude_st_modes = validate_exclude_st_modes(args.exclude_st_mode)
    payload: dict[str, Any] = {
        "trade_dates": [str(d) for d in trade_days],
        "model_weights": weights,
        "max_risk_penalty": args.max_risk_penalty,
        "exclude_st_modes": exclude_st_modes,
        "daily_results": [],
    }
    for day in trade_days:
        logger.info("processing trade_date=%s", day)
        samples = rerank.dataset_builder.build_samples(start_date=day, end_date=day, exclude_bj=args.exclude_bj)
        scores, sample_map = score_samples(samples, artifacts, weights)
        ranked = [code for code, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        risk_codes = ranked[:max(top_ns)]
        risks = build_risk_map(repo, risk_codes, sample_map, day.strftime("%Y%m%d"))
        stock_name_map = load_stock_name_map(settings.database_url, risk_codes)
        configs = []
        for n in top_ns:
            base = ranked[:n]
            extreme_available = apply_risk(base, scores, risks, "extreme_veto", 0.0, args.max_risk_penalty)
            hard_filtered = len(base) - len(extreme_available)
            for mode in modes:
                active_weights = risk_weights if mode == "failure_soft" else [0.0]
                for rw in active_weights:
                    adjusted = apply_risk(base, scores, risks, mode, rw, args.max_risk_penalty)
                    for exclude_st_mode in exclude_st_modes:
                        for k in top_ks:
                            selected_rows, st_filtered_count = select_topk_with_st_mode(
                                adjusted,
                                top_k=k,
                                stock_name_map=stock_name_map,
                                exclude_st_mode=exclude_st_mode,
                            )
                            picked = []
                            for rank, (code, final_score, penalty) in enumerate(selected_rows, 1):
                                sample, risk = sample_map[code], risks.get(code, {})
                                stock_name = stock_name_map.get(code, code)
                                picked.append({
                                    "rank": rank,
                                    "ts_code": code,
                                    "name": stock_name,
                                    "is_st": is_st_stock_name(stock_name),
                                    "model_rank": ranked.index(code) + 1,
                                    "model_score": round(scores[code], 6),
                                    "final_score": round(final_score, 6),
                                    "risk_penalty": round(penalty, 6),
                                    "distribution_risk_score": risk.get("distribution_risk_score"),
                                    "candidate_risk_blocked": risk.get("candidate_risk_blocked"),
                                    "relay_candidate_veto": risk.get("relay_candidate_veto"),
                                    "risk_flags": risk.get("risk_flags", []),
                                    "return_1d": sample.get("return_1d"),
                                    "return_3d": sample.get("return_3d"),
                                    "return_5d": sample.get("return_5d"),
                                })
                            configs.append({
                                "model_top_n": n,
                                "risk_mode": mode,
                                "risk_weight": rw if mode == "failure_soft" else None,
                                "top_k": k,
                                "exclude_st_mode": exclude_st_mode,
                                "st_filtered_count": st_filtered_count,
                                "available_after_risk": len(adjusted),
                                "hard_filtered_count": hard_filtered if mode in {"extreme_veto", "failure_soft"} else 0,
                                "picked": picked,
                                "summary": summarize(picked),
                            })
        payload["daily_results"].append({"trade_date": str(day), "full_sample_size": len(sample_map), "configs": configs})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for day in payload["daily_results"]:
        for cfg in day["configs"]:
            key = json.dumps({
                "model_top_n": cfg["model_top_n"],
                "risk_mode": cfg["risk_mode"],
                "risk_weight": cfg["risk_weight"],
                "top_k": cfg["top_k"],
                "exclude_st_mode": cfg.get("exclude_st_mode"),
            }, sort_keys=True)
            grouped.setdefault(key, []).append(cfg["summary"])
    summary = [{**json.loads(key), **aggregate(items)} for key, items in grouped.items()]
    summary.sort(key=lambda x: x.get("avg_return_3d") if x.get("avg_return_3d") is not None else -999, reverse=True)
    payload["summary"] = summary
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 150)
    print("全市场模型 TopN + 风险模式回测汇总（按平均3日收益排序）")
    print("=" * 150)
    print(f"{'TopN':<8} {'RiskMode':<14} {'RiskW':<8} {'TopK':<6} {'STMode':<8} {'1日收益':<10} {'3日收益':<10} {'5日收益':<10} {'1日准确':<10} {'3日准确':<10}")
    print("-" * 150)
    for row in summary:
        fmt = lambda key: f"{row[key] * 100:.2f}%" if row.get(key) is not None else "N/A"
        rw = "-" if row.get("risk_weight") is None else f"{float(row['risk_weight']):.2f}"
        st_mode = str(row.get("exclude_st_mode") or "none")
        print(f"{row['model_top_n']:<8} {row['risk_mode']:<14} {rw:<8} {row['top_k']:<6} {st_mode:<8} {fmt('avg_return_1d'):<10} {fmt('avg_return_3d'):<10} {fmt('avg_return_5d'):<10} {fmt('accuracy_1d'):<10} {fmt('accuracy_3d'):<10}")


if __name__ == "__main__":
    main()

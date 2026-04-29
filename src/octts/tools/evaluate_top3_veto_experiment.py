from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import pandas as pd

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.evaluate_regression_ranking_variants import (
    _apply_blended_scores,
    _rebuild_single_trade_date_pool_with_scores,
)
from octts.tools.train_raw_market_model import _fit_model, resolve_feature_columns

DEFAULT_VETO_RULES = ["high_position", "high_turnover_spike", "high_volatility"]


def _f(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _dist(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(int(item[key]) for item in items).items())}


def _metrics(selected: pd.DataFrame, abs_col: str, rel_col: str) -> dict[str, Any]:
    return {
        "picked_codes": selected["ts_code"].astype(str).tolist(),
        "absolute_returns": [None if pd.isna(v) else float(v) for v in selected[abs_col].tolist()],
        "relative_returns": [None if pd.isna(v) else float(v) for v in selected[rel_col].tolist()],
        "avg_absolute_return": _mean(selected[abs_col]),
        "avg_relative_return": _mean(selected[rel_col]),
        "count_abs_ge_0": int((selected[abs_col] >= 0.0).sum()),
        "count_abs_gt_0": int((selected[abs_col] > 0.0).sum()),
        "count_rel_ge_0": int((selected[rel_col] >= 0.0).sum()),
        "count_rel_gt_0": int((selected[rel_col] > 0.0).sum()),
        "count_rel_ge_1pct": int((selected[rel_col] >= 0.01).sum()),
        "count_rel_ge_2pct": int((selected[rel_col] >= 0.02).sum()),
        "count_rel_ge_3pct": int((selected[rel_col] >= 0.03).sum()),
    }


def _summary(days: list[dict[str, Any]], final_pick: int, abs_col: str, rel_col: str) -> dict[str, Any]:
    total = sum(len(item["picked_codes"]) for item in days)
    return {
        "evaluated_days": int(len(days)),
        "final_pick": int(final_pick),
        "absolute_target": abs_col,
        "relative_target": rel_col,
        "avg_absolute_return": float(sum(item["avg_absolute_return"] for item in days) / len(days)) if days else 0.0,
        "avg_relative_return": float(sum(item["avg_relative_return"] for item in days) / len(days)) if days else 0.0,
        "per_stock_hit_rates": {
            "abs_ge_0": float(sum(item["count_abs_ge_0"] for item in days) / total) if total else 0.0,
            "abs_gt_0": float(sum(item["count_abs_gt_0"] for item in days) / total) if total else 0.0,
            "rel_ge_0": float(sum(item["count_rel_ge_0"] for item in days) / total) if total else 0.0,
            "rel_gt_0": float(sum(item["count_rel_gt_0"] for item in days) / total) if total else 0.0,
            "rel_ge_1pct": float(sum(item["count_rel_ge_1pct"] for item in days) / total) if total else 0.0,
            "rel_ge_2pct": float(sum(item["count_rel_ge_2pct"] for item in days) / total) if total else 0.0,
            "rel_ge_3pct": float(sum(item["count_rel_ge_3pct"] for item in days) / total) if total else 0.0,
        },
        "daily_hit_distribution": {
            "abs_ge_0": _dist(days, "count_abs_ge_0"),
            "abs_gt_0": _dist(days, "count_abs_gt_0"),
            "rel_ge_0": _dist(days, "count_rel_ge_0"),
            "rel_gt_0": _dist(days, "count_rel_gt_0"),
            "rel_ge_1pct": _dist(days, "count_rel_ge_1pct"),
            "rel_ge_2pct": _dist(days, "count_rel_ge_2pct"),
            "rel_ge_3pct": _dist(days, "count_rel_ge_3pct"),
        },
    }


def _reasons(row: pd.Series, rules: list[str]) -> list[str]:
    p20 = _f(row.get("price_position_20d"))
    vol_ratio = _f(row.get("volume_ratio"))
    turnover = _f(row.get("turnover_rate"))
    turnover_rank = _f(row.get("turnover_rate_rank_pct"))
    vol5 = _f(row.get("volatility_5d"))
    draw10 = _f(row.get("max_drawdown_10d_past"))
    amt_ratio = _f(row.get("amount_ratio_1d_5d"))
    ma5 = _f(row.get("close_to_ma5"))
    ma10 = _f(row.get("close_to_ma10"))
    pct = _f(row.get("pct_change"))
    up5 = _f(row.get("up_days_5d"))
    rel3 = _f(row.get("stock_vs_market_return_3d"))
    rel10 = _f(row.get("stock_vs_market_return_10d"))
    nh20 = _f(row.get("new_high_gap_20d"))
    out: list[str] = []
    if "high_position" in rules and p20 is not None and p20 >= 0.95:
        out.append("high_position")
    if "high_turnover_spike" in rules and ((turnover is not None and turnover >= 18 and vol_ratio is not None and vol_ratio >= 2.0) or (turnover_rank is not None and turnover_rank >= 0.98 and vol_ratio is not None and vol_ratio >= 1.8)):
        out.append("high_turnover_spike")
    if "high_volatility" in rules and vol5 is not None and vol5 >= 0.09:
        out.append("high_volatility")
    if "deep_drawdown" in rules and draw10 is not None and draw10 <= -0.14:
        out.append("deep_drawdown")
    if "chase_gap" in rules and amt_ratio is not None and amt_ratio >= 2.2 and pct is not None and pct >= 6.0:
        out.append("chase_gap")
    if "too_far_above_ma" in rules and ((ma5 is not None and ma5 >= 1.08) or (ma10 is not None and ma10 >= 1.12)):
        out.append("too_far_above_ma")
    if "overextended_runup" in rules and up5 is not None and up5 >= 4 and rel3 is not None and rel3 >= 0.10:
        out.append("overextended_runup")
    if "near_20d_new_high" in rules and nh20 is not None and nh20 <= 0.01 and p20 is not None and p20 >= 0.92:
        out.append("near_20d_new_high")
    if "too_hot_10d" in rules and rel10 is not None and rel10 >= 0.18:
        out.append("too_hot_10d")
    return out


def _select(day: pd.DataFrame, final_pick: int, rules: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ranked = day.sort_values(["model_score", "ts_code"], ascending=[False, True]).reset_index(drop=True)
    kept, skipped = [], []
    for _, row in ranked.iterrows():
        reasons = _reasons(row, rules)
        if reasons and len(ranked) - len(skipped) > final_pick:
            skipped.append({"ts_code": str(row.get("ts_code")), "model_score": float(row.get("model_score") or 0.0), "reasons": reasons})
            continue
        kept.append(row)
        if len(kept) >= final_pick:
            break
    if not kept:
        kept = [row for _, row in ranked.head(final_pick).iterrows()]
    return pd.DataFrame(kept), skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Compare baseline Top3 and risk-veto Top3.")
    p.add_argument("--input", required=True)
    p.add_argument("--target", default="vs_market_1d", choices=["vs_market_1d", "vs_market_3d", "vs_market_5d"])
    p.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    p.add_argument("--pool-limit", type=int, default=200)
    p.add_argument("--final-pick", type=int, default=3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-trade-days", type=int, default=20)
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--exclude-bj", action="store_true")
    p.add_argument("--score-mode", default="model_rule_blend", choices=["model_only", "model_rule_blend"])
    p.add_argument("--rule-weight", type=float, default=0.3)
    p.add_argument("--feature-columns", default="")
    p.add_argument("--feature-file", default="")
    p.add_argument("--veto-rules", default=",".join(DEFAULT_VETO_RULES))
    p.add_argument("--output-daily-limit", type=int, default=20)
    args = p.parse_args()

    frame = pd.read_csv(args.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"})
        return
    abs_col = {"vs_market_1d": "return_1d", "vs_market_3d": "return_3d", "vs_market_5d": "return_5d"}[args.target]
    need = ["trade_date", "ts_code", args.target, abs_col]
    miss = [c for c in need if c not in frame.columns]
    if miss:
        print_json({"evaluated": False, "reason": "missing_columns", "columns": miss})
        return

    labeled = frame[frame[args.target].notna()].copy()
    labeled["trade_date"] = pd.to_datetime(labeled["trade_date"])
    labeled = labeled.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    labeled["ts_code"] = labeled["ts_code"].astype(str).str.strip()
    feat_cols = resolve_feature_columns(labeled, feature_columns_arg=args.feature_columns, feature_file_arg=args.feature_file)
    features = labeled[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target = pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0)

    split = int(len(labeled) * (1 - args.test_size))
    split = max(1, min(split, len(labeled) - 1))
    x_train, y_train = features.iloc[:split], target.iloc[:split]
    x_test = features.iloc[split:].reset_index(drop=True)
    extra = [c for c in feat_cols if c not in {abs_col, args.target}]
    meta = labeled.iloc[split:][["trade_date", "ts_code", abs_col, args.target] + extra].reset_index(drop=True)

    model = _fit_model(args.model_type, x_train, y_train, is_regression=True)
    scored = pd.concat([meta, pd.Series(model.predict(x_test), name="model_score")], axis=1)
    scored["model_score"] = pd.to_numeric(scored["model_score"], errors="coerce").fillna(0.0)
    scored[abs_col] = pd.to_numeric(scored[abs_col], errors="coerce")
    scored[args.target] = pd.to_numeric(scored[args.target], errors="coerce")
    if args.start_date:
        scored = scored[scored["trade_date"] >= pd.Timestamp(args.start_date)]
    if args.end_date:
        scored = scored[scored["trade_date"] <= pd.Timestamp(args.end_date)]
    if scored.empty:
        print_json({"evaluated": False, "reason": "empty_scored_range"})
        return

    trade_dates = sorted({v.date() for v in scored["trade_date"]})
    if args.max_trade_days > 0:
        trade_dates = trade_dates[-args.max_trade_days:]
        scored = scored[scored["trade_date"].dt.date.isin(trade_dates)].reset_index(drop=True)

    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    rebuilt = {d: _rebuild_single_trade_date_pool_with_scores(repo, d, exclude_bj=args.exclude_bj) for d in trade_dates}
    pools = {d: [item["ts_code"] for item in items] for d, items in rebuilt.items()}
    if args.score_mode == "model_rule_blend":
        scored = _apply_blended_scores(scored, rebuilt, rule_weight=args.rule_weight)

    rules = [r.strip() for r in args.veto_rules.split(",") if r.strip()]
    base_days, veto_days = [], []
    reason_counts: Counter[str] = Counter()
    for trade_day, day in scored.groupby("trade_date", sort=True):
        d = trade_day.date()
        codes = set(pools.get(d, [])[: args.pool_limit])
        cand = day[day["ts_code"].astype(str).isin(codes)].copy()
        if cand.empty:
            continue
        baseline = cand.nlargest(min(args.final_pick, len(cand)), columns="model_score").copy()
        veto, skipped = _select(cand, args.final_pick, rules)
        for item in skipped:
            for r in item["reasons"]:
                reason_counts[r] += 1
        b = _metrics(baseline, abs_col, args.target)
        b["trade_date"] = d.isoformat()
        v = _metrics(veto, abs_col, args.target)
        v["trade_date"] = d.isoformat()
        v["skipped_candidates"] = skipped[:5]
        base_days.append(b)
        veto_days.append(v)

    if not base_days or not veto_days:
        print_json({"evaluated": False, "reason": "no_daily_results"})
        return

    print_json({
        "evaluated": True,
        "input": args.input,
        "model_type": args.model_type,
        "feature_count": int(len(feat_cols)),
        "enabled_veto_rules": rules,
        "date_range": {"start": min(trade_dates).isoformat(), "end": max(trade_dates).isoformat(), "trade_days": int(len(trade_dates))},
        "baseline_summary": _summary(base_days, args.final_pick, abs_col, args.target),
        "veto_summary": _summary(veto_days, args.final_pick, abs_col, args.target),
        "veto_reason_counts": dict(reason_counts),
        "daily_comparison": [{"trade_date": base_days[i]["trade_date"], "baseline": base_days[i], "veto": veto_days[i]} for i in range(min(len(base_days), len(veto_days), args.output_daily_limit))],
        "daily_result_count": int(min(len(base_days), len(veto_days))),
    })


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.evaluate_regression_ranking_variants import (
    _apply_blended_scores,
    _build_month_stats_for_target,
    _evaluate_candidate_pool_sizes_for_target,
    _rebuild_single_trade_date_pool_with_scores,
)
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS, _fit_model, resolve_feature_columns
from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository

SELECTED34_FILE = "data/raw_market_selected_features_v1.txt"
FULL42_NAME = "full42"
SELECTED34_NAME = "selected34"
CURRENT_DEFAULT_NAME = "current_default_artifact"
# Historical 37-feature baseline that matched the prior default artifact before switching to selected34.
CURRENT_DEFAULT_COLUMNS = [
    "close",
    "pct_change",
    "turnover_rate",
    "volume_ratio",
    "market_cap",
    "pe_ttm",
    "pb",
    "amount",
    "vol",
    "volatility_5d",
    "volatility_10d",
    "max_drawdown_10d_past",
    "close_to_ma5",
    "close_to_ma10",
    "close_to_ma20",
    "price_position_20d",
    "avg_turnover_rate_5d",
    "avg_volume_ratio_5d",
    "market_return_1d",
    "market_return_3d",
    "market_return_5d",
    "market_up_ratio_1d",
    "market_up_ratio_3d_avg",
    "market_up_days_5d",
    "stock_vs_market_return_1d",
    "stock_vs_market_return_3d",
    "stock_vs_market_return_5d",
    "stock_vs_market_return_10d",
    "pct_change_rank_pct",
    "turnover_rate_rank_pct",
    "volume_ratio_rank_pct",
    "up_days_3d",
    "up_days_5d",
    "new_high_gap_20d",
    "new_low_gap_20d",
    "amount_ratio_3d_10d",
    "turnover_rate_change_5d",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Compare raw-market feature sets: current_default_artifact vs selected34 vs full42.")
    p.add_argument("--input", required=True)
    p.add_argument("--target", default="vs_market_1d", choices=["vs_market_1d", "vs_market_3d", "vs_market_5d"])
    p.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    p.add_argument("--pool-limit", type=int, default=200)
    p.add_argument("--final-pick", type=int, default=3)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--start-date", default="")
    p.add_argument("--end-date", default="")
    p.add_argument("--max-trade-days", type=int, default=10)
    p.add_argument("--exclude-bj", action="store_true")
    p.add_argument("--score-mode", default="model_rule_blend", choices=["model_only", "model_rule_blend"])
    p.add_argument("--rule-weight", type=float, default=0.3)
    p.add_argument("--output-file", default="")
    p.add_argument("--compact", action="store_true")
    a = p.parse_args()

    frame = pd.read_csv(a.input, low_memory=False)
    if frame.empty:
        print_json({"evaluated": False, "reason": "empty_dataset"}, output_file=a.output_file or None)
        return

    labeled = frame[frame[a.target].notna()].copy()
    if labeled.empty:
        print_json({"evaluated": False, "reason": "no_labeled_rows", "target": a.target}, output_file=a.output_file or None)
        return

    labeled["trade_date"] = pd.to_datetime(labeled["trade_date"])
    labeled = labeled.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    labeled["ts_code"] = labeled["ts_code"].astype(str).str.strip()

    split_index = int(len(labeled) * (1 - a.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    meta_test = labeled.iloc[split_index:][["trade_date", "ts_code"]].reset_index(drop=True)
    y_test = pd.to_numeric(labeled[a.target], errors="coerce").fillna(0.0).iloc[split_index:].reset_index(drop=True)

    trade_dates = sorted({value.date() for value in meta_test["trade_date"]})
    if a.start_date:
        trade_dates = [d for d in trade_dates if d >= pd.Timestamp(a.start_date).date()]
    if a.end_date:
        trade_dates = [d for d in trade_dates if d <= pd.Timestamp(a.end_date).date()]
    if a.max_trade_days > 0:
        trade_dates = trade_dates[-a.max_trade_days:]

    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    rebuilt_pool_info = {
        trade_date: _rebuild_single_trade_date_pool_with_scores(repo, trade_date, exclude_bj=a.exclude_bj)
        for trade_date in trade_dates
    }
    candidate_pools = {trade_date: [item["ts_code"] for item in items] for trade_date, items in rebuilt_pool_info.items()}

    selected34_path = Path(SELECTED34_FILE)
    selected34_columns = [
        line.strip() for line in selected34_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ] if selected34_path.exists() else []

    feature_sets = [
        {"name": CURRENT_DEFAULT_NAME, "columns": [c for c in CURRENT_DEFAULT_COLUMNS if c in labeled.columns]},
        {"name": SELECTED34_NAME, "columns": [c for c in selected34_columns if c in labeled.columns]},
        {"name": FULL42_NAME, "columns": [c for c in RAW_MARKET_FEATURE_COLUMNS if c in labeled.columns]},
    ]

    results = []
    for feature_set in feature_sets:
        results.append(
            evaluate_feature_set(
                labeled=labeled,
                feature_name=feature_set["name"],
                feature_columns=feature_set["columns"],
                target_column=a.target,
                model_type=a.model_type,
                split_index=split_index,
                meta_test=meta_test,
                y_test=y_test,
                trade_dates=trade_dates,
                rebuilt_pool_info=rebuilt_pool_info,
                candidate_pools=candidate_pools,
                start_date=a.start_date,
                end_date=a.end_date,
                pool_limit=a.pool_limit,
                final_pick=a.final_pick,
                score_mode=a.score_mode,
                rule_weight=a.rule_weight,
                compact=a.compact,
            )
        )

    payload = {
        "evaluated": True,
        "input": a.input,
        "target": a.target,
        "model_type": a.model_type,
        "pool_limit": int(a.pool_limit),
        "final_pick": int(a.final_pick),
        "exclude_bj": bool(a.exclude_bj),
        "score_mode": a.score_mode,
        "rule_weight": float(a.rule_weight),
        "date_range": {
            "start": min(trade_dates).isoformat() if trade_dates else None,
            "end": max(trade_dates).isoformat() if trade_dates else None,
            "trade_days": int(len(trade_dates)),
        },
        "feature_sets": [{"name": item["name"], "feature_count": item["feature_count"], "feature_columns": item["feature_columns"] if not a.compact else []} for item in results],
        "summary": build_summary(results),
        "results": results,
    }
    print_json(payload, output_file=a.output_file or None)



def evaluate_feature_set(
    *,
    labeled: pd.DataFrame,
    feature_name: str,
    feature_columns: List[str],
    target_column: str,
    model_type: str,
    split_index: int,
    meta_test: pd.DataFrame,
    y_test: pd.Series,
    trade_dates: List[Any],
    rebuilt_pool_info: Dict[Any, List[Dict[str, Any]]],
    candidate_pools: Dict[Any, List[str]],
    start_date: str,
    end_date: str,
    pool_limit: int,
    final_pick: int,
    score_mode: str,
    rule_weight: float,
    compact: bool,
) -> Dict[str, Any]:
    features = labeled[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_train = features.iloc[:split_index]
    y_train = pd.to_numeric(labeled[target_column], errors="coerce").fillna(0.0).iloc[:split_index]
    x_test = features.iloc[split_index:].reset_index(drop=True)
    model = _fit_model(model_type, x_train, y_train, is_regression=True)
    predictions = pd.Series(model.predict(x_test), name="model_score")
    scored = pd.concat([meta_test, predictions, y_test.rename(target_column)], axis=1)
    scored["model_score"] = pd.to_numeric(scored["model_score"], errors="coerce").fillna(0.0)
    if start_date:
        scored = scored[scored["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        scored = scored[scored["trade_date"] <= pd.Timestamp(end_date)]
    scored = scored[scored["trade_date"].dt.date.isin(trade_dates)].reset_index(drop=True)

    if score_mode == "model_rule_blend":
        scored = _apply_blended_scores(scored, rebuilt_pool_info, rule_weight=rule_weight)

    evaluation = _evaluate_candidate_pool_sizes_for_target(scored, candidate_pools, [pool_limit], final_pick, target_column)
    result = evaluation["results"][0]
    month_stats = _build_month_stats_for_target(scored, candidate_pools, pool_limit, final_pick, target_column)
    return {
        "name": feature_name,
        "evaluated": True,
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "summary": {
            "evaluated_days": result["evaluated_days"],
            "avg_target_return": result["avg_target_return"],
            "hit_rates": result["hit_rates"],
            "pool_baseline": result["pool_baseline"],
        },
        "month_stats": month_stats,
        "sample_daily_results": [] if compact else result["daily_results"][:10],
    }



def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_avg = max(results, key=lambda item: float((item.get("summary") or {}).get("avg_target_return") or float("-inf"))) if results else None
    hit_keys = [">=1%", ">=2%", ">=3%"]
    best_hits = {}
    for key in hit_keys:
        ranked = [item for item in results if ((item.get("summary") or {}).get("hit_rates") or {}).get(key) is not None]
        if ranked:
            winner = max(ranked, key=lambda item: float(((item.get("summary") or {}).get("hit_rates") or {}).get(key) or 0.0))
            best_hits[key] = {
                "name": winner.get("name"),
                "value": ((winner.get("summary") or {}).get("hit_rates") or {}).get(key),
            }
    return {
        "best_avg_target_return": None if best_avg is None else {
            "name": best_avg.get("name"),
            "value": (best_avg.get("summary") or {}).get("avg_target_return"),
        },
        "best_hit_rates": best_hits,
    }


if __name__ == "__main__":
    main()

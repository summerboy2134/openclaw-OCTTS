from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.evaluate_regression_ranking_variants import (
    _apply_blended_scores,
    _build_month_stats_for_target,
    _evaluate_candidate_pool_sizes_for_target,
    _rebuild_single_trade_date_pool_with_scores,
)
from octts.tools.train_raw_market_model import _fit_model


START_FEATURE_COLUMNS = [
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
    "price_position_10d",
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


def _evaluate_feature_set(
    labeled: pd.DataFrame,
    *,
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
    if scored.empty:
        return {
            "evaluated": False,
            "reason": "empty_scored_range",
            "feature_count": int(len(feature_columns)),
            "feature_columns": feature_columns,
        }

    if score_mode == "model_rule_blend":
        scored = _apply_blended_scores(scored, rebuilt_pool_info, rule_weight=rule_weight)

    evaluation = _evaluate_candidate_pool_sizes_for_target(
        scored,
        candidate_pools,
        [pool_limit],
        final_pick,
        target_column,
    )
    result = evaluation["results"][0]
    month_stats = _build_month_stats_for_target(
        scored,
        candidate_pools,
        pool_limit,
        final_pick,
        target_column,
    )
    return {
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run backward elimination for raw-market rerank features using final Top3 evaluation metrics."
    )
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_1d", choices=["vs_market_1d", "vs_market_3d", "vs_market_5d"])
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "lightgbm", "xgboost"])
    parser.add_argument("--pool-limit", type=int, default=200)
    parser.add_argument("--final-pick", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-trade-days", type=int, default=20)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--score-mode", default="model_rule_blend", choices=["model_only", "model_rule_blend"])
    parser.add_argument("--rule-weight", type=float, default=0.3)
    parser.add_argument("--min-improvement", type=float, default=0.0005)
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
    labeled["ts_code"] = labeled["ts_code"].astype(str).str.strip()

    feature_columns = [column for column in START_FEATURE_COLUMNS if column in labeled.columns]
    missing_start_columns = [column for column in START_FEATURE_COLUMNS if column not in labeled.columns]

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    meta_test = labeled.iloc[split_index:][["trade_date", "ts_code"]].reset_index(drop=True)
    y_test = pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0).iloc[split_index:].reset_index(drop=True)

    trade_dates = sorted({value.date() for value in meta_test["trade_date"]})
    if args.max_trade_days > 0:
        trade_dates = trade_dates[-args.max_trade_days:]

    settings = get_settings()
    repo = MarketRawDataRepository(settings.database_url)
    rebuilt_pool_info = {
        trade_date: _rebuild_single_trade_date_pool_with_scores(repo, trade_date, exclude_bj=args.exclude_bj)
        for trade_date in trade_dates
    }
    candidate_pools = {trade_date: [item["ts_code"] for item in items] for trade_date, items in rebuilt_pool_info.items()}

    history: List[Dict[str, Any]] = []
    current_columns = list(feature_columns)
    current_result = _evaluate_feature_set(
        labeled,
        feature_columns=current_columns,
        target_column=args.target,
        model_type=args.model_type,
        split_index=split_index,
        meta_test=meta_test,
        y_test=y_test,
        trade_dates=trade_dates,
        rebuilt_pool_info=rebuilt_pool_info,
        candidate_pools=candidate_pools,
        start_date=args.start_date,
        end_date=args.end_date,
        pool_limit=args.pool_limit,
        final_pick=args.final_pick,
        score_mode=args.score_mode,
        rule_weight=args.rule_weight,
    )
    history.append({
        "step": 0,
        "action": "baseline_start",
        "removed_feature": None,
        "result": current_result,
    })

    step = 0
    while len(current_columns) > 1:
        candidate_trials: List[Dict[str, Any]] = []
        for feature in current_columns:
            trial_columns = [column for column in current_columns if column != feature]
            trial_result = _evaluate_feature_set(
                labeled,
                feature_columns=trial_columns,
                target_column=args.target,
                model_type=args.model_type,
                split_index=split_index,
                meta_test=meta_test,
                y_test=y_test,
                trade_dates=trade_dates,
                rebuilt_pool_info=rebuilt_pool_info,
                candidate_pools=candidate_pools,
                start_date=args.start_date,
                end_date=args.end_date,
                pool_limit=args.pool_limit,
                final_pick=args.final_pick,
                score_mode=args.score_mode,
                rule_weight=args.rule_weight,
            )
            candidate_trials.append({
                "removed_feature": feature,
                "result": trial_result,
            })

        candidate_trials = sorted(
            candidate_trials,
            key=lambda item: (
                not item["result"].get("evaluated", False),
                -float(item["result"].get("summary", {}).get("avg_target_return", float("-inf"))),
                -float(item["result"].get("summary", {}).get("hit_rates", {}).get(">=2%", float("-inf"))),
                item["removed_feature"],
            ),
        )
        best_trial = candidate_trials[0]
        current_score = float(current_result.get("summary", {}).get("avg_target_return", float("-inf")))
        best_score = float(best_trial["result"].get("summary", {}).get("avg_target_return", float("-inf")))
        improvement = best_score - current_score
        step += 1

        history.append({
            "step": step,
            "action": "best_removal_trial",
            "removed_feature": best_trial["removed_feature"],
            "improvement": improvement,
            "result": best_trial["result"],
            "top_trials": candidate_trials[:5],
        })

        if improvement < float(args.min_improvement):
            break

        current_columns = best_trial["result"]["feature_columns"]
        current_result = best_trial["result"]

    print_json(
        {
            "evaluated": True,
            "input": args.input,
            "target": args.target,
            "model_type": args.model_type,
            "pool_limit": int(args.pool_limit),
            "final_pick": int(args.final_pick),
            "exclude_bj": bool(args.exclude_bj),
            "score_mode": args.score_mode,
            "rule_weight": float(args.rule_weight),
            "min_improvement": float(args.min_improvement),
            "date_range": {
                "start": min(trade_dates).isoformat() if trade_dates else None,
                "end": max(trade_dates).isoformat() if trade_dates else None,
                "trade_days": int(len(trade_dates)),
            },
            "missing_start_columns": missing_start_columns,
            "start_feature_count": int(len(feature_columns)),
            "final_feature_count": int(len(current_columns)),
            "final_feature_columns": current_columns,
            "final_result": current_result,
            "history": history,
        }
    )


if __name__ == "__main__":
    main()

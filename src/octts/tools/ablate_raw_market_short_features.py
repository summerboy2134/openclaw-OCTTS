from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import print_json
from octts.tools.evaluate_light_rule_pool_long_range import _build_month_stats
from octts.tools.evaluate_regression_ranking_variants import (
    _apply_blended_scores,
    _build_month_stats_for_target,
    _evaluate_candidate_pool_sizes_for_target,
    _rebuild_single_trade_date_pool_with_scores,
)
from octts.tools.train_raw_market_model import _fit_model


BASE_FEATURE_COLUMNS = [
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

INCREMENTAL_FEATURES = [
    "price_position_10d",
    "stock_vs_market_return_2d",
    "new_high_gap_10d",
    "amount_ratio_1d_5d",
    "turnover_rate_change_1d",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-click incremental ablation for new raw-market features using final rerank evaluation."
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
    target = pd.to_numeric(labeled[args.target], errors="coerce").fillna(0.0)

    split_index = int(len(labeled) * (1 - args.test_size))
    split_index = max(1, min(split_index, len(labeled) - 1))
    meta_test = labeled.iloc[split_index:][["trade_date", "ts_code"]].reset_index(drop=True)
    y_test = target.iloc[split_index:].reset_index(drop=True)

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

    experiment_definitions = [
        {"name": "baseline_old37", "extra_features": []},
        *[
            {"name": f"plus_{feature}", "extra_features": [feature]}
            for feature in INCREMENTAL_FEATURES
        ],
        {"name": "all_new_5", "extra_features": INCREMENTAL_FEATURES},
    ]

    experiments: List[Dict[str, Any]] = []
    for definition in experiment_definitions:
        feature_columns = [
            column
            for column in BASE_FEATURE_COLUMNS + definition["extra_features"]
            if column in labeled.columns
        ]
        features = labeled[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x_train = features.iloc[:split_index]
        y_train = target.iloc[:split_index]
        x_test = features.iloc[split_index:].reset_index(drop=True)

        model = _fit_model(args.model_type, x_train, y_train, is_regression=True)
        predictions = pd.Series(model.predict(x_test), name="model_score")
        scored = pd.concat([meta_test, predictions, y_test.rename(args.target)], axis=1)
        scored["model_score"] = pd.to_numeric(scored["model_score"], errors="coerce").fillna(0.0)

        if args.start_date:
            scored = scored[scored["trade_date"] >= pd.Timestamp(args.start_date)]
        if args.end_date:
            scored = scored[scored["trade_date"] <= pd.Timestamp(args.end_date)]
        scored = scored[scored["trade_date"].dt.date.isin(trade_dates)].reset_index(drop=True)
        if scored.empty:
            experiments.append(
                {
                    "name": definition["name"],
                    "feature_count": int(len(feature_columns)),
                    "extra_features": definition["extra_features"],
                    "evaluated": False,
                    "reason": "empty_scored_range",
                }
            )
            continue

        if args.score_mode == "model_rule_blend":
            scored = _apply_blended_scores(scored, rebuilt_pool_info, rule_weight=args.rule_weight)

        evaluation = _evaluate_candidate_pool_sizes_for_target(
            scored,
            candidate_pools,
            [args.pool_limit],
            args.final_pick,
            args.target,
        )
        result = evaluation["results"][0]
        month_stats = _build_month_stats_for_target(
            scored,
            candidate_pools,
            args.pool_limit,
            args.final_pick,
            args.target,
        )
        experiments.append(
            {
                "name": definition["name"],
                "feature_count": int(len(feature_columns)),
                "extra_features": definition["extra_features"],
                "evaluated": True,
                "summary": {
                    "evaluated_days": result["evaluated_days"],
                    "avg_target_return": result["avg_target_return"],
                    "hit_rates": result["hit_rates"],
                    "pool_baseline": result["pool_baseline"],
                },
                "month_stats": month_stats,
            }
        )

    ranked_experiments = sorted(
        experiments,
        key=lambda item: (
            not item.get("evaluated", False),
            -float(item.get("summary", {}).get("avg_target_return", float("-inf"))),
            -float(item.get("summary", {}).get("hit_rates", {}).get(">=2%", float("-inf"))),
            item["name"],
        ),
    )

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
            "date_range": {
                "start": min(trade_dates).isoformat() if trade_dates else None,
                "end": max(trade_dates).isoformat() if trade_dates else None,
                "trade_days": int(len(trade_dates)),
            },
            "experiments": experiments,
            "ranked_experiments": ranked_experiments,
        }
    )


if __name__ == "__main__":
    main()

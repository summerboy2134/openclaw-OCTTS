from __future__ import annotations

import argparse
import time
from datetime import datetime

from octts.config import get_settings
from octts.services.recommendation_tracker import RecommendationTracker
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.services.short_term_training_data import ShortTermTrainingDataBuilder
from octts.tools.common import configure_tool_logging, print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill short-term training samples.")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--months-back", type=int, default=6)
    parser.add_argument("--skip-performance-update", action="store_true")
    parser.add_argument("--rebuild-pool-states", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-every", type=int, default=5)
    parser.add_argument("--batch-sleep-seconds", type=float, default=8.0)
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "backfill_training_data")
    logger.info("Backfill task started: start_date=%s, end_date=%s", args.start_date, args.end_date)

    if not args.skip_performance_update:
        tracker = RecommendationTracker(settings)
        performance_summary = tracker.update_recommendation_performance(lookback_days=180)
        logger.info("Performance update summary: %s", performance_summary)

    builder = ShortTermTrainingDataBuilder(settings)
    engineer = ShortTermFeatureEngineer(settings, store=builder.store)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None
    rebuilt_trade_dates = None
    if args.rebuild_pool_states:
        trade_dates = engineer.list_trade_dates(months_back=args.months_back, end_date=end_date)
        if start_date:
            trade_dates = [trade_day for trade_day in trade_dates if trade_day >= start_date]
        if end_date:
            trade_dates = [trade_day for trade_day in trade_dates if trade_day <= end_date]
        rebuilt_trade_dates = trade_dates
        processed_trade_days = 0
        generated_pool_rows = 0
        sleep_every = max(int(args.sleep_every), 0)
        sleep_seconds = max(float(args.sleep_seconds), 0.0)
        batch_sleep_seconds = max(float(args.batch_sleep_seconds), 0.0)
        total_trade_days = len(trade_dates)
        for index, trade_day in enumerate(trade_dates, start=1):
            logger.info(
                "Backfill feature engineering start: trade_date=%s (%s/%s)",
                trade_day.isoformat(),
                index,
                total_trade_days,
            )
            payload = engineer.build_trade_date_pool_states(trade_day)
            pool_states = payload.get("pool_states") or []
            if pool_states:
                builder.store.upsert_recommendation_pool_states(pool_states)
                generated_pool_rows += len(pool_states)
            processed_trade_days += 1
            logger.info(
                "Backfill feature engineering complete: trade_date=%s, pool_states=%s, processed_trade_days=%s/%s",
                trade_day.isoformat(),
                len(pool_states),
                processed_trade_days,
                total_trade_days,
            )
            if sleep_seconds > 0:
                logger.info(
                    "Backfill per-trade-date throttle: processed_trade_days=%s/%s, sleep_seconds=%.2f",
                    processed_trade_days,
                    total_trade_days,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
            if sleep_every > 0 and processed_trade_days % sleep_every == 0 and batch_sleep_seconds > 0:
                logger.info(
                    "Backfill batch throttle: processed_trade_days=%s/%s, sleep_every=%s, batch_sleep_seconds=%.2f",
                    processed_trade_days,
                    total_trade_days,
                    sleep_every,
                    batch_sleep_seconds,
                )
                time.sleep(batch_sleep_seconds)
        logger.info(
            "Backfill pool-state rebuild summary: trade_days=%s, pool_rows=%s",
            processed_trade_days,
            generated_pool_rows,
        )
    result = builder.backfill_samples(
        start_date=start_date,
        end_date=end_date,
        trade_dates=rebuilt_trade_dates,
    )
    logger.info("Backfill task complete: %s", result)
    print_json(result)


if __name__ == "__main__":
    main()

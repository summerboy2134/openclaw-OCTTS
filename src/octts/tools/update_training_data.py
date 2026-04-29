from __future__ import annotations

import argparse
from datetime import datetime

from octts.config import get_settings
from octts.services.recommendation_tracker import RecommendationTracker
from octts.services.screening_store import ScreeningStore
from octts.services.short_term_training_data import ShortTermTrainingDataBuilder
from octts.tools.common import configure_tool_logging, print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Update short-term training samples for a single trade date.")
    parser.add_argument("--trade-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--skip-performance-update", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "update_training_data")
    logger.info("Update task started: trade_date=%s", args.trade_date)

    if not args.skip_performance_update:
        tracker = RecommendationTracker(settings)
        performance_summary = tracker.update_recommendation_performance(lookback_days=30)
        logger.info("Performance update summary: %s", performance_summary)

    store = ScreeningStore(settings)
    trade_date = None
    if args.trade_date:
        trade_date = datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    else:
        history = store.list_recommendation_history(limit=1)
        if history:
            trade_date = datetime.strptime(history[0]["trade_date"], "%Y-%m-%d").date()
    if trade_date is None:
        result = {"updated": False, "reason": "no_trade_date_available"}
        logger.warning("Update task skipped: %s", result)
        print_json(result)
        return

    builder = ShortTermTrainingDataBuilder(settings, store=store)
    persisted = builder.persist_samples_for_trade_date(trade_date)
    result = {
        "updated": True,
        "trade_date": trade_date.isoformat(),
        "sample_count": len(persisted),
    }
    logger.info("Update task complete: %s", result)
    print_json(result)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from octts.config import get_settings
from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder
from octts.tools.common import configure_tool_logging, print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw-market training dataset from local SQLite.")
    parser.add_argument("--start-date", required=True, help="Start trade date, e.g. 2025-10-01")
    parser.add_argument("--end-date", required=True, help="End trade date, e.g. 2026-03-31")
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument("--exclude-bj", action="store_true", help="Exclude Beijing Stock Exchange symbols (.BJ)")
    parser.add_argument("--output", default="data/raw_market_training_dataset.csv")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "build_raw_market_training_dataset")
    builder = RawMarketTrainingDatasetBuilder(settings)
    summary, samples = builder.build_dataset_summary(
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d").date(),
        min_history_days=args.min_history_days,
        exclude_bj=args.exclude_bj,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(samples[0].model_dump(mode="json").keys()) if samples else []
    with output_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample.model_dump(mode="json"))

    result = {**summary, "output_path": str(output_path)}
    logger.info("Raw-market training dataset build complete: %s", result)
    print_json(result)


if __name__ == "__main__":
    main()

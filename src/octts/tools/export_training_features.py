from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sqlalchemy import text

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import configure_tool_logging, print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export precomputed training_features table to CSV.")
    parser.add_argument("--start-date", required=True, help="Start trade date, e.g. 2025-08-01")
    parser.add_argument("--end-date", required=True, help="End trade date, e.g. 2026-05-15")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--drop-unlabeled-target", default="", help="Optional target column; rows with NULL target are excluded")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "export_training_features")
    repo = MarketRawDataRepository(settings.database_url)

    where_clauses = ["trade_date >= :start_date", "trade_date <= :end_date"]
    params: Dict[str, Any] = {
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    if args.drop_unlabeled_target:
        where_clauses.append(f"{args.drop_unlabeled_target} IS NOT NULL")

    query = text(f"SELECT * FROM training_features WHERE {' AND '.join(where_clauses)} ORDER BY trade_date ASC, ts_code ASC")
    with repo._db.engine.connect() as conn:
        frame = pd.read_sql_query(query, conn, params=params)

    if "id" in frame.columns:
        frame = frame.drop(columns=["id"])
    if "created_at" in frame.columns:
        frame = frame.drop(columns=["created_at"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    result = {
        "exported": True,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "drop_unlabeled_target": args.drop_unlabeled_target or None,
        "output": str(output_path),
    }
    logger.info("Training features exported: %s", result)
    print_json(result)


if __name__ == "__main__":
    main()

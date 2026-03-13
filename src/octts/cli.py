from __future__ import annotations

import argparse
import json

from octts.api import _build_pipeline
from octts.schemas.report import AnalysisRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCTTS analysis manually.")
    parser.add_argument("--phase", choices=["morning", "afternoon", "review"], required=True)
    parser.add_argument("--stock", action="append", dest="stocks", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    pipeline = _build_pipeline()
    result = pipeline.run(
        AnalysisRequest(
            phase=args.phase,
            stock_pool=args.stocks,
            trade_date=args.trade_date,
            notify=not args.no_notify,
        )
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import asyncio
import json

from octts.api import _build_pipeline
from octts.config import get_settings
from octts.schemas.report import AnalysisRequest
from octts.services.enhanced_screening_scheduler import EnhancedScreeningScheduler
from octts.services.execution_confirmation_service import ExecutionConfirmationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCTTS analysis manually.")
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze", help="Run stock analysis pipeline")
    analyze_parser.add_argument("--phase", choices=["morning", "afternoon", "review"], required=True)
    analyze_parser.add_argument("--stock", action="append", dest="stocks", default=None)
    analyze_parser.add_argument("--trade-date", default=None)
    analyze_parser.add_argument("--no-notify", action="store_true")

    candidate_parser = subparsers.add_parser("screen-candidates", help="Run post-close candidate screening")
    candidate_parser.add_argument("--no-notify", action="store_true")

    confirm_parser = subparsers.add_parser("confirm-execution", help="Run pre-open execution confirmation")
    confirm_parser.add_argument("--source-trade-date", default=None)
    confirm_parser.add_argument("--force", action="store_true")

    parser.add_argument("--phase", choices=["morning", "afternoon", "review"], required=False)
    parser.add_argument("--stock", action="append", dest="legacy_stocks", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    if args.command == "screen-candidates":
        settings = get_settings()
        original_notify = settings.screening_notify
        if args.no_notify:
            settings.screening_notify = False
        try:
            result = asyncio.run(EnhancedScreeningScheduler(settings).run_intelligent_screening())
        finally:
            settings.screening_notify = original_notify
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "confirm-execution":
        result = asyncio.run(
            ExecutionConfirmationService(get_settings()).run_pre_open_confirmation(
                source_trade_date=args.source_trade_date,
                force=args.force,
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    phase = getattr(args, "phase", None)
    stocks = getattr(args, "stocks", None) or getattr(args, "legacy_stocks", None)
    if args.command == "analyze" or phase:
        pipeline = _build_pipeline()
        result = pipeline.run(
            AnalysisRequest(
                phase=phase,
                stock_pool=stocks,
                trade_date=args.trade_date,
                notify=not args.no_notify,
            )
        )
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

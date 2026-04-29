from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import func

from octts.config import get_settings
from octts.models.screening_models import (
    DatabaseManager,
    MarketDaily,
    MarketDailyBasic,
    MarketLimitListDaily,
    MarketMoneyflowDaily,
    MarketTopListDaily,
)
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import configure_tool_logging, print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local raw market data coverage for a date range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--field-detail",
        action="store_true",
        help="Also inspect key field coverage used by full-universe risk backtests.",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")

    settings = get_settings()
    logger = configure_tool_logging(settings, "inspect_market_data_coverage")
    repo = MarketRawDataRepository(settings.database_url)
    payload = repo.summarize_market_data_coverage(
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    logger.info(
        "Market data coverage generated: start_date=%s, end_date=%s, trading_days=%s",
        start_date.isoformat(),
        end_date.isoformat(),
        payload.get("trading_days"),
    )
    output = _trim_payload(payload)
    if args.field_detail:
        output["field_detail"] = _build_field_detail(
            database_url=settings.database_url,
            trading_dates=payload.get("trading_dates") or [],
        )
    print_json(output)


def _trim_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    datasets = payload.get("datasets") or {}
    trimmed = {
        "start_date": payload.get("start_date"),
        "end_date": payload.get("end_date"),
        "trading_days": payload.get("trading_days"),
        "trading_dates": payload.get("trading_dates"),
        "datasets": {},
    }
    for name, data in datasets.items():
        trimmed["datasets"][name] = {
            "covered_trade_days": data.get("covered_trade_days"),
            "missing_trade_days": data.get("missing_trade_days"),
            "missing_dates": data.get("missing_dates"),
            "min_rows": data.get("min_rows"),
            "max_rows": data.get("max_rows"),
        }
    return trimmed


def _build_field_detail(*, database_url: str, trading_dates: list[str]) -> Dict[str, Any]:
    db = DatabaseManager(database_url)
    session = db.get_session()
    try:
        details: Dict[str, Any] = {}
        for trade_date_text in trading_dates:
            trade_date = datetime.strptime(trade_date_text, "%Y%m%d").date()
            daily_count = _count_rows(session, MarketDaily, trade_date)
            basic_count = _count_rows(session, MarketDailyBasic, trade_date)
            moneyflow_count = _count_rows(session, MarketMoneyflowDaily, trade_date)
            limit_count = _count_rows(session, MarketLimitListDaily, trade_date)
            top_count = _count_rows(session, MarketTopListDaily, trade_date)
            daily_codes = _distinct_codes(session, MarketDaily, trade_date)
            basic_codes = _distinct_codes(session, MarketDailyBasic, trade_date)
            moneyflow_codes = _distinct_codes(session, MarketMoneyflowDaily, trade_date)
            details[trade_date_text] = {
                "row_counts": {
                    "market_daily": daily_count,
                    "market_daily_basic": basic_count,
                    "market_moneyflow_daily": moneyflow_count,
                    "market_limit_list_daily": limit_count,
                    "market_top_list_daily": top_count,
                },
                "code_counts": {
                    "daily_codes": len(daily_codes),
                    "daily_basic_codes": len(basic_codes),
                    "daily_and_basic_codes": len(daily_codes & basic_codes),
                    "daily_missing_basic_codes": len(daily_codes - basic_codes),
                    "basic_missing_daily_codes": len(basic_codes - daily_codes),
                    "moneyflow_codes": len(moneyflow_codes),
                    "moneyflow_coverage_vs_basic": _ratio(len(moneyflow_codes & basic_codes), len(basic_codes)),
                },
                "required_daily_fields": _field_stats(
                    session,
                    MarketDaily,
                    trade_date,
                    daily_count,
                    {
                        "open": MarketDaily.open,
                        "high": MarketDaily.high,
                        "low": MarketDaily.low,
                        "close": MarketDaily.close,
                        "pct_chg": MarketDaily.pct_chg,
                        "vol": MarketDaily.vol,
                        "amount": MarketDaily.amount,
                    },
                ),
                "required_daily_basic_fields": _field_stats(
                    session,
                    MarketDailyBasic,
                    trade_date,
                    basic_count,
                    {
                        "turnover_rate": MarketDailyBasic.turnover_rate,
                        "volume_ratio": MarketDailyBasic.volume_ratio,
                        "total_mv": MarketDailyBasic.total_mv,
                        "pe_ttm": MarketDailyBasic.pe_ttm,
                        "pb": MarketDailyBasic.pb,
                    },
                ),
                "risk_moneyflow_fields": _field_stats(
                    session,
                    MarketMoneyflowDaily,
                    trade_date,
                    moneyflow_count,
                    {
                        "net_mf_amount": MarketMoneyflowDaily.net_mf_amount,
                        "buy_lg_amount": MarketMoneyflowDaily.buy_lg_amount,
                        "sell_lg_amount": MarketMoneyflowDaily.sell_lg_amount,
                        "buy_elg_amount": MarketMoneyflowDaily.buy_elg_amount,
                        "sell_elg_amount": MarketMoneyflowDaily.sell_elg_amount,
                    },
                ),
                "risk_limit_list_fields_naturally_sparse": _field_stats(
                    session,
                    MarketLimitListDaily,
                    trade_date,
                    limit_count,
                    {
                        "open_times": MarketLimitListDaily.open_times,
                        "first_time": MarketLimitListDaily.first_time,
                        "last_time": MarketLimitListDaily.last_time,
                    },
                ),
                "risk_top_list_fields_naturally_sparse": _field_stats(
                    session,
                    MarketTopListDaily,
                    trade_date,
                    top_count,
                    {
                        "net_amount": MarketTopListDaily.net_amount,
                        "net_rate": MarketTopListDaily.net_rate,
                    },
                ),
            }
        return details
    finally:
        session.close()


def _count_rows(session: Any, model: Any, trade_date: Any) -> int:
    return int(session.query(func.count()).select_from(model).filter(model.trade_date == trade_date).scalar() or 0)


def _distinct_codes(session: Any, model: Any, trade_date: Any) -> set[str]:
    return {
        str(row[0]).strip().upper()
        for row in session.query(model.ts_code).filter(model.trade_date == trade_date).distinct().all()
        if row and row[0]
    }


def _field_stats(session: Any, model: Any, trade_date: Any, total_count: int, columns: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for name, column in columns.items():
        non_null_count = int(
            session.query(func.count())
            .select_from(model)
            .filter(model.trade_date == trade_date, column.isnot(None))
            .scalar()
            or 0
        )
        result[name] = {
            "non_null": non_null_count,
            "total": total_count,
            "coverage": _ratio(non_null_count, total_count),
        }
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from typing import Any, List, Dict

from octts.config import get_settings
from octts.models.screening_models import (
    DatabaseManager,
    MarketIndustryMoneyflowDaily,
    MarketLimitListDaily,
    MarketMoneyflowDaily,
    MarketMoneyflowMarketDaily,
    MarketTopListDaily,
)
from octts.tools.common import configure_tool_logging, print_json
from octts.services.market_raw_data_repository import MarketRawDataRepository

DEFAULT_EXCHANGE = "SSE"
MONEYFLOW_LOOKBACK_DAYS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill relay and funds raw market data into SQLite.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--ts-code", default="600000.SH", help="Sample ts_code for stock-level probes like moneyflow")
    parser.add_argument("--skip-moneyflow", action="store_true")
    parser.add_argument("--skip-top-list", action="store_true")
    parser.add_argument("--skip-limit-list", action="store_true")
    parser.add_argument("--skip-industry-moneyflow", action="store_true")
    parser.add_argument("--skip-market-moneyflow", action="store_true")
    parser.add_argument("--coverage-only", action="store_true", help="Print local dataset coverage without fetching new rows")
    parser.add_argument("--moneyflow-all-stocks", action="store_true", help="Backfill stock-level moneyflow for all listed stocks instead of a single sample ts_code")
    parser.add_argument("--stock-list-status", default="L", help="Stock list status used with --moneyflow-all-stocks (L/D/P)")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--max-trade-days", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.2)
    parser.add_argument("--sleep-every", type=int, default=3)
    parser.add_argument("--batch-sleep-seconds", type=float, default=6.0)
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "backfill_relay_market_data")
    logger.info(
        "Relay market backfill started: start_date=%s, end_date=%s, only_missing=%s, force_refresh=%s",
        args.start_date,
        args.end_date,
        args.only_missing,
        args.force_refresh,
    )

    start_date = _parse_cli_date(args.start_date)
    end_date = _parse_cli_date(args.end_date)
    if start_date > end_date:
        raise ValueError("start-date must be <= end-date")
    if args.only_missing and args.force_refresh:
        raise ValueError("only-missing and force-refresh cannot be used together")

    client = _build_tushare_client(settings)
    db = DatabaseManager(settings.database_url)
    repo = MarketRawDataRepository(settings.database_url)

    if args.coverage_only:
        coverage = repo.summarize_market_data_coverage(
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            exchange=DEFAULT_EXCHANGE,
        )
        logger.info("Relay market coverage summary generated: trading_days=%s", coverage.get("trading_days"))
        print_json(coverage)
        return

    summary: dict[str, Any] = {
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "moneyflow": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
        "top_list": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
        "limit_list": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
        "industry_moneyflow": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
        "market_moneyflow": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
    }

    trading_dates = _list_open_trading_dates(repo, client, start_date=start_date, end_date=end_date)
    if args.max_trade_days is not None:
        trading_dates = trading_dates[: max(args.max_trade_days, 0)]

    sleep_every = max(int(args.sleep_every), 0)
    sleep_seconds = max(float(args.sleep_seconds), 0.0)
    batch_sleep_seconds = max(float(args.batch_sleep_seconds), 0.0)

    for index, trade_day in enumerate(trading_dates, start=1):
        trade_date_text = trade_day.strftime("%Y%m%d")
        logger.info("Relay market backfill start: trade_date=%s (%s/%s)", trade_day.isoformat(), index, len(trading_dates))

        if not args.skip_moneyflow:
            _process_moneyflow_dataset(
                logger=logger,
                client=client,
                db=db,
                repo=repo,
                trade_day=trade_day,
                sample_ts_code=args.ts_code,
                all_stocks=bool(args.moneyflow_all_stocks),
                stock_list_status=str(args.stock_list_status or "L").strip() or "L",
                force_refresh=bool(args.force_refresh),
                summary=summary["moneyflow"],
                only_missing=bool(args.only_missing),
            )
            _throttle_if_needed(sleep_seconds)

        if not args.skip_top_list:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketTopListDaily,
                dataset_name="top_list",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.top_list(trade_date=trade_date_text)),
                upsert_rows=lambda rows: db.upsert_market_top_list_daily(rows, force_refresh=args.force_refresh),
                summary=summary["top_list"],
                only_missing=args.only_missing,
            )
            _throttle_if_needed(sleep_seconds)

        if not args.skip_limit_list:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketLimitListDaily,
                dataset_name="limit_list",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.limit_list_d(trade_date=trade_date_text)),
                upsert_rows=lambda rows: db.upsert_market_limit_list_daily(rows, force_refresh=args.force_refresh),
                summary=summary["limit_list"],
                only_missing=args.only_missing,
            )
            _throttle_if_needed(sleep_seconds)

        if not args.skip_industry_moneyflow:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketIndustryMoneyflowDaily,
                dataset_name="industry_moneyflow",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.moneyflow_ind_ths(trade_date=trade_date_text)),
                upsert_rows=lambda rows: db.upsert_market_industry_moneyflow_daily(rows, force_refresh=args.force_refresh),
                summary=summary["industry_moneyflow"],
                only_missing=args.only_missing,
            )
            _throttle_if_needed(sleep_seconds)

        if not args.skip_market_moneyflow:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketMoneyflowMarketDaily,
                dataset_name="market_moneyflow",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.moneyflow_mkt_dc(start_date=trade_date_text, end_date=trade_date_text)),
                upsert_rows=lambda rows: db.upsert_market_moneyflow_market_daily(rows, force_refresh=args.force_refresh),
                summary=summary["market_moneyflow"],
                only_missing=args.only_missing,
            )
            _throttle_if_needed(sleep_seconds)

        if sleep_every > 0 and index % sleep_every == 0 and batch_sleep_seconds > 0:
            logger.info(
                "Relay market backfill throttle: processed_trade_days=%s/%s, batch_sleep_seconds=%.2f",
                index,
                len(trading_dates),
                batch_sleep_seconds,
            )
            time.sleep(batch_sleep_seconds)

    logger.info("Relay market backfill complete: %s", summary)
    print_json(summary)


def _build_tushare_client(settings):
    from octts.clients.tushare_client import TushareClient

    return TushareClient(settings)


def _parse_cli_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _list_open_trading_dates(repo: MarketRawDataRepository, client, *, start_date: date, end_date: date) -> List[date]:
    local_values = repo.get_calendar_dates(
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        exchange=DEFAULT_EXCHANGE,
    )
    if local_values:
        return [datetime.strptime(value, "%Y%m%d").date() for value in local_values]

    values = client.fetch_trading_dates(
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
    )
    return [datetime.strptime(value, "%Y%m%d").date() for value in values]


def _process_moneyflow_dataset(
    *,
    logger,
    client,
    db: DatabaseManager,
    repo,
    trade_day: date,
    sample_ts_code: str,
    all_stocks: bool,
    stock_list_status: str,
    force_refresh: bool,
    summary: Dict[str, Any],
    only_missing: bool,
) -> None:
    summary["trade_days"] += 1
    trade_date_text = trade_day.strftime("%Y%m%d")
    if only_missing and all_stocks and repo.count_rows_for_trade_date(model=MarketMoneyflowDaily, trade_date=trade_date_text) > 0:
        summary["skipped_days"] += 1
        logger.info("Skip moneyflow backfill: trade_date=%s, reason=already_present", trade_day.isoformat())
        return
    if only_missing and not all_stocks and db.has_market_data_for_trade_date(model=MarketMoneyflowDaily, trade_date=trade_day):
        summary["skipped_days"] += 1
        logger.info("Skip moneyflow backfill: trade_date=%s, reason=sample_probe_already_present", trade_day.isoformat())
        return

    if all_stocks:
        rows = _fetch_all_stock_moneyflow_rows(
            client=client,
            trade_day=trade_day,
            stock_list_status=stock_list_status,
            logger=logger,
        )
    else:
        rows = _fetch_rows(client._pro.moneyflow(ts_code=sample_ts_code, start_date=trade_date_text, end_date=trade_date_text))

    inserted_rows = db.upsert_market_moneyflow_daily(rows, force_refresh=force_refresh)
    summary["fetched_rows"] += len(rows)
    summary["inserted_rows"] += inserted_rows
    logger.info(
        "moneyflow backfill complete: trade_date=%s, fetched_rows=%s, inserted_rows=%s, all_stocks=%s",
        trade_day.isoformat(),
        len(rows),
        inserted_rows,
        all_stocks,
    )


def _fetch_all_stock_moneyflow_rows(*, client, trade_day: date, stock_list_status: str, logger) -> List[dict[str, Any]]:
    trade_date_text = trade_day.strftime("%Y%m%d")
    start_date_text = (trade_day - timedelta(days=MONEYFLOW_LOOKBACK_DAYS)).strftime("%Y%m%d")
    stock_rows = client.fetch_stock_list(list_status=stock_list_status)
    ts_codes = [str(row.get("ts_code") or "").strip() for row in stock_rows if str(row.get("ts_code") or "").strip()]
    aggregated: Dict[tuple[str, str], dict[str, Any]] = {}
    for index, ts_code in enumerate(ts_codes, start=1):
        rows = _fetch_rows(client._pro.moneyflow(ts_code=ts_code, start_date=start_date_text, end_date=trade_date_text))
        for row in rows:
            row_trade_date = str(row.get("trade_date") or "").strip()
            row_ts_code = str(row.get("ts_code") or ts_code).strip()
            if row_trade_date != trade_date_text or not row_ts_code:
                continue
            aggregated[(row_trade_date, row_ts_code)] = row
        if index % 200 == 0:
            logger.info(
                "Moneyflow all-stock fetch progress: trade_date=%s, processed=%s/%s, matched_rows=%s",
                trade_day.isoformat(),
                index,
                len(ts_codes),
                len(aggregated),
            )
    logger.info(
        "Moneyflow all-stock fetch complete: trade_date=%s, stock_count=%s, matched_rows=%s",
        trade_day.isoformat(),
        len(ts_codes),
        len(aggregated),
    )
    return list(aggregated.values())


def _process_trade_date_dataset(
    *,
    logger,
    db: DatabaseManager,
    model,
    dataset_name: str,
    trade_day: date,
    fetch_rows,
    upsert_rows,
    summary: dict[str, Any],
    only_missing: bool,
) -> None:
    summary["trade_days"] += 1
    if only_missing and db.has_market_data_for_trade_date(model=model, trade_date=trade_day):
        summary["skipped_days"] += 1
        logger.info("Skip %s backfill: trade_date=%s, reason=already_present", dataset_name, trade_day.isoformat())
        return

    rows = fetch_rows()
    inserted_rows = upsert_rows(rows)
    summary["fetched_rows"] += len(rows)
    summary["inserted_rows"] += inserted_rows
    logger.info(
        "%s backfill complete: trade_date=%s, fetched_rows=%s, inserted_rows=%s",
        dataset_name,
        trade_day.isoformat(),
        len(rows),
        inserted_rows,
    )


def _throttle_if_needed(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _fetch_rows(df) -> List[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return list(df.to_dict(orient="records"))


if __name__ == "__main__":
    main()

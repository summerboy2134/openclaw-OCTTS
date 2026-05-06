from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from typing import Any, Iterable, List, Sequence

from octts.config import get_settings
from octts.models.screening_models import DatabaseManager, MarketAdjFactor, MarketDaily, MarketDailyBasic, MarketStockBasic
from octts.tools.common import configure_tool_logging, print_json
from octts.services.market_raw_data_repository import MarketRawDataRepository


RAW_DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
RAW_DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
    "dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
)
RAW_ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"
DEFAULT_EXCHANGE = "SSE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw market data into SQLite.")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--stock-basic-only", action="store_true", help="Only backfill market_stock_basic from stock_basic.")
    parser.add_argument("--skip-trade-cal", action="store_true")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-daily-basic", action="store_true")
    parser.add_argument("--skip-adj-factor", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--max-trade-days", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-every", type=int, default=3)
    parser.add_argument("--batch-sleep-seconds", type=float, default=6.0)
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "backfill_market_raw_data")
    client = _build_tushare_client(settings)
    db = DatabaseManager(settings.database_url)

    if args.stock_basic_only:
        rows = _fetch_stock_basic_rows(client)
        inserted = _upsert_market_stock_basic(db, rows)
        summary = {
            "stock_basic": {
                "fetched_rows": len(rows),
                "inserted_rows": inserted,
                "force_refresh": bool(args.force_refresh),
            }
        }
        logger.info("Stock basic backfill complete: %s", summary)
        print_json(summary)
        return

    if not args.start_date or not args.end_date:
        raise ValueError("--start-date and --end-date are required unless --stock-basic-only is used")

    logger.info(
        "Raw market backfill started: start_date=%s, end_date=%s, only_missing=%s, force_refresh=%s",
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

    repo = MarketRawDataRepository(settings.database_url)

    summary: dict[str, Any] = {
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "trade_calendar": {"fetched": 0, "inserted": 0, "skipped": False},
        "daily": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
        "daily_basic": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
        "adj_factor": {"trade_days": 0, "fetched_rows": 0, "inserted_rows": 0, "skipped_days": 0},
    }

    if not args.skip_trade_cal:
        if args.only_missing and repo.has_full_trade_calendar_range(
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            exchange=DEFAULT_EXCHANGE,
        ):
            logger.info("Skip trade calendar backfill: start_date=%s, end_date=%s, reason=already_present", start_date, end_date)
            summary["trade_calendar"]["skipped"] = True
        else:
            calendar_rows = _fetch_trade_calendar_rows(client, start_date=start_date, end_date=end_date)
            inserted = db.upsert_market_trade_calendar(
                calendar_rows,
                exchange=DEFAULT_EXCHANGE,
                force_refresh=args.force_refresh,
            )
            summary["trade_calendar"]["fetched"] = len(calendar_rows)
            summary["trade_calendar"]["inserted"] = inserted
            logger.info(
                "Trade calendar backfill complete: fetched=%s, inserted=%s",
                len(calendar_rows),
                inserted,
            )

    trading_dates = _list_open_trading_dates(repo, client, start_date=start_date, end_date=end_date)
    if args.max_trade_days is not None:
        trading_dates = trading_dates[: max(args.max_trade_days, 0)]

    sleep_every = max(int(args.sleep_every), 0)
    sleep_seconds = max(float(args.sleep_seconds), 0.0)
    batch_sleep_seconds = max(float(args.batch_sleep_seconds), 0.0)

    for index, trade_day in enumerate(trading_dates, start=1):
        trade_date_text = trade_day.strftime("%Y%m%d")
        logger.info("Raw market backfill start: trade_date=%s (%s/%s)", trade_day.isoformat(), index, len(trading_dates))

        if not args.skip_daily:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketDaily,
                dataset_name="daily",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.daily(trade_date=trade_date_text, fields=RAW_DAILY_FIELDS)),
                upsert_rows=lambda rows: db.upsert_market_daily(rows, force_refresh=args.force_refresh),
                summary=summary["daily"],
                only_missing=args.only_missing,
            )

        if not args.skip_daily_basic:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketDailyBasic,
                dataset_name="daily_basic",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.daily_basic(trade_date=trade_date_text, fields=RAW_DAILY_BASIC_FIELDS)),
                upsert_rows=lambda rows: db.upsert_market_daily_basic(rows, force_refresh=args.force_refresh),
                summary=summary["daily_basic"],
                only_missing=args.only_missing,
            )

        if not args.skip_adj_factor:
            _process_trade_date_dataset(
                logger=logger,
                db=db,
                model=MarketAdjFactor,
                dataset_name="adj_factor",
                trade_day=trade_day,
                fetch_rows=lambda: _fetch_rows(client._pro.adj_factor(trade_date=trade_date_text, fields=RAW_ADJ_FACTOR_FIELDS)),
                upsert_rows=lambda rows: db.upsert_market_adj_factor(rows, force_refresh=args.force_refresh),
                summary=summary["adj_factor"],
                only_missing=args.only_missing,
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if sleep_every > 0 and index % sleep_every == 0 and batch_sleep_seconds > 0:
            logger.info(
                "Raw market backfill throttle: processed_trade_days=%s/%s, batch_sleep_seconds=%.2f",
                index,
                len(trading_dates),
                batch_sleep_seconds,
            )
            time.sleep(batch_sleep_seconds)

    logger.info("Raw market backfill complete: %s", summary)
    print_json(summary)


def _build_tushare_client(settings):
    from octts.clients.tushare_client import TushareClient

    return TushareClient(settings)


def _parse_cli_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _fetch_trade_calendar_rows(client, *, start_date: date, end_date: date) -> List[dict[str, Any]]:
    df = client._pro.trade_cal(
        exchange=DEFAULT_EXCHANGE,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    return _fetch_rows(df)


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


def _fetch_rows(df) -> List[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return list(df.to_dict(orient="records"))


def _fetch_stock_basic_rows(client) -> List[dict[str, Any]]:
    df = client._pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,is_hs",
    )
    return _fetch_rows(df)


def _upsert_market_stock_basic(db: DatabaseManager, rows: Iterable[dict[str, Any]]) -> int:
    session = db.get_session()
    count = 0
    try:
        for row in rows:
            ts_code = str(row.get("ts_code") or "").strip().upper()
            if not ts_code:
                continue
            session.merge(
                MarketStockBasic(
                    ts_code=ts_code,
                    symbol=_clean_text(row.get("symbol")),
                    name=_clean_text(row.get("name")),
                    area=_clean_text(row.get("area")),
                    industry=_clean_text(row.get("industry")),
                    market=_clean_text(row.get("market")),
                    list_date=_parse_optional_yyyymmdd(row.get("list_date")),
                    delist_date=_parse_optional_yyyymmdd(row.get("delist_date")),
                    is_hs=_clean_text(row.get("is_hs")),
                )
            )
            count += 1
        session.commit()
        return count
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_optional_yyyymmdd(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError:
        return None


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from octts.config import get_settings
from octts.models.screening_models import DatabaseManager
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.backfill_market_raw_data import RAW_DAILY_BASIC_FIELDS, RAW_DAILY_FIELDS
from octts.tools.common import configure_tool_logging, print_json

DEFAULT_EXCHANGE = "SSE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair local market_daily close/pre_close discontinuities by refetching affected trade dates.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--window-days", type=int, default=2, help="Refetch +/- N open trade days around each anomaly")
    parser.add_argument("--ts-code", action="append", default=[], help="Optional code filter; repeatable")
    parser.add_argument("--apply", action="store_true", help="Actually modify DB. Default is dry-run")
    parser.add_argument("--skip-daily-basic", action="store_true")
    parser.add_argument("--max-repair-days", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-every", type=int, default=3)
    parser.add_argument("--batch-sleep-seconds", type=float, default=5.0)
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "repair_market_daily_discontinuities")
    db_path = sqlite_path(settings.database_url)
    start = norm_date(args.start_date)
    end = norm_date(args.end_date)
    codes = [str(x).strip().upper() for x in args.ts_code if str(x).strip()]
    threshold = abs(float(args.threshold_pct)) / 100.0

    before = scan(db_path, start=start, end=end, threshold=threshold, codes=codes)
    repo = MarketRawDataRepository(settings.database_url)
    trade_dates = repo.list_trading_dates(start_date=compact(start), end_date=compact(end), exchange=DEFAULT_EXCHANGE)
    if not trade_dates:
        from octts.clients.tushare_client import TushareClient
        trade_dates = TushareClient(settings).fetch_trading_dates(start_date=compact(start), end_date=compact(end))
    trade_dates = [norm_date(x) for x in trade_dates]
    repair_dates = build_repair_dates(before, trade_dates, max(int(args.window_days), 0))
    if args.max_repair_days is not None:
        repair_dates = repair_dates[: max(int(args.max_repair_days), 0)]

    summary: dict[str, Any] = {
        "database": str(db_path),
        "dry_run": not args.apply,
        "start_date": start,
        "end_date": end,
        "threshold_pct": args.threshold_pct,
        "window_days": args.window_days,
        "ts_codes": codes,
        "before": summarize(before),
        "repair_trade_dates": repair_dates,
        "backup_path": None,
        "daily": {"fetched_rows": 0, "upserted_rows": 0, "empty_days": 0},
        "daily_basic": {"fetched_rows": 0, "upserted_rows": 0, "empty_days": 0},
        "after": None,
    }
    if not args.apply:
        print_json(summary)
        return

    backup = db_path.with_name(f"{db_path.name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(db_path, backup)
    summary["backup_path"] = str(backup)
    logger.info("Database backup created: %s", backup)

    from octts.clients.tushare_client import TushareClient
    client = TushareClient(settings)
    db = DatabaseManager(settings.database_url)
    for idx, d in enumerate(repair_dates, start=1):
        td = compact(d)
        logger.info("Repair refresh: trade_date=%s (%s/%s)", td, idx, len(repair_dates))
        daily_rows = rows(client._pro.daily(trade_date=td, fields=RAW_DAILY_FIELDS))
        if daily_rows:
            summary["daily"]["fetched_rows"] += len(daily_rows)
            summary["daily"]["upserted_rows"] += db.upsert_market_daily(daily_rows, force_refresh=True)
        else:
            summary["daily"]["empty_days"] += 1
            logger.warning("No daily rows fetched for %s", td)
        if not args.skip_daily_basic:
            basic_rows = rows(client._pro.daily_basic(trade_date=td, fields=RAW_DAILY_BASIC_FIELDS))
            if basic_rows:
                summary["daily_basic"]["fetched_rows"] += len(basic_rows)
                summary["daily_basic"]["upserted_rows"] += db.upsert_market_daily_basic(basic_rows, force_refresh=True)
            else:
                summary["daily_basic"]["empty_days"] += 1
                logger.warning("No daily_basic rows fetched for %s", td)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        if args.sleep_every > 0 and idx % args.sleep_every == 0 and args.batch_sleep_seconds > 0:
            time.sleep(args.batch_sleep_seconds)

    after = scan(db_path, start=start, end=end, threshold=threshold, codes=codes)
    summary["after"] = summarize(after)
    print_json(summary)


def scan(db_path: Path, *, start: str, end: str, threshold: float, codes: list[str]) -> list[dict[str, Any]]:
    code_sql = ""
    params: list[Any] = [end]
    if codes:
        code_sql = " and ts_code in (" + ",".join("?" for _ in codes) + ")"
        params.extend(codes)
    params.extend([start, end, threshold])
    sql = f"""
    with x as (
      select ts_code, trade_date, close, pre_close,
             lag(trade_date) over(partition by ts_code order by trade_date) prev_trade_date,
             lag(close) over(partition by ts_code order by trade_date) prev_close
      from market_daily where trade_date <= ? {code_sql}
    )
    select ts_code, trade_date, prev_trade_date, prev_close, pre_close, close,
           (prev_close / pre_close - 1.0) * 100.0 gap_pct
    from x
    where trade_date >= ? and trade_date <= ?
      and prev_close is not null and pre_close is not null and pre_close != 0
      and abs(prev_close / pre_close - 1.0) > ?
    order by abs(prev_close / pre_close - 1.0) desc, trade_date, ts_code
    """
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(sql, params)]


def build_repair_dates(items: list[dict[str, Any]], trade_dates: list[str], window: int) -> list[str]:
    idx = {d: i for i, d in enumerate(trade_dates)}
    picked: set[str] = set()
    for item in items:
        d = norm_date(item["trade_date"])
        if d not in idx:
            picked.add(d)
            if item.get("prev_trade_date"):
                picked.add(norm_date(item["prev_trade_date"]))
            continue
        i = idx[d]
        picked.update(trade_dates[max(0, i - window): min(len(trade_dates), i + window + 1)])
    return sorted(picked)


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_date: dict[str, int] = {}
    by_code: dict[str, int] = {}
    examples = []
    for item in items:
        d = norm_date(item["trade_date"])
        c = item["ts_code"]
        by_date[d] = by_date.get(d, 0) + 1
        by_code[c] = by_code.get(c, 0) + 1
    for item in items[:50]:
        examples.append({
            "ts_code": item["ts_code"],
            "trade_date": norm_date(item["trade_date"]),
            "prev_trade_date": norm_date(item["prev_trade_date"]) if item.get("prev_trade_date") else None,
            "prev_close": item.get("prev_close"),
            "pre_close": item.get("pre_close"),
            "close": item.get("close"),
            "gap_pct": round(float(item["gap_pct"]), 4) if item.get("gap_pct") is not None else None,
        })
    return {
        "count": len(items),
        "by_date_top20": sorted(by_date.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        "by_code_top20": sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
        "examples_top50": examples,
    }


def rows(df: Any) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return list(df.to_dict(orient="records"))


def sqlite_path(url: str) -> Path:
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError(f"Only sqlite URLs are supported: {url}")
    p = Path(url[len(prefix):])
    return p if p.is_absolute() else Path.cwd() / p


def norm_date(v: Any) -> str:
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError(f"Invalid date: {v}")


def compact(v: Any) -> str:
    return norm_date(v).replace("-", "")


if __name__ == "__main__":
    main()

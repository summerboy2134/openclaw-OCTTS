"""补全缺失的日线历史数据"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.tools.common import configure_tool_logging, print_json


def find_missing_stocks(
    repo: MarketRawDataRepository,
    trade_dates: List[str],
    lookback_days: int = 120,
) -> Dict[str, Dict[str, Any]]:
    """找出在 snapshot 中但本地数据库缺失日线数据的股票"""
    import json
    from pathlib import Path

    all_missing: Dict[str, Dict[str, Any]] = {}

    for trade_date in trade_dates:
        snapshot_path = Path(f"memory/history/screening_cache/{trade_date}.json")
        if not snapshot_path.exists():
            continue

        with snapshot_path.open("r") as f:
            snapshot = json.load(f)

        daily = snapshot.get("daily", {})
        stocks = snapshot.get("stocks", [])

        stock_codes = {s.get("ts_code") for s in stocks if s.get("ts_code")}

        # 计算回溯起始日期
        anchor = datetime.strptime(trade_date, "%Y%m%d")
        start_date = (anchor - timedelta(days=lookback_days)).strftime("%Y%m%d")

        for ts_code in stock_codes:
            daily_rows = daily.get(ts_code, [])
            if not daily_rows or len(daily_rows) < 20:
                # 检查本地数据库
                local_rows = repo.get_daily_range(
                    ts_code=ts_code, start_date=start_date, end_date=trade_date
                )
                if not local_rows:
                    if ts_code not in all_missing:
                        all_missing[ts_code] = {
                            "first_seen_date": trade_date,
                            "dates_checked": [],
                        }
                    all_missing[ts_code]["dates_checked"].append(trade_date)

    return all_missing


def backfill_daily_data(
    ts_codes: List[str],
    start_date: str,
    end_date: str,
    *,
    throttle_seconds: float = 0.5,
    max_retries: int = 3,
    logger,
) -> Dict[str, Any]:
    """从 Tushare API 获取缺失的日线数据"""
    from octts.clients.tushare_client import TushareClient

    settings = get_settings()
    client = TushareClient(settings)
    repo = MarketRawDataRepository(settings.database_url)

    results: Dict[str, Any] = {
        "success": [],
        "failed": [],
        "empty": [],
        "total_fetched_rows": 0,
    }

    total = len(ts_codes)
    for index, ts_code in enumerate(ts_codes, start=1):
        logger.info(
            "Backfill progress: %s/%s - fetching %s",
            index, total, ts_code
        )

        rows = None
        last_error = None
        for attempt in range(max_retries):
            try:
                df = client._call_pro_bar(
                    ts_code=ts_code,
                    asset="E",
                    start_date=start_date,
                    end_date=end_date,
                    freq="D",
                    adj="qfq",
                )
                if df is not None and not df.empty:
                    rows = df.to_dict(orient="records")
                break
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Backfill retry %s/%s for %s: %s",
                    attempt + 1, max_retries, ts_code, exc
                )
                time.sleep(throttle_seconds * 2)

        if rows:
            # 保存到本地数据库
            raw_rows = []
            for row in rows:
                raw_rows.append({
                    "ts_code": row.get("ts_code"),
                    "trade_date": row.get("trade_date"),
                    "open": _safe_float(row.get("open")),
                    "high": _safe_float(row.get("high")),
                    "low": _safe_float(row.get("low")),
                    "close": _safe_float(row.get("close")),
                    "pre_close": _safe_float(row.get("pre_close")),
                    "change": _safe_float(row.get("change")),
                    "pct_chg": _safe_float(row.get("pct_chg")),
                    "vol": _safe_float(row.get("vol")),
                    "amount": _safe_float(row.get("amount")),
                })
            saved = repo.save_daily(raw_rows)
            results["success"].append(ts_code)
            results["total_fetched_rows"] += len(rows)
            logger.info(
                "Backfill success: %s fetched %s rows, saved %s rows",
                ts_code, len(rows), saved
            )
        elif last_error:
            results["failed"].append({"ts_code": ts_code, "error": last_error})
            logger.warning("Backfill failed: %s - %s", ts_code, last_error)
        else:
            results["empty"].append(ts_code)
            logger.info("Backfill empty: %s has no data in range", ts_code)

        # 限流
        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    return results


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="补全缺失的日线历史数据")
    parser.add_argument(
        "--trade-dates",
        default="20260317,20260318,20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260330",
        help="要检查的交易日期，逗号分隔",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="回溯天数",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.5,
        help="每次 API 调用间隔秒数",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="每只股票最大重试次数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查缺失，不执行回填",
    )
    parser.add_argument(
        "--output-file",
        help="可选，保存结果到文件",
    )
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_tool_logging(settings, "backfill_missing_daily_data")
    repo = MarketRawDataRepository(settings.database_url)

    trade_dates = [d.strip() for d in args.trade_dates.split(",") if d.strip()]

    # 1. 找出缺失的股票
    logger.info("Scanning for missing daily data across %s trade dates", len(trade_dates))
    missing = find_missing_stocks(repo, trade_dates, lookback_days=args.lookback_days)

    if not missing:
        logger.info("No missing daily data found")
        print_json({"status": "complete", "missing_count": 0, "message": "No missing data"})
        return

    missing_codes = sorted(missing.keys())
    logger.info("Found %s stocks with missing daily data: %s", len(missing_codes), missing_codes)

    if args.dry_run:
        print_json({
            "status": "dry_run",
            "missing_count": len(missing_codes),
            "missing_codes": missing_codes,
            "details": missing,
        })
        return

    # 2. 计算回填日期范围
    all_dates = sorted(trade_dates)
    end_date = all_dates[-1]
    start_date = (
        datetime.strptime(all_dates[0], "%Y%m%d") - timedelta(days=args.lookback_days)
    ).strftime("%Y%m%d")

    logger.info(
        "Starting backfill: %s stocks, date range %s -> %s",
        len(missing_codes), start_date, end_date
    )

    # 3. 执行回填
    results = backfill_daily_data(
        missing_codes,
        start_date,
        end_date,
        throttle_seconds=args.throttle_seconds,
        max_retries=args.max_retries,
        logger=logger,
    )

    summary = {
        "status": "complete",
        "missing_count": len(missing_codes),
        "success_count": len(results["success"]),
        "failed_count": len(results["failed"]),
        "empty_count": len(results["empty"]),
        "total_fetched_rows": results["total_fetched_rows"],
        "success_codes": results["success"],
        "failed": results["failed"],
        "empty_codes": results["empty"],
    }

    logger.info(
        "Backfill complete: success=%s, failed=%s, empty=%s, total_rows=%s",
        len(results["success"]),
        len(results["failed"]),
        len(results["empty"]),
        results["total_fetched_rows"],
    )

    print_json(summary, output_file=args.output_file)


if __name__ == "__main__":
    main()

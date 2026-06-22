from __future__ import annotations

import argparse
import time
from typing import Any, Callable, Dict, List

from octts.config import get_settings
from octts.tools.common import print_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Tushare permission coverage for relay, funds, news, and earnings endpoints.")
    parser.add_argument("--ts-code", default="600000.SH", help="Sample stock code for stock-level endpoints")
    parser.add_argument("--event-ts-code", default="301360.SZ", help="Sample stock code for disclosure / abnormal-event probes")
    parser.add_argument("--news-src", default="sina", help="News source to test, e.g. sina")
    parser.add_argument("--news-start", default="2026-04-01 00:00:00", help="News query start time")
    parser.add_argument("--news-end", default="2026-04-13 23:59:59", help="News query end time")
    parser.add_argument("--finance-start", default="20240101", help="Financial query start date YYYYMMDD")
    parser.add_argument("--finance-end", default="20260331", help="Financial query end date YYYYMMDD")
    parser.add_argument("--trade-date", default="20260331", help="Sample trade date YYYYMMDD for market/funds endpoints")
    parser.add_argument("--start-date", default="20260301", help="Sample start date YYYYMMDD for range-based endpoints")
    parser.add_argument("--end-date", default="20260331", help="Sample end date YYYYMMDD for range-based endpoints")
    parser.add_argument("--hk-trade-date", default="20260331", help="Sample trade date YYYYMMDD for hk_hold")
    parser.add_argument("--event-start", default="20260528", help="Sample disclosure/event start date YYYYMMDD")
    parser.add_argument("--event-end", default="20260601", help="Sample disclosure/event end date YYYYMMDD")
    parser.add_argument("--suspend-seconds", type=float, default=1.2, help="Sleep between endpoint calls to avoid rate bursts")
    parser.add_argument("--skip-news", action="store_true", help="Skip news-related endpoints")
    parser.add_argument("--skip-finance", action="store_true", help="Skip finance-related endpoints")
    parser.add_argument("--skip-relay", action="store_true", help="Skip relay/funds-related endpoints")
    parser.add_argument("--skip-events", action="store_true", help="Skip disclosure / abnormal-event endpoint probes")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.tushare_token:
        raise ValueError("TINYSHARE_TOKEN or TUSHARE_TOKEN is required.")

    try:
        import tinyshare as ts
    except ImportError:
        import tushare as ts

    import tushare.pro.client as client

    base_url = settings.tushare_base_url or "https://tt.xiaodefa.cn"
    client.DataApi._DataApi__http_url = base_url
    pro = ts.pro_api(settings.tushare_token)
    try:
        pro._DataApi__http_url = base_url
    except Exception:
        pass

    checks: List[tuple[str, Callable[[], Any], Dict[str, Any]]] = []

    if not args.skip_news:
        checks.extend(
            [
                (
                    "news",
                    lambda: pro.news(src=args.news_src, start_date=args.news_start, end_date=args.news_end),
                    {"src": args.news_src, "start_date": args.news_start, "end_date": args.news_end},
                ),
                (
                    "major_news",
                    lambda: pro.major_news(src=args.news_src, start_date=args.news_start, end_date=args.news_end),
                    {"src": args.news_src, "start_date": args.news_start, "end_date": args.news_end},
                ),
            ]
        )

    if not args.skip_finance:
        checks.extend(
            [
                (
                    "forecast",
                    lambda: pro.forecast(ts_code=args.ts_code, start_date=args.finance_start, end_date=args.finance_end),
                    {"ts_code": args.ts_code, "start_date": args.finance_start, "end_date": args.finance_end},
                ),
                (
                    "express",
                    lambda: pro.express(ts_code=args.ts_code, start_date=args.finance_start, end_date=args.finance_end),
                    {"ts_code": args.ts_code, "start_date": args.finance_start, "end_date": args.finance_end},
                ),
                (
                    "fina_indicator",
                    lambda: pro.fina_indicator(ts_code=args.ts_code, start_date=args.finance_start, end_date=args.finance_end),
                    {"ts_code": args.ts_code, "start_date": args.finance_start, "end_date": args.finance_end},
                ),
            ]
        )

    if not args.skip_relay:
        checks.extend(
            [
                (
                    "moneyflow",
                    lambda: pro.moneyflow(ts_code=args.ts_code, start_date=args.start_date, end_date=args.end_date),
                    {"ts_code": args.ts_code, "start_date": args.start_date, "end_date": args.end_date},
                ),
                (
                    "top_list",
                    lambda: pro.top_list(trade_date=args.trade_date),
                    {"trade_date": args.trade_date},
                ),
                (
                    "limit_list_d",
                    lambda: pro.limit_list_d(trade_date=args.trade_date),
                    {"trade_date": args.trade_date},
                ),
                (
                    "moneyflow_mkt_dc",
                    lambda: pro.moneyflow_mkt_dc(start_date=args.start_date, end_date=args.end_date),
                    {"start_date": args.start_date, "end_date": args.end_date},
                ),
                (
                    "moneyflow_ind_ths",
                    lambda: pro.moneyflow_ind_ths(trade_date=args.trade_date),
                    {"trade_date": args.trade_date},
                ),
                (
                    "hk_hold",
                    lambda: pro.hk_hold(trade_date=args.hk_trade_date),
                    {"trade_date": args.hk_trade_date},
                ),
                (
                    "margin_detail",
                    lambda: pro.margin_detail(ts_code=args.ts_code, start_date=args.start_date, end_date=args.end_date),
                    {"ts_code": args.ts_code, "start_date": args.start_date, "end_date": args.end_date},
                ),
            ]
        )

    if not args.skip_events:
        event_params = {"ts_code": args.event_ts_code, "start_date": args.event_start, "end_date": args.event_end}
        trade_event_params = {"ts_code": args.event_ts_code, "trade_date": args.event_end}
        checks.extend(
            [
                (
                    "anns_d",
                    lambda: _call_endpoint(pro, "anns_d", **event_params),
                    event_params,
                ),
                (
                    "stk_special",
                    lambda: _call_endpoint(pro, "stk_special", **trade_event_params),
                    trade_event_params,
                ),
                (
                    "stock_special",
                    lambda: _call_endpoint(pro, "stock_special", **trade_event_params),
                    trade_event_params,
                ),
                (
                    "abnormal_change",
                    lambda: _call_endpoint(pro, "abnormal_change", **trade_event_params),
                    trade_event_params,
                ),
                (
                    "disclosure_ann",
                    lambda: _call_endpoint(pro, "disclosure_ann", **event_params),
                    event_params,
                ),
                (
                    "notice",
                    lambda: _call_endpoint(pro, "notice", **event_params),
                    event_params,
                ),
                (
                    "stk_auction_c",
                    lambda: _call_endpoint(pro, "stk_auction_c", trade_date=args.trade_date, ts_code=args.ts_code),
                    {"trade_date": args.trade_date, "ts_code": args.ts_code},
                ),
                (
                    "stk_auction_d",
                    lambda: _call_endpoint(pro, "stk_auction_d", trade_date=args.trade_date, ts_code=args.ts_code),
                    {"trade_date": args.trade_date, "ts_code": args.ts_code},
                ),
                (
                    "auction", 
                    lambda: _call_endpoint(pro, "auction", trade_date=args.trade_date, ts_code=args.ts_code),
                    {"trade_date": args.trade_date, "ts_code": args.ts_code},
                ),
            ]
        )

    results: List[Dict[str, Any]] = []
    for idx, (name, fetcher, params) in enumerate(checks):
        results.append(_run_check(name, fetcher, params))
        if idx < len(checks) - 1 and args.suspend_seconds > 0:
            time.sleep(args.suspend_seconds)

    print_json(
        {
            "checked": True,
            "base_url": base_url,
            "token_env": "TINYSHARE_TOKEN/TUSHARE_TOKEN",
            "summary": {
                "total": len(results),
                "success_count": sum(1 for item in results if item["ok"]),
                "non_empty_count": sum(1 for item in results if item.get("rows", 0) > 0),
            },
            "results": results,
        }
    )


def _call_endpoint(pro: Any, endpoint: str, **params: Any) -> Any:
    method = getattr(pro, endpoint, None)
    if method is not None:
        return method(**params)
    return pro.query(endpoint, **params)


def _run_check(name: str, fetcher: Callable[[], Any], params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        df = fetcher()
    except Exception as exc:
        return {
            "endpoint": name,
            "ok": False,
            "params": params,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    if df is None:
        return {
            "endpoint": name,
            "ok": True,
            "params": params,
            "rows": 0,
            "columns": [],
            "sample": [],
            "note": "returned_none",
        }

    columns = list(getattr(df, "columns", []))
    rows = int(len(df)) if hasattr(df, "__len__") else 0
    sample = df.head(3).to_dict(orient="records") if hasattr(df, "head") else []
    note = "empty_result" if rows == 0 else "has_data"

    return {
        "endpoint": name,
        "ok": True,
        "params": params,
        "rows": rows,
        "columns": columns,
        "sample": sample,
        "note": note,
    }


if __name__ == "__main__":
    main()

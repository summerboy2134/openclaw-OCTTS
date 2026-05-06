from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timedelta, date
from pathlib import Path
from statistics import mean
from typing import Any

from octts.config import get_settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.tools.common import configure_tool_logging, print_json
from octts.tools.compare_top3_score_fields import _load_local_stock_universe, _looks_unfillable_limit_up, _run_one_day, _safe_float

CACHE_VERSION = "fusion_top3_rotation_v1"
HORIZONS = [3, 4, 5]


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest fusion_70_30 Top3 rotation")
    p.add_argument("--trade-dates")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--score-field", default="fusion_70_30")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--candidate-limit", type=int, default=200)
    p.add_argument("--exclude-bj", action="store_true")
    p.add_argument("--exclude-st", action="store_true")
    p.add_argument("--pool", choices=["stage2_top20", "stage1_candidate"], default="stage2_top20")
    p.add_argument("--require-fillable-entry", action="store_true")
    p.add_argument("--refill-unfillable", action="store_true")
    p.add_argument("--rotation-sell-offsets", default="2,3", help="Comma-separated trading-day offsets to sell weakest position, e.g. 2,3")
    p.add_argument("--max-positions", type=int, default=3, help="Maximum concurrent holdings for refill rotation")
    p.add_argument("--refill-after-sell", action="store_true", help="After selling old holdings, buy highest-ranked current TopK candidate that is not already held and is open-fillable")
    p.add_argument("--base-cache-dir", default="tmp/compare_top3_score_fields_cache")
    p.add_argument("--selection-cache-file", default="tmp/fusion_70_30_top3_selection_cache.json")
    p.add_argument("--force-refresh-selection-cache", action="store_true")
    p.add_argument("--force-refresh-base-cache", action="store_true")
    p.add_argument("--output-file", default="tmp/fusion_70_30_top3_rotation_backtest.json")
    p.add_argument("--compact", action="store_true")
    a = p.parse_args()
    settings = get_settings(); log = configure_tool_logging(settings, "backtest_fusion_top3_rotation")
    repo = MarketRawDataRepository(settings.database_url); days = resolve_days(a, repo); sell_offsets = parse_ints(a.rotation_sell_offsets)
    cache = load_cache(a.selection_cache_file); eng = ShortTermFeatureEngineer(settings); universe = _load_local_stock_universe(settings.database_url)
    log.info("rotation backtest start: days=%s score_field=%s top_k=%s sell_offsets=%s", len(days), a.score_field, a.top_k, sell_offsets)
    results = []
    for i, d in enumerate(days, 1):
        log.info("rotation day start: trade_date=%s (%s/%s)", d.isoformat(), i, len(days))
        try:
            sel = selection(a, repo, eng, universe, cache, d, log)
            res = eval_day(repo, d, sel, bool(a.require_fillable_entry), sell_offsets, bool(a.refill_after_sell), max(1, a.max_positions))
            results.append(res)
            log.info("rotation day complete: trade_date=%s base3=%s base5=%s strategies=%s", d.isoformat(), val(res,"baseline","3"), val(res,"baseline","5"), strategy_brief(res))
        except Exception as exc:
            log.exception("rotation day failed: trade_date=%s", d.isoformat()); results.append({"trade_date": d.isoformat(), "evaluated": False, "error": str(exc)})
    save_cache(a.selection_cache_file, cache)
    payload = {"evaluated": any(x.get("evaluated") for x in results), "trade_dates": [d.isoformat() for d in days], "score_field": a.score_field, "top_k": a.top_k, "pool": a.pool, "entry_mode": "next_open", "horizons": HORIZONS, "rotation_sell_offsets": sell_offsets, "max_positions": max(1, a.max_positions), "refill_after_sell": bool(a.refill_after_sell), "rotation_rule": {"entry": "next trading day open", "sell": "sell weakest position at configured trading-day open offset", "refill": "if enabled, buy highest-ranked current TopK candidate not already held and open-fillable", "evaluate": "3rd/4th/5th trading day close", "sold_cash": "held flat after sell unless refill-after-sell is enabled"}, "selection_cache_file": a.selection_cache_file, "summary": summarize(results, sell_offsets), "daily_results": results}
    log.info("rotation backtest complete: summary=%s", payload["summary"])
    print_json(compact(payload) if a.compact else payload, output_file=a.output_file)


def resolve_days(a, repo):
    if a.trade_dates: return [datetime.strptime(x.strip(), "%Y-%m-%d").date() for x in a.trade_dates.split(",") if x.strip()]
    if not a.start_date or not a.end_date: raise ValueError("Either --trade-dates or both --start-date/--end-date are required")
    return [datetime.strptime(x, "%Y%m%d").date() for x in repo.list_trading_dates(start_date=a.start_date.replace("-", ""), end_date=a.end_date.replace("-", ""))]


def parse_ints(raw):
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    vals = sorted({x for x in vals if x >= 2})
    if not vals: raise ValueError("--rotation-sell-offsets must contain at least one integer >= 2")
    return vals


def load_cache(path):
    p = Path(path)
    if not p.exists(): return {"cache_version": CACHE_VERSION, "items": {}}
    try: data = json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {"cache_version": CACHE_VERSION, "items": {}}
    return data if data.get("cache_version") == CACHE_VERSION and isinstance(data.get("items"), dict) else {"cache_version": CACHE_VERSION, "items": {}}


def save_cache(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); data["cache_version"] = CACHE_VERSION; data["saved_at"] = datetime.now().isoformat(); p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def key(a, d):
    data = {"v": CACHE_VERSION, "d": d.isoformat(), "field": a.score_field, "top_k": a.top_k, "candidate_limit": a.candidate_limit, "exclude_bj": a.exclude_bj, "exclude_st": a.exclude_st, "pool": a.pool, "fillable": a.require_fillable_entry, "refill": a.refill_unfillable}
    return f"{d.isoformat()}|{hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]}"


def selection(a, repo, eng, universe, cache, d: date, log):
    k = key(a, d); items = cache.setdefault("items", {})
    if not a.force_refresh_selection_cache and k in items:
        out = dict(items[k]); out["selection_cache_hit"] = True; log.info("top3 selection cache hit: trade_date=%s", d.isoformat()); return out
    log.info("top3 selection build start: trade_date=%s", d.isoformat())
    day = _run_one_day(trade_day=d, engineer=eng, repo=repo, stock_universe=universe, score_fields=[a.score_field], top_k=max(1, a.top_k), candidate_limit=max(1, a.candidate_limit), exclude_bj=a.exclude_bj, exclude_st=a.exclude_st, pool_name=a.pool, entry_mode="next_open", require_fillable_entry=a.require_fillable_entry, refill_unfillable=a.refill_unfillable, cache_dir=a.base_cache_dir, force_refresh_cache=a.force_refresh_base_cache)
    field = (day.get("field_results") or {}).get(a.score_field) or {}
    out = {"cache_key": k, "trade_date": d.isoformat(), "score_field": a.score_field, "top_codes": list(field.get("top_codes") or []), "top_picks": list(field.get("top_picks") or []), "pool_name": day.get("pool_name"), "pool_size": day.get("pool_size"), "diagnostics": day.get("diagnostics") or {}, "base_cache_hit": bool(day.get("cache_hit")), "selection_cache_hit": False}
    items[k] = out; log.info("top3 selection build complete: trade_date=%s top_codes=%s", d.isoformat(), out["top_codes"]); return dict(out)


def eval_day(repo, d: date, sel: dict[str, Any], require_fillable: bool, sell_offsets: list[int], refill_after_sell: bool = False, max_positions: int = 3):
    picks = list(sel.get("top_picks") or []); codes = [str(x.get("ts_code") or "").strip().upper() for x in picks if str(x.get("ts_code") or "").strip()]
    max_idx = max([5, *sell_offsets]); tds = repo.list_trading_dates(start_date=d.strftime("%Y%m%d"), end_date=(d + timedelta(days=60)).strftime("%Y%m%d"))
    if len(tds) <= max_idx or not codes: return {"trade_date": d.isoformat(), "evaluated": False, "reason": "insufficient_dates_or_empty_picks", "picked": picks, "trading_dates": tds}
    needed = sorted({1, *HORIZONS, *sell_offsets}); bars = repo.get_daily_by_trade_dates(ts_codes=codes, trading_dates=[tds[i] for i in needed])
    candidates = build_rows(picks, bars, tds, sell_offsets, require_fillable)
    rows = candidates[:max_positions]
    if not rows: return {"trade_date": d.isoformat(), "evaluated": False, "reason": "no_fillable_positions", "picked": picks, "trading_dates": tds}
    strategies = {}
    for o in sell_offsets:
        sold = weakest(rows, str(o))
        refill = pick_refill(candidates, rows, sold, str(o)) if refill_after_sell else None
        strategies[str(o)] = {"sold_position": compact_pos(sold, str(o)) if sold else None, "refill_position": compact_refill(refill, str(o)) if refill else None, **rotated(rows, sold, str(o), refill)}
    return {"trade_date": d.isoformat(), "evaluated": True, "selection_cache_hit": sel.get("selection_cache_hit"), "base_cache_hit": sel.get("base_cache_hit"), "trading_dates": {"signal_date": tds[0], "entry_date": tds[1], "target_dates": {str(h): tds[h] for h in HORIZONS}, "rotation_sell_dates": {str(o): tds[o] for o in sell_offsets}}, "picked": rows, "candidates": candidates, "baseline": baseline(rows), "rotation_strategies": strategies}


def build_rows(picks, bars, tds, sell_offsets, require_fillable):
    rows = []
    for p in picks:
        c = str(p.get("ts_code") or "").strip().upper()
        eb = (bars.get(c) or {}).get(tds[1])
        entry = _safe_float((eb or {}).get("open"))
        fillable = None if not eb else not _looks_unfillable_limit_up(entry_bar=eb, ts_code=c, name=p.get("name"), market=p.get("market"))
        if require_fillable and fillable is False:
            continue
        rets = {str(h): ret(entry, _safe_float(((bars.get(c) or {}).get(tds[h]) or {}).get("close"))) for h in HORIZONS}
        target_closes = {str(h): _safe_float(((bars.get(c) or {}).get(tds[h]) or {}).get("close")) for h in HORIZONS}
        sells = {str(o): {"date": tds[o], "open": _safe_float(((bars.get(c) or {}).get(tds[o]) or {}).get("open"))} for o in sell_offsets}
        sell_returns = {str(o): ret(entry, sells[str(o)]["open"]) for o in sell_offsets}
        refill_returns = {str(o): {str(h): ret(sells[str(o)]["open"], target_closes[str(h)]) for h in HORIZONS} for o in sell_offsets}
        rows.append({**p, "ts_code": c, "entry_date": tds[1], "entry_open": entry, "fillable": fillable, "returns": rets, "target_close_by_horizon": target_closes, "sell_open_by_offset": sells, "sell_return_by_offset": sell_returns, "refill_return_by_offset": refill_returns})
    return rows


def ret(a, b): return None if a in (None, 0.0) or b is None else round((float(b) - float(a)) / float(a), 6)
def avg(xs): return round(mean(xs), 6) if xs else None
def metric(xs):
    vals = [x for x in xs if x is not None]; return {"position_count": len(vals), "portfolio_return": avg(vals), "positive_rate": None if not vals else round(sum(1 for x in vals if x > 0) / len(vals), 4)}
def baseline(rows): return {str(h): metric([_safe_float((r.get("returns") or {}).get(str(h))) for r in rows]) for h in HORIZONS}
def weakest(rows, offset):
    vals = [r for r in rows if (r.get("sell_return_by_offset") or {}).get(offset) is not None]; return min(vals, key=lambda r: (float((r.get("sell_return_by_offset") or {}).get(offset) or 0), int(r.get("rank") or 999))) if vals else None


def pick_refill(candidates, held_rows, sold, offset):
    if not sold:
        return None
    held = {str(r.get("ts_code") or "").strip().upper() for r in held_rows}
    sold_code = str(sold.get("ts_code") or "").strip().upper()
    held.discard(sold_code)
    for r in sorted(candidates, key=lambda x: int(x.get("rank") or 999)):
        code = str(r.get("ts_code") or "").strip().upper()
        if not code or code in held or code == sold_code:
            continue
        if r.get("fillable") is False:
            continue
        buy_open = _safe_float(((r.get("sell_open_by_offset") or {}).get(offset) or {}).get("open"))
        if buy_open is None:
            continue
        return {**r, "refill_buy_offset": int(offset), "refill_buy_date": ((r.get("sell_open_by_offset") or {}).get(offset) or {}).get("date"), "refill_buy_open": buy_open}
    return None


def refill_leg_return(refill, offset, h):
    return _safe_float((((refill or {}).get("refill_return_by_offset") or {}).get(offset) or {}).get(str(h)))


def rotated(rows, sold, offset, refill=None):
    code = str((sold or {}).get("ts_code") or ""); sr = _safe_float((sold or {}).get("sell_return_by_offset", {}).get(offset)); refill_code = str((refill or {}).get("ts_code") or ""); out = {}
    for h in HORIZONS:
        legs, rem = [], []
        for r in rows:
            r_code = str(r.get("ts_code") or "")
            if code and r_code == code:
                if refill and refill_code:
                    rr = refill_leg_return(refill, offset, h)
                    if sr is not None and rr is not None: legs.append(round((1 + sr) * (1 + rr) - 1, 6))
                    elif sr is not None: legs.append(sr)
                elif sr is not None:
                    legs.append(sr)
            else:
                v = _safe_float((r.get("returns") or {}).get(str(h)))
                if v is not None: legs.append(v); rem.append(v)
        out[str(h)] = {**metric(legs), "remaining_positions_avg_return": avg(rem), "remaining_position_count": len(rem)}
    return out

def compact_pos(p, offset):
    sell_info = (p.get("sell_open_by_offset") or {}).get(offset, {})
    return {
        "rank": p.get("rank"),
        "ts_code": p.get("ts_code"),
        "name": p.get("name"),
        "score_value": p.get("score_value"),
        "entry_date": p.get("entry_date"),
        "entry_open": p.get("entry_open"),
        "rotation_sell_offset": int(offset),
        "rotation_sell_date": sell_info.get("date"),
        "rotation_sell_open": sell_info.get("open"),
        "rotation_sell_return": (p.get("sell_return_by_offset") or {}).get(offset),
        "returns": p.get("returns"),
    }


def compact_refill(p, offset):
    if not p:
        return None
    return {
        "rank": p.get("rank"),
        "ts_code": p.get("ts_code"),
        "name": p.get("name"),
        "score_value": p.get("score_value"),
        "refill_buy_offset": int(offset),
        "refill_buy_date": p.get("refill_buy_date"),
        "refill_buy_open": p.get("refill_buy_open"),
        "refill_returns": ((p.get("refill_return_by_offset") or {}).get(offset) or {}),
    }


def val(res, part, h):
    return ((res.get(part) or {}).get(h) or {}).get("portfolio_return")


def strategy_val(res, offset, h):
    strategy = (res.get("rotation_strategies") or {}).get(str(offset)) or {}
    return (strategy.get(str(h)) or {}).get("portfolio_return")


def strategy_brief(res):
    return {
        k: {
            "sold": ((v.get("sold_position") or {}).get("ts_code")),
            "rot3": ((v.get("3") or {}).get("portfolio_return")),
            "rot5": ((v.get("5") or {}).get("portfolio_return")),
        }
        for k, v in (res.get("rotation_strategies") or {}).items()
    }

def agg(xs): return {"sample_days": len(xs), "avg_return": avg(xs), "positive_rate": None if not xs else round(sum(1 for x in xs if x > 0) / len(xs), 4)}
def summarize(days, sell_offsets):
    ok = [d for d in days if d.get("evaluated")]; out = {"evaluated_days": len(ok), "failed_days": len(days) - len(ok), "baseline": {}, "rotation_strategies": {}, "best_strategy_by_horizon": {}, "sold_positions": {}}
    for h in HORIZONS:
        b = [val(d,"baseline",str(h)) for d in ok if val(d,"baseline",str(h)) is not None]; out["baseline"][str(h)] = agg(b)
    for o in sell_offsets:
        sk = str(o); out["rotation_strategies"][sk] = {"horizons": {}, "delta": {}, "sold_positions": []}
        for h in HORIZONS:
            b = [val(d,"baseline",str(h)) for d in ok if val(d,"baseline",str(h)) is not None]; r = [strategy_val(d, sk, str(h)) for d in ok if strategy_val(d, sk, str(h)) is not None]
            out["rotation_strategies"][sk]["horizons"][str(h)] = agg(r); ds = [rv-bv for bv,rv in zip(b,r)]; out["rotation_strategies"][sk]["delta"][str(h)] = {"avg_return_delta": avg(ds), "improved_days": sum(1 for x in ds if x > 0), "paired_days": len(ds)}
        for d in ok:
            s = (((d.get("rotation_strategies") or {}).get(sk) or {}).get("sold_position") or {})
            rf = (((d.get("rotation_strategies") or {}).get(sk) or {}).get("refill_position") or {})
            if s: out["rotation_strategies"][sk]["sold_positions"].append({"trade_date": d.get("trade_date"), "ts_code": s.get("ts_code"), "name": s.get("name"), "rotation_sell_return": s.get("rotation_sell_return"), "refill_ts_code": rf.get("ts_code"), "refill_rank": rf.get("rank"), "refill_buy_open": rf.get("refill_buy_open"), "return_3d": (s.get("returns") or {}).get("3"), "return_5d": (s.get("returns") or {}).get("5")})
    for h in HORIZONS:
        best = None
        for o in sell_offsets:
            m = out["rotation_strategies"][str(o)]["horizons"][str(h)].get("avg_return")
            if m is not None and (best is None or m > best["avg_return"]): best = {"rotation_sell_offset": o, "avg_return": m}
        out["best_strategy_by_horizon"][str(h)] = best
    return out

def compact(payload): return {k: payload.get(k) for k in ["evaluated", "trade_dates", "score_field", "top_k", "pool", "entry_mode", "rotation_rule", "summary"]}

if __name__ == "__main__": main()

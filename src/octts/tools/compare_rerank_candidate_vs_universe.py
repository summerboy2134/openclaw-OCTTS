from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

import pandas as pd

from octts.config import Settings
from octts.models.screening_models import DatabaseManager, MarketStockBasic, MarketDailyBasic, MarketDaily
from octts.schemas.screener import ScreenResult
from octts.services.regression_rerank_service import RegressionRerankService
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.short_term_feature_engineering import ShortTermFeatureEngineer
from octts.services.stock_screener import StockScreener

logger = logging.getLogger(__name__)


def parse_dates(raw: str) -> list[datetime.date]:
    return [datetime.strptime(x.strip(), "%Y-%m-%d").date() for x in raw.split(",") if x.strip()]


def weighted_rank(scores_by_model: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    ranked_parts: list[pd.Series] = []
    for model_name, score_map in scores_by_model.items():
        s = pd.Series(score_map, dtype=float)
        ranked_parts.append(s.rank(method="average", pct=True) * weights[model_name])
    combo = ranked_parts[0]
    for extra in ranked_parts[1:]:
        combo = combo.add(extra, fill_value=0.0)
    return {str(code): float(score) for code, score in combo.items()}


def summarize_picks(rows: list[dict]) -> dict:
    r1 = [float(x["return_1d"]) for x in rows if x.get("return_1d") is not None]
    r3 = [float(x["return_3d"]) for x in rows if x.get("return_3d") is not None]
    r5 = [float(x["return_5d"]) for x in rows if x.get("return_5d") is not None]
    acc = [1.0 if float(x.get("return_1d") or 0.0) > 0 else 0.0 for x in rows if x.get("return_1d") is not None]
    return {
        "avg_return_1d": sum(r1) / len(r1) if r1 else None,
        "avg_return_3d": sum(r3) / len(r3) if r3 else None,
        "avg_return_5d": sum(r5) / len(r5) if r5 else None,
        "accuracy_1d": sum(acc) / len(acc) if acc else None,
    }


def build_market_snapshot_from_db(*, settings: Settings, trade_day: datetime.date, exclude_bj: bool) -> dict:
    db = DatabaseManager(settings.database_url)
    trade_date_value = trade_day
    trade_date_text = trade_day.strftime("%Y%m%d")
    session = db.get_session()
    try:
        basic_codes = {
            str(value[0]).strip().upper()
            for value in session.query(MarketDailyBasic.ts_code)
            .filter(MarketDailyBasic.trade_date == trade_date_value)
            .distinct()
            .all()
            if value and value[0]
        }
        daily_codes = {
            str(value[0]).strip().upper()
            for value in session.query(MarketDaily.ts_code)
            .filter(MarketDaily.trade_date == trade_date_value)
            .distinct()
            .all()
            if value and value[0]
        }
        ts_codes = sorted(basic_codes | daily_codes)
        if exclude_bj:
            ts_codes = [code for code in ts_codes if not code.endswith(".BJ")]

        stock_basic_rows = {
            str(row.ts_code).strip().upper(): row
            for row in session.query(MarketStockBasic).filter(MarketStockBasic.ts_code.in_(ts_codes)).all()
            if row.ts_code
        } if ts_codes else {}

        stocks = []
        for ts_code in ts_codes:
            row = stock_basic_rows.get(ts_code)
            stocks.append(
                {
                    "ts_code": ts_code,
                    "symbol": row.symbol if row else None,
                    "name": row.name if row else ts_code,
                    "area": row.area if row else None,
                    "industry": row.industry if row else None,
                    "market": row.market if row else None,
                    "list_date": row.list_date.strftime("%Y%m%d") if row and row.list_date else None,
                }
            )
    finally:
        session.close()

    repo = MarketRawDataRepository(settings.database_url)
    start_date_text = (trade_day - timedelta(days=120)).strftime("%Y%m%d")
    trading_dates = repo.list_trading_dates(start_date=start_date_text, end_date=trade_date_text)
    if not trading_dates:
        trading_dates = [trade_date_text]

    daily_basic_by_date = repo.get_daily_basic_by_trade_dates(ts_codes=ts_codes, trading_dates=[trade_date_text])
    daily_rows_by_date = repo.get_daily_by_trade_dates(ts_codes=ts_codes, trading_dates=trading_dates)

    daily_basic = {
        ts_code: date_map.get(trade_date_text)
        for ts_code, date_map in daily_basic_by_date.items()
        if date_map.get(trade_date_text)
    }
    daily = {
        ts_code: [date_map[trade_date] for trade_date in trading_dates if trade_date in date_map]
        for ts_code, date_map in daily_rows_by_date.items()
    }

    logger.info(
        "Built market snapshot from DB: trade_date=%s, stocks=%s, daily_basic=%s, daily_history=%s",
        trade_day.isoformat(),
        len(stocks),
        len(daily_basic),
        sum(1 for rows in daily.values() if rows),
    )
    return {
        "snapshot_version": "db_only",
        "trade_date": trade_date_text,
        "created_at": datetime.now().isoformat(),
        "stocks": stocks,
        "daily_basic": daily_basic,
        "daily": daily,
    }


def load_or_build_screening_results_from_db(
    *,
    settings: Settings,
    engineer: ShortTermFeatureEngineer,
    trade_day: datetime.date,
    exclude_bj: bool,
) -> dict[str, ScreenResult]:
    db = DatabaseManager(settings.database_url)
    history = db.get_screening_history(
        start_date=datetime.combine(trade_day, dt_time.min),
        end_date=datetime.combine(trade_day, dt_time.max),
        limit=1000,
    )
    results: dict[str, ScreenResult] = {}
    for item in history:
        run_id = item.get("run_id")
        strategy_id = item.get("strategy")
        if not run_id or not strategy_id:
            continue
        result = db.get_screening_result(run_id)
        if result is None:
            continue
        if exclude_bj:
            result.stocks = [stock for stock in result.stocks if not str(stock.ts_code).strip().upper().endswith(".BJ")]
            result.total_count = len(result.stocks)
        results[str(strategy_id)] = result

    if results:
        logger.info("Loaded screening results from DB: trade_date=%s, strategies=%s", trade_day.isoformat(), len(results))
        return results

    logger.info("No screening results in DB for %s, rebuilding from database market data", trade_day.isoformat())
    screener = engineer.scheduler.screener
    trade_date_text = trade_day.strftime("%Y%m%d")
    market_snapshot = build_market_snapshot_from_db(settings=settings, trade_day=trade_day, exclude_bj=exclude_bj)
    results = {}
    for preset in screener.get_presets():
        criteria = preset.criteria.model_copy(deep=True)
        criteria.max_recent_loss_years = None
        criteria.require_positive_3d_moneyflow = False
        if exclude_bj:
            criteria.exclude_bj = True
        result = screener.screen(criteria, trade_date=trade_date_text, market_snapshot=market_snapshot)
        results[preset.id] = result
        if settings.use_database:
            engineer.store._get_db_manager().save_screening_result(preset.id, result)
    logger.info("Rebuilt and saved screening results: trade_date=%s, strategies=%s", trade_day.isoformat(), len(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate-pool-only vs full-universe pure model scoring (database-first)")
    parser.add_argument("--trade-dates", required=True, help="Comma-separated YYYY-MM-DD dates")
    parser.add_argument("--candidate-limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--exclude-bj", action="store_true")
    parser.add_argument("--output", default="tmp/rerank_universe_comparison.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    settings = Settings()
    engineer = ShortTermFeatureEngineer(settings)
    rerank = RegressionRerankService(settings)
    artifact_paths = rerank._resolve_default_artifact_paths()
    if not artifact_paths:
        raise RuntimeError("Default ensemble artifacts not found")
    artifacts = rerank._load_default_artifacts(artifact_paths)
    weights = {str(spec["model_name"]): float(spec["weight"]) for spec in artifacts}

    payload: dict[str, object] = {
        "trade_dates": [d.isoformat() for d in parse_dates(args.trade_dates)],
        "model_weights": weights,
        "top_k": int(args.top_k),
        "results": [],
    }

    for trade_day in parse_dates(args.trade_dates):
        screening_results = load_or_build_screening_results_from_db(
            settings=settings,
            engineer=engineer,
            trade_day=trade_day,
            exclude_bj=bool(args.exclude_bj),
        )
        candidate_codes = engineer.scheduler._get_top_stocks(screening_results, limit=max(int(args.candidate_limit), settings.screening_top_n))
        candidate_codes = engineer.scheduler._filter_out_tracked_and_holding_codes(candidate_codes)
        if args.exclude_bj:
            candidate_codes = [x for x in candidate_codes if not str(x).strip().upper().endswith(".BJ")]

        candidate_samples = rerank.dataset_builder.build_samples_for_codes(candidate_codes, start_date=trade_day, end_date=trade_day)
        full_samples = rerank.dataset_builder.build_samples(start_date=trade_day, end_date=trade_day, exclude_bj=bool(args.exclude_bj))
        candidate_sample_codes = {s.ts_code.strip().upper() for s in candidate_samples}
        missing_candidate_codes = [code for code in candidate_codes if code not in candidate_sample_codes]
        logger.info(
            "Comparison universe stats: trade_date=%s, candidate_codes=%s, candidate_samples=%s, full_samples=%s, missing_candidate_samples=%s, missing_examples=%s",
            trade_day.isoformat(),
            len(candidate_codes),
            len(candidate_samples),
            len(full_samples),
            len(missing_candidate_codes),
            missing_candidate_codes[:10],
        )

        universe_results = []
        for mode, samples in (("candidate_pool_only", candidate_samples), ("full_universe", full_samples)):
            if not samples:
                universe_results.append({"mode": mode, "picked": [], "summary": {}})
                continue
            sample_map = {s.ts_code.strip().upper(): s.model_dump(mode="python") for s in samples}
            score_maps: dict[str, dict[str, float]] = {}
            for spec in artifacts:
                model_name = str(spec["model_name"])
                artifact = spec["artifact"]
                feature_columns = list(artifact.get("feature_columns") or [])
                model = artifact.get("model")
                rows, codes = [], []
                for ts_code, sample in sample_map.items():
                    rows.append({c: sample.get(c, 0.0) for c in feature_columns})
                    codes.append(ts_code)
                frame = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce").fillna(0.0)
                preds = model.predict(frame)
                score_maps[model_name] = {code: float(score) for code, score in zip(codes, preds)}

            combo_scores = weighted_rank(score_maps, weights)
            picked = sorted(combo_scores.items(), key=lambda x: x[1], reverse=True)[: max(int(args.top_k), 1)]
            picked_rows = []
            for ts_code, score in picked:
                sample = sample_map[ts_code]
                picked_rows.append(
                    {
                        "ts_code": ts_code,
                        "score": round(float(score), 6),
                        "return_1d": sample.get("return_1d"),
                        "return_3d": sample.get("return_3d"),
                        "return_5d": sample.get("return_5d"),
                    }
                )
            universe_results.append({"mode": mode, "pool_size": len(sample_map), "picked": picked_rows, "summary": summarize_picks(picked_rows)})

        payload["results"].append(
            {
                "trade_date": trade_day.isoformat(),
                "candidate_pool_size": len(candidate_codes),
                "candidate_sample_size": len(candidate_samples),
                "candidate_missing_sample_count": len(missing_candidate_codes),
                "candidate_missing_sample_examples": missing_candidate_codes[:10],
                "full_sample_size": len(full_samples),
                "universes": universe_results,
            }
        )

    summary: dict[str, dict[str, list[float]]] = {}
    for item in payload["results"]:
        for universe in item["universes"]:
            mode = universe["mode"]
            summary.setdefault(mode, {"avg_return_1d": [], "avg_return_3d": [], "avg_return_5d": [], "accuracy_1d": []})
            for key in summary[mode]:
                value = universe["summary"].get(key)
                if value is not None:
                    summary[mode][key].append(float(value))

    payload["summary"] = {
        mode: {k: (sum(v) / len(v) if v else None) for k, v in values.items()} for mode, values in summary.items()
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 120)
    print("纯模型打分宇宙对比（数据库优先，不含 moneyflow veto / analysis selection / soft filter）")
    print("=" * 120)
    print(f"{'模式':<24} {'平均1日收益':<12} {'平均3日收益':<12} {'平均5日收益':<12} {'1日准确率':<10}")
    print("-" * 120)
    for mode, values in payload["summary"].items():
        r1 = f"{values['avg_return_1d'] * 100:.2f}%" if values['avg_return_1d'] is not None else "N/A"
        r3 = f"{values['avg_return_3d'] * 100:.2f}%" if values['avg_return_3d'] is not None else "N/A"
        r5 = f"{values['avg_return_5d'] * 100:.2f}%" if values['avg_return_5d'] is not None else "N/A"
        acc = f"{values['accuracy_1d'] * 100:.2f}%" if values['accuracy_1d'] is not None else "N/A"
        print(f"{mode:<24} {r1:<12} {r3:<12} {r5:<12} {acc:<10}")


if __name__ == "__main__":
    main()

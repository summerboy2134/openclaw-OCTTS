from __future__ import annotations

from datetime import date, timedelta

import pytest

from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder


def _build_minimal_maps() -> tuple[list[str], dict[str, dict], dict[str, dict], dict[str, dict]]:
    start = date(2026, 1, 1)
    trading_dates = [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(21)]
    daily_map = {
        trade_date: {
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "pre_close": 10.0,
            "pct_chg": 0.0,
            "amount": 1000.0,
            "vol": 100.0,
        }
        for trade_date in trading_dates
    }
    basic_map = {
        trade_date: {
            "turnover_rate": 1.0,
            "volume_ratio": 1.0,
            "total_mv": 100000.0,
            "pe_ttm": 12.0,
            "pb": 1.2,
        }
        for trade_date in trading_dates
    }
    moneyflow_map = {
        trade_date: {
            "net_mf_amount": -1.0,
            "buy_lg_amount": 0.0,
            "sell_lg_amount": 2.0,
            "buy_elg_amount": 1.0,
            "sell_elg_amount": 0.0,
        }
        for trade_date in trading_dates[-3:]
    }
    return trading_dates, daily_map, basic_map, moneyflow_map


def test_limit_chase_risk_can_ignore_stock_moneyflow_without_dropping_moneyflow_features() -> None:
    builder = RawMarketTrainingDatasetBuilder.__new__(RawMarketTrainingDatasetBuilder)
    trading_dates, daily_map, basic_map, moneyflow_map = _build_minimal_maps()
    kwargs = {
        "ts_code": "000001.SZ",
        "sample_trade_dates": [trading_dates[-1]],
        "all_trading_dates": trading_dates,
        "daily_map": daily_map,
        "basic_map": basic_map,
        "adj_factor_map": {},
        "limit_map": {},
        "moneyflow_map": moneyflow_map,
        "market_context": {trading_dates[-1]: {"market_return_1d": 0.0, "market_up_ratio_1d": 1.0}},
        "rank_context": {},
        "min_history_days": 20,
    }

    with_moneyflow = builder._build_samples_for_code(
        **kwargs,
        include_stock_moneyflow_in_limit_chase_risk=True,
    )[0]
    without_moneyflow = builder._build_samples_for_code(
        **kwargs,
        include_stock_moneyflow_in_limit_chase_risk=False,
    )[0]

    assert with_moneyflow.moneyflow_net_3d == pytest.approx(-3.0)
    assert without_moneyflow.moneyflow_net_3d == pytest.approx(-3.0)
    assert with_moneyflow.limit_chase_failure_risk_score == 2.0
    assert without_moneyflow.limit_chase_failure_risk_score == 0.0

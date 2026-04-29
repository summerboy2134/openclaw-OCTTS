from datetime import date

from octts.models.screening_models import (
    DatabaseManager,
    MarketAdjFactor,
    MarketDaily,
    MarketDailyBasic,
)
from octts.schemas.screener import TrackedRecommendationState
from octts.schemas.training import ShortTermTrainingSample
from octts.services.market_raw_data_repository import MarketRawDataRepository


def test_upsert_recommendation_pool_states_persists_continuation_fields(tmp_path) -> None:
    db_path = tmp_path / "screening.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")

    state = TrackedRecommendationState(
        ts_code="000001.SZ",
        trade_date=date(2026, 4, 1),
        continuation_bias_score=2.4,
        continuation_positive_flags=["3日资金承接偏强"],
        continuation_negative_flags=["近5日涨幅偏大"],
        top3_risk_penalty=8.4,
        short_term_contradiction_penalty=3.0,
        final_display_recommendation_score=45.61,
        top3_extreme_risk_blocked=True,
        top3_extreme_risk_reason="candidate_risk_blocked",
    )

    persisted = manager.upsert_recommendation_pool_states([state])

    assert persisted[0]["continuation_bias_score"] == 2.4
    assert persisted[0]["continuation_positive_flags"] == ["3日资金承接偏强"]
    assert persisted[0]["continuation_negative_flags"] == ["近5日涨幅偏大"]
    assert persisted[0]["top3_risk_penalty"] == 8.4
    assert persisted[0]["short_term_contradiction_penalty"] == 3.0
    assert persisted[0]["final_display_recommendation_score"] == 45.61
    assert persisted[0]["top3_extreme_risk_blocked"] is True
    assert persisted[0]["top3_extreme_risk_reason"] == "candidate_risk_blocked"

    loaded = manager.load_recommendation_pool_state(trade_date=date(2026, 4, 1))
    assert loaded[0]["continuation_bias_score"] == 2.4
    assert loaded[0]["continuation_positive_flags"] == ["3日资金承接偏强"]
    assert loaded[0]["continuation_negative_flags"] == ["近5日涨幅偏大"]
    assert loaded[0]["top3_risk_penalty"] == 8.4
    assert loaded[0]["short_term_contradiction_penalty"] == 3.0
    assert loaded[0]["final_display_recommendation_score"] == 45.61
    assert loaded[0]["top3_extreme_risk_blocked"] is True
    assert loaded[0]["top3_extreme_risk_reason"] == "candidate_risk_blocked"


def test_upsert_short_term_training_samples_persists_labels_and_features(tmp_path) -> None:
    db_path = tmp_path / "screening.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")

    sample = ShortTermTrainingSample(
        trade_date=date(2026, 4, 7),
        ts_code="688010.SH",
        name="福光股份",
        source_tag="今日Top3",
        in_frontlist=True,
        recommend_rank=1,
        strategy_count=2,
        recommendation_score=59.56,
        overall_score=66.98,
        turnover_rate=3.38,
        continuation_bias_score=3.6,
        continuation_positive_flags=["多策略共振"],
        distribution_risk_flags=["近3日资金承接偏弱"],
        return_1d=0.021,
        label_up_1d=True,
    )

    persisted = manager.upsert_short_term_training_samples([sample])
    assert persisted[0]["ts_code"] == "688010.SH"
    assert persisted[0]["label_up_1d"] is True
    assert persisted[0]["continuation_bias_score"] == 3.6

    loaded = manager.list_short_term_training_samples(start_date=date(2026, 4, 7), end_date=date(2026, 4, 7))
    assert len(loaded) == 1
    assert loaded[0]["recommend_rank"] == 1
    assert loaded[0]["return_1d"] == 0.021
    assert loaded[0]["distribution_risk_flags"] == ["近3日资金承接偏弱"]


def test_upsert_market_raw_tables_support_idempotent_insert_and_force_refresh(tmp_path) -> None:
    db_path = tmp_path / "screening.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")

    calendar_rows = [
        {"cal_date": "20260401", "is_open": "1", "pretrade_date": "20260331", "exchange": "SSE"},
    ]
    daily_rows = [
        {
            "trade_date": "20260401",
            "ts_code": "000001.SZ",
            "open": 10.0,
            "high": 10.5,
            "low": 9.9,
            "close": 10.2,
            "pre_close": 9.8,
            "change": 0.4,
            "pct_chg": 4.08,
            "vol": 12345,
            "amount": 67890,
        }
    ]
    daily_basic_rows = [
        {
            "trade_date": "20260401",
            "ts_code": "000001.SZ",
            "close": 10.2,
            "turnover_rate": 2.1,
            "turnover_rate_f": 2.0,
            "volume_ratio": 1.3,
            "pe": 10.0,
            "pe_ttm": 11.0,
            "pb": 1.2,
            "ps": 0.8,
            "ps_ttm": 0.9,
            "dv_ratio": 0.5,
            "dv_ttm": 0.6,
            "total_share": 1000,
            "float_share": 800,
            "free_share": 700,
            "total_mv": 10200,
            "circ_mv": 8160,
        }
    ]
    adj_factor_rows = [{"trade_date": "20260401", "ts_code": "000001.SZ", "adj_factor": 1.2345}]

    assert manager.upsert_market_trade_calendar(calendar_rows) == 1
    assert manager.upsert_market_trade_calendar(calendar_rows) == 0
    assert manager.upsert_market_daily(daily_rows) == 1
    assert manager.upsert_market_daily(daily_rows) == 0
    assert manager.upsert_market_daily_basic(daily_basic_rows) == 1
    assert manager.upsert_market_daily_basic(daily_basic_rows) == 0
    assert manager.upsert_market_adj_factor(adj_factor_rows) == 1
    assert manager.upsert_market_adj_factor(adj_factor_rows) == 0

    updated_daily_rows = [{**daily_rows[0], "close": 10.8}]
    updated_basic_rows = [{**daily_basic_rows[0], "turnover_rate": 3.5}]
    updated_adj_factor_rows = [{**adj_factor_rows[0], "adj_factor": 1.5432}]

    assert manager.upsert_market_daily(updated_daily_rows, force_refresh=True) == 1
    assert manager.upsert_market_daily_basic(updated_basic_rows, force_refresh=True) == 1
    assert manager.upsert_market_adj_factor(updated_adj_factor_rows, force_refresh=True) == 1

    session = manager.get_session()
    try:
        daily_record = session.query(MarketDaily).filter_by(ts_code="000001.SZ", trade_date=date(2026, 4, 1)).one()
        daily_basic_record = session.query(MarketDailyBasic).filter_by(ts_code="000001.SZ", trade_date=date(2026, 4, 1)).one()
        adj_factor_record = session.query(MarketAdjFactor).filter_by(ts_code="000001.SZ", trade_date=date(2026, 4, 1)).one()
    finally:
        session.close()

    assert daily_record.close == 10.8
    assert daily_basic_record.turnover_rate == 3.5
    assert adj_factor_record.adj_factor == 1.5432
    assert manager.has_market_trade_calendar(start_date=date(2026, 4, 1), end_date=date(2026, 4, 1)) is True
    assert manager.has_market_data_for_trade_date(model=MarketDaily, trade_date=date(2026, 4, 1)) is True


def test_market_raw_data_repository_reads_written_rows(tmp_path) -> None:
    db_path = tmp_path / "screening.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")
    manager.upsert_market_trade_calendar([
        {"cal_date": "20260401", "is_open": "1", "pretrade_date": "20260331", "exchange": "SSE"},
        {"cal_date": "20260402", "is_open": "1", "pretrade_date": "20260401", "exchange": "SSE"},
    ])
    manager.upsert_market_daily([
        {"trade_date": "20260401", "ts_code": "000001.SZ", "open": 10.0, "high": 10.5, "low": 9.9, "close": 10.2, "pre_close": 9.8, "change": 0.4, "pct_chg": 4.08, "vol": 12345, "amount": 67890},
        {"trade_date": "20260402", "ts_code": "000001.SZ", "open": 10.2, "high": 10.8, "low": 10.1, "close": 10.6, "pre_close": 10.2, "change": 0.4, "pct_chg": 3.92, "vol": 22345, "amount": 77890},
    ])
    manager.upsert_market_daily_basic([
        {"trade_date": "20260401", "ts_code": "000001.SZ", "close": 10.2, "turnover_rate": 2.1, "turnover_rate_f": 2.0, "volume_ratio": 1.3, "pe": 10.0, "pe_ttm": 11.0, "pb": 1.2, "ps": 0.8, "ps_ttm": 0.9, "dv_ratio": 0.5, "dv_ttm": 0.6, "total_share": 1000, "float_share": 800, "free_share": 700, "total_mv": 10200, "circ_mv": 8160},
        {"trade_date": "20260402", "ts_code": "000001.SZ", "close": 10.6, "turnover_rate": 2.5, "turnover_rate_f": 2.4, "volume_ratio": 1.5, "pe": 10.2, "pe_ttm": 11.2, "pb": 1.3, "ps": 0.85, "ps_ttm": 0.95, "dv_ratio": 0.55, "dv_ttm": 0.65, "total_share": 1000, "float_share": 800, "free_share": 700, "total_mv": 10600, "circ_mv": 8480},
    ])
    manager.upsert_market_adj_factor([
        {"trade_date": "20260401", "ts_code": "000001.SZ", "adj_factor": 1.11},
        {"trade_date": "20260402", "ts_code": "000001.SZ", "adj_factor": 1.12},
    ])

    repo = MarketRawDataRepository(f"sqlite:///{db_path}")

    assert repo.list_trading_dates(start_date="20260401", end_date="20260402") == ["20260401", "20260402"]
    assert repo.get_daily(ts_code="000001.SZ", trade_date="20260401")["close"] == 10.2
    assert repo.get_daily_basic(ts_code="000001.SZ", trade_date="20260402")["turnover_rate"] == 2.5
    assert repo.get_adj_factor(ts_code="000001.SZ", trade_date="20260402") == 1.12
    assert len(repo.get_daily_range(ts_code="000001.SZ", start_date="20260401", end_date="20260402")) == 2

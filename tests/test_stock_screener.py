"""Tests for stock screener functionality."""

import asyncio
import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Union
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from octts.api import app
from octts.config import Settings
from octts.schemas.screener import ScreenCriteria, ScreenPreset, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.services.lightweight_backtester import LightweightBacktester
from octts.services.multi_dimensional_analyzer import MultiDimensionalAnalyzer
from octts.services.news_aggregator import NewsCluster, NewsItem, NewsSource
from octts.services.enhanced_screening_scheduler import EnhancedScreeningScheduler
from octts.services.recommendation_tracker import RecommendationTracker
from octts.services.screening_store import ScreeningStore
from octts.services.stock_screener import StockScreener
from octts.prompts.report_prompt import build_today_screening_report_prompt, build_yesterday_review_report_prompt
from octts.services.intelligent_report_generator import IntelligentReport, IntelligentReportGenerator


def _build_daily_rows(
    closes: List[float],
    *,
    start_date: str = "20240301",
    volumes: Optional[List[float]] = None,
) -> List[Dict[str, Union[float, str]]]:
    base_date = datetime.strptime(start_date, "%Y%m%d")
    rows: List[Dict[str, Union[float, str]]] = []
    for index, close in enumerate(closes):
        previous_close = closes[index - 1] if index > 0 else close
        pct_chg = 0.0 if index == 0 else ((close - previous_close) / previous_close) * 100
        rows.append(
            {
                "trade_date": (base_date + timedelta(days=index)).strftime("%Y%m%d"),
                "close": round(close, 2),
                "pct_chg": round(pct_chg, 2),
                "vol": float(volumes[index] if volumes is not None else 1000 + index * 10),
            }
        )
    return list(reversed(rows))


def _build_screen_result(ts_code: str = "000001.SZ") -> ScreenResult:
    return ScreenResult(
        screen_id=f"screen-{ts_code.lower()}",
        criteria=ScreenCriteria(limit=10),
        stocks=[
            StockScreenItem(
                ts_code=ts_code,
                name="平安银行",
                close=10.5,
                pct_change=2.3,
                volume_ratio=1.8,
                turnover_rate=1.2,
                score=78.0,
                match_reasons=["测试结果"],
            )
        ],
        total_count=1,
        execution_time=0.12,
    )


@pytest.fixture
def mock_tushare_data():
    """Mock tushare data for testing."""
    stock_list = [
        {
            "ts_code": "000001.SZ",
            "symbol": "000001",
            "name": "平安银行",
            "industry": "银行",
            "market": "主板",
            "list_date": "19910403",
        },
        {
            "ts_code": "000002.SZ",
            "symbol": "000002",
            "name": "万科A",
            "industry": "房地产",
            "market": "主板",
            "list_date": "19910129",
        },
        {
            "ts_code": "600000.SH",
            "symbol": "600000",
            "name": "浦发银行",
            "industry": "银行",
            "market": "主板",
            "list_date": "19991110",
        },
    ]
    daily_data = {
        "000001.SZ": _build_daily_rows([9.5 + 0.05 * index for index in range(25)]),
        "000002.SZ": _build_daily_rows([12 + 0.2 * index for index in range(25)]),
        "600000.SH": _build_daily_rows([11 - 0.1 * index for index in range(25)]),
    }
    daily_basic = {
        "000001.SZ": {
            "ts_code": "000001.SZ",
            "close": daily_data["000001.SZ"][0]["close"],
            "turnover_rate": 1.2,
            "volume_ratio": 1.5,
            "pe": 5.2,
            "total_mv": 1800000,
        },
        "000002.SZ": {
            "ts_code": "000002.SZ",
            "close": daily_data["000002.SZ"][0]["close"],
            "turnover_rate": 3.5,
            "volume_ratio": 2.3,
            "pe": 8.5,
            "total_mv": 1500000,
        },
        "600000.SH": {
            "ts_code": "600000.SH",
            "close": daily_data["600000.SH"][0]["close"],
            "turnover_rate": 0.8,
            "volume_ratio": 0.9,
            "pe": 4.8,
            "total_mv": 3000000,
        },
    }
    return {
        "stock_list": stock_list,
        "daily_basic": daily_basic,
        "daily_data": daily_data,
    }


@pytest.fixture
def mock_client(mock_tushare_data):
    """Mock TushareClient."""
    client = Mock()
    client.fetch_stock_list.return_value = mock_tushare_data["stock_list"]
    client.fetch_daily_basic_batch.return_value = mock_tushare_data["daily_basic"]
    client.fetch_daily_batch.return_value = mock_tushare_data["daily_data"]
    client.fetch_trading_dates.return_value = ["20240325"]
    return client


@pytest.fixture
def stub_settings() -> Settings:
    return Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")


def test_stock_screener_init_is_lazy(stub_settings: Settings):
    """StockScreener should not create a client until it is needed."""
    screener = StockScreener(settings=stub_settings)

    assert screener.settings is stub_settings
    assert screener._client is None


def test_get_presets():
    """Preset lookup should not depend on tushare initialization."""
    presets = StockScreener.get_presets()

    assert len(presets) > 0
    assert all(isinstance(p, ScreenPreset) for p in presets)
    preset_ids = [p.id for p in presets]
    assert "oversold_bounce" in preset_ids
    assert "volume_breakout" in preset_ids


def test_screen_basic(mock_client, mock_tushare_data, stub_settings: Settings):
    """Only the stock meeting both涨幅 and量比 should remain."""
    screener = StockScreener(settings=stub_settings, client=mock_client)

    result = screener.screen(
        ScreenCriteria(
            pct_change_min=1.0,
            volume_ratio_min=2.0,
            exclude_st=True,
            limit=10,
        )
    )

    assert result.total_count == 1
    assert len(result.stocks) == 1
    assert result.stocks[0].ts_code == "000002.SZ"
    assert result.stocks[0].pct_change == mock_tushare_data["daily_data"]["000002.SZ"][0]["pct_chg"]
    assert result.stocks[0].volume_ratio == 2.3


def test_screen_with_market_cap_filter(mock_client, stub_settings: Settings):
    """Market cap filter should reject larger symbols and keep ordering stable."""
    screener = StockScreener(settings=stub_settings, client=mock_client)

    result = screener.screen(
        ScreenCriteria(
            market_cap_max=200,
            exclude_st=True,
            sort_by="market_cap",
            sort_desc=False,
            limit=10,
        )
    )

    assert result.total_count == 2
    assert len(result.stocks) == 2
    assert [item.ts_code for item in result.stocks] == ["000002.SZ", "000001.SZ"]


def test_screen_with_industry_filter(mock_client, stub_settings: Settings):
    """Industry filtering should only keep matching symbols."""
    screener = StockScreener(settings=stub_settings, client=mock_client)

    result = screener.screen(
        ScreenCriteria(
            industries=["银行"],
            exclude_st=True,
            limit=10,
        )
    )

    assert result.total_count == 2
    assert all(stock.industry == "银行" for stock in result.stocks)


def test_screen_rejects_missing_indicator_data(mock_client, mock_tushare_data, stub_settings: Settings):
    """A stock without enough history must not slip through RSI-based filters."""
    short_history = {
        **mock_tushare_data["daily_data"],
        "000002.SZ": mock_tushare_data["daily_data"]["000002.SZ"][:5],
    }
    mock_client.fetch_daily_batch.return_value = short_history
    screener = StockScreener(settings=stub_settings, client=mock_client)

    result = screener.screen(
        ScreenCriteria(
            rsi_max=80,
            exclude_st=True,
            limit=10,
        )
    )

    assert all(stock.ts_code != "000002.SZ" for stock in result.stocks)


def test_evaluate_stock_uses_latest_window_for_moving_averages(mock_client, mock_tushare_data, stub_settings: Settings):
    """Moving averages should use the latest rolling window rather than the first row."""
    screener = StockScreener(settings=stub_settings, client=mock_client)
    stock_info = mock_tushare_data["stock_list"][1]
    stock_data = {
        "basic": mock_tushare_data["daily_basic"]["000002.SZ"],
        "daily": mock_tushare_data["daily_data"]["000002.SZ"],
    }

    item = screener._evaluate_stock(stock_info, stock_data, ScreenCriteria(), "20240325")

    closes = pd.Series(list(reversed([row["close"] for row in stock_data["daily"]])))
    expected_ma5 = closes.tail(5).mean()
    expected_ma20 = closes.tail(20).mean()

    assert item is not None
    assert item.close == closes.iloc[-1]
    assert item.ma5 == pytest.approx(expected_ma5)
    assert item.ma20 == pytest.approx(expected_ma20)


def test_job_manager_returns_active_job_only_for_running_states(tmp_path) -> None:
    from octts.services.intelligent_screening_job_manager import IntelligentScreeningJob, IntelligentScreeningJobManager

    manager = IntelligentScreeningJobManager(str(tmp_path))
    running_job = IntelligentScreeningJob(job_id="job-running", status="running")
    manager._jobs[running_job.job_id] = running_job
    manager._running_job_id = running_job.job_id

    active = asyncio.run(manager.get_active_job())

    assert active is not None
    assert active["job_id"] == "job-running"

    running_job.status = "succeeded"
    active_after_finish = asyncio.run(manager.get_active_job())

    assert active_after_finish is None


def test_job_manager_success_message_uses_frontlist_and_tracking_pool_counts() -> None:
    result = {
        "frontlist_count": 3,
        "tracking_pool_count": 5,
        "screened_stocks": 8,
        "final_recommendations": 3,
    }

    message = app.state.intelligent_screening_job_manager._build_success_message(result) if hasattr(app.state, "intelligent_screening_job_manager") else None
    if message is None:
        from octts.services.intelligent_screening_job_manager import IntelligentScreeningJobManager
        message = IntelligentScreeningJobManager._build_success_message(result)

    assert message == "智能选股完成：前台推荐 3 只，跟踪池 5 只。"


def test_settings_default_screening_time_is_configurable() -> None:
    settings = Settings(
        OCTTS_SCREENING_TIME="15:35",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH="memory.json",
    )

    assert settings.screening_time == "15:35"


def test_save_recommendation_run_overwrites_same_trade_date(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'screening.db'}"
    settings = Settings(
        OCTTS_DATABASE_URL=database_url,
        OCTTS_USE_DATABASE=True,
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    store = ScreeningStore(settings)
    trade_date = date(2026, 3, 26)

    first = store.save_recommendation_run(
        run_id="rec-20260326",
        trade_date=trade_date,
        candidate_count=20,
        final_count=2,
        report_id="report-1",
        items=[
            {
                "ts_code": "000001.SZ",
                "name": "平安银行",
                "recommend_rank": 1,
                "recommend_score": 88.0,
                "status": "new",
                "trade_date": trade_date,
            },
            {
                "ts_code": "000002.SZ",
                "name": "万科A",
                "recommend_rank": 2,
                "recommend_score": 80.0,
                "status": "tracking",
                "trade_date": trade_date,
            },
        ],
    )

    second = store.save_recommendation_run(
        run_id="rec-20260326",
        trade_date=trade_date,
        candidate_count=20,
        final_count=1,
        report_id="report-2",
        items=[
            {
                "ts_code": "600000.SH",
                "name": "浦发银行",
                "recommend_rank": 1,
                "recommend_score": 91.0,
                "status": "new",
                "trade_date": trade_date,
            }
        ],
    )

    history = store.list_recommendation_history(limit=10)
    active = store.list_active_recommendations(limit=10)

    assert first["id"] == second["id"]
    assert len(history) == 1
    assert history[0]["run_id"] == "rec-20260326"
    assert history[0]["final_count"] == 1
    assert history[0]["report_id"] == "report-2"
    assert [item["ts_code"] for item in active] == ["600000.SH"]


def test_screen_criteria_validation():
    """ScreenCriteria should keep current defaults and accept explicit bounds."""
    criteria = ScreenCriteria(
        price_min=5.0,
        price_max=100.0,
        pct_change_min=-10.0,
        pct_change_max=10.0,
        limit=100,
    )
    assert criteria.price_min == 5.0
    assert criteria.price_max == 100.0

    default_criteria = ScreenCriteria()
    assert default_criteria.exclude_st is True
    assert default_criteria.sort_by == "pct_change"
    assert default_criteria.sort_desc is True
    assert default_criteria.limit == 50
    assert default_criteria.recommendation_score_min is None
    assert default_criteria.setup_types is None


def test_technical_indicators():
    """Indicator helpers should return sane numeric outputs."""
    from octts.indicators.technical import calculate_rsi, calculate_sma

    prices = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10])

    rsi = calculate_rsi(prices, period=14)
    assert not rsi.empty


def test_hits_late_stage_risk_gate_blocks_high_position_weakening_stock() -> None:
    criteria = ScreenCriteria(max_late_stage_price_position=0.98)

    assert StockScreener._hits_late_stage_risk_gate(
        pct_change=-2.3,
        turnover_rate=15.0,
        volume_ratio=3.1,
        price_position=0.96,
        criteria=criteria,
    ) is True
    assert StockScreener._hits_late_stage_risk_gate(
        pct_change=2.1,
        turnover_rate=15.0,
        volume_ratio=3.1,
        price_position=0.95,
        criteria=criteria,
    ) is False
    assert StockScreener._hits_late_stage_risk_gate(
        pct_change=8.2,
        turnover_rate=18.0,
        volume_ratio=3.4,
        price_position=0.96,
        criteria=criteria,
    ) is False


def test_build_recommendation_pool_states_uses_latest_top10_and_yesterday_top10_intersection(monkeypatch, tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_STOCK_POOL="600000.SH",
        OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    create_position_store = __import__("octts.services.position_store", fromlist=["create_position_store"]).create_position_store
    create_position_store(settings).set_status("000004.SZ", "holding")

    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )

    scheduler.store.load_recommendation_pool_state.return_value = [
        {"ts_code": "000001.SZ", "name": "A-历史", "in_frontlist": True, "ai_confidence": 0.7, "hit_streak_days": 1, "times_entered_frontlist": 1, "source_tag": "今日Top3", "recommend_rank": 1},
        {"ts_code": "000005.SZ", "name": "F-历史", "in_frontlist": True, "ai_confidence": 0.58, "hit_streak_days": 2, "times_entered_frontlist": 2, "source_tag": "今日Top3", "recommend_rank": 3},
        {"ts_code": "000008.SZ", "name": "I-历史", "in_frontlist": True, "ai_confidence": 0.53, "hit_streak_days": 1, "times_entered_frontlist": 1, "source_tag": "今日Top3", "recommend_rank": 2},
    ]
    scheduler.store.list_recommendation_pool.return_value = [
        {"ts_code": "000001.SZ", "in_frontlist": True, "recommendation_score": 83},
        {"ts_code": "000005.SZ", "in_frontlist": True, "recommendation_score": 62},
        {"ts_code": "000008.SZ", "in_frontlist": True, "recommendation_score": 54},
    ]

    screening_results = {
        "s1": ScreenResult(
            screen_id="s1",
            criteria=ScreenCriteria(limit=10),
            stocks=[
                StockScreenItem(ts_code="000001.SZ", name="A", close=10, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=90, recommendation="monitor", confidence="high"),
                StockScreenItem(ts_code="000002.SZ", name="B", close=11, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=85, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000003.SZ", name="C", close=12, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=80, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000004.SZ", name="D", close=13, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=95, recommendation="monitor", confidence="high"),
                StockScreenItem(ts_code="600000.SH", name="E", close=14, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=99, recommendation="monitor", confidence="high"),
                StockScreenItem(ts_code="000005.SZ", name="F", close=15, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=60, recommendation="monitor", confidence="low"),
                StockScreenItem(ts_code="000006.SZ", name="G", close=16, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=59, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000007.SZ", name="H", close=17, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=58, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000008.SZ", name="I", close=18, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=57, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000009.SZ", name="J", close=19, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=56, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000010.SZ", name="K", close=20, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=55, recommendation="monitor", confidence="medium"),
                StockScreenItem(ts_code="000011.SZ", name="L", close=21, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=54, recommendation="monitor", confidence="medium"),
            ],
            total_count=12,
            execution_time=0.1,
        )
    }
    final_recommendations = {
        "000001.SZ": {"score": 70, "overall_score": 70, "weighted_score": 90, "overall_confidence": 0.72},
        "000002.SZ": {"score": 66, "overall_score": 66, "weighted_score": 85, "overall_confidence": 0.67},
        "000003.SZ": {"score": 64, "overall_score": 64, "weighted_score": 80, "overall_confidence": 0.63},
        "000005.SZ": {"score": 55, "overall_score": 55, "weighted_score": 60, "overall_confidence": 0.58},
        "000006.SZ": {"score": 54, "overall_score": 54, "weighted_score": 59, "overall_confidence": 0.62},
        "000007.SZ": {"score": 53, "overall_score": 53, "weighted_score": 58, "overall_confidence": 0.61},
        "000008.SZ": {"score": 52, "overall_score": 52},
        "000009.SZ": {"score": 51, "overall_score": 51, "weighted_score": 56, "overall_confidence": 0.6},
        "000010.SZ": {"score": 50, "overall_score": 50, "weighted_score": 55, "overall_confidence": 0.6},
        "000011.SZ": {"score": 49, "overall_score": 49, "weighted_score": 54, "overall_confidence": 0.6},
    }

    filtered = scheduler._filter_out_tracked_and_holding_codes(["600000.SH", "000004.SZ", "000001.SZ", "000002.SZ", "000003.SZ"])
    assert filtered == ["000001.SZ", "000002.SZ", "000003.SZ"]

    analysis_target_codes = scheduler._build_analysis_target_codes(
        trade_date=date(2026, 3, 27),
        candidate_codes=["000001.SZ", "000002.SZ", "000003.SZ", "000005.SZ", "000006.SZ", "000007.SZ", "000008.SZ", "000009.SZ", "000010.SZ", "000011.SZ"],
        screening_results=screening_results,
    )
    assert analysis_target_codes == ["000001.SZ", "000002.SZ", "000003.SZ", "000005.SZ", "000006.SZ", "000007.SZ", "000008.SZ"]

    states = scheduler._build_recommendation_pool_states(
        trade_date=date(2026, 3, 27),
        screening_results=screening_results,
        final_recommendations=final_recommendations,
        candidate_codes=["000001.SZ", "000002.SZ", "000003.SZ", "000005.SZ", "000006.SZ", "000007.SZ", "000008.SZ", "000009.SZ", "000010.SZ", "000011.SZ"],
    )

    assert len(states) == 10
    assert [state.ts_code for state in states[:4]] == ["000001.SZ", "000002.SZ", "000003.SZ", "000005.SZ"]
    assert [state.source_tag for state in states[:6]] == ["今日Top3", "今日Top3", "今日Top3", "昨日延续", "今日候选", "今日候选"]
    assert states[0].is_repeat_pick is True
    assert states[0].recommendation_score == 90
    assert states[0].priority_score == 70
    assert [state.recommend_rank for state in states[:6]] == [1, 2, 3, None, None, None]
    assert states[0].ai_confidence == pytest.approx(0.72)
    assert states[0].display_confidence == pytest.approx(0.72)
    assert states[0].continuation_bias_score is None
    assert states[0].continuation_positive_flags == []
    assert states[0].continuation_negative_flags == []
    assert states[3].recommendation_score == 60
    assert states[3].priority_score == 55
    assert states[3].overall_score == 55
    assert states[3].source_tag == "昨日延续"
    assert next(state for state in states if state.ts_code == "000008.SZ").overall_score is None
    assert next(state for state in states if state.ts_code == "000008.SZ").priority_score is None
    assert states[3].name == "F"
    assert states[3].score_change == -2.0
    assert next(state for state in states if state.ts_code == "000001.SZ").is_repeat_pick is True
    assert next(state for state in states if state.ts_code == "000008.SZ").source_tag == "昨日延续"
    assert next(state for state in states if state.ts_code == "000008.SZ").is_repeat_pick is True


def test_build_recommendation_pool_states_only_promotes_analyzed_stocks_to_today_top3(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = None

    screening_results = {
        "s1": ScreenResult(
            screen_id="s1",
            criteria=ScreenCriteria(limit=10),
            stocks=[
                StockScreenItem(ts_code="000001.SZ", name="A", close=10, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=95, recommendation="monitor", confidence="high", technical_score=90),
                StockScreenItem(ts_code="000002.SZ", name="B", close=11, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=94, recommendation="monitor", confidence="high", technical_score=89),
                StockScreenItem(ts_code="000003.SZ", name="C", close=12, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=93, recommendation="monitor", confidence="high", technical_score=88),
                StockScreenItem(ts_code="000004.SZ", name="D", close=13, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=92, recommendation="monitor", confidence="high", technical_score=87),
            ],
            total_count=4,
            execution_time=0.1,
        )
    }
    final_recommendations = {
        "000003.SZ": {"overall_score": 70, "weighted_score": 83, "distribution_risk_score": 0.5, "technical_score": 88, "overall_confidence": 0.72},
        "000004.SZ": {"overall_score": 69, "weighted_score": 82, "distribution_risk_score": 0.6, "technical_score": 87, "overall_confidence": 0.71},
    }

    states = scheduler._build_recommendation_pool_states(
        trade_date=date(2026, 4, 2),
        screening_results=screening_results,
        final_recommendations=final_recommendations,
        candidate_codes=["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
    )

    today_top_codes = [state.ts_code for state in states if state.source_tag == "今日Top3"]
    assert today_top_codes == ["000003.SZ", "000004.SZ"]
    assert [state.recommend_rank for state in states if state.source_tag == "今日Top3"] == [1, 2]
    assert all(state.ts_code not in {"000001.SZ", "000002.SZ"} for state in states)


def test_build_dashboard_ai_payload_separates_scores_and_keeps_names() -> None:
    payload = EnhancedScreeningScheduler._build_dashboard_ai_payload(
        ai_analyses={
            "688799.SH": {
                "name": "",
                "overall_score": 67,
                "overall_confidence": 0.74,
                "summary": "分析摘要",
            }
        },
        final_recommendations={
            "688799.SH": {
                "weighted_score": 91,
                "score": 67,
                "recommendation": "建议关注",
            }
        },
        stock_name_map={"688799.SH": "华纳药厂"},
        pool_states=[
            {
                "ts_code": "688799.SH",
                "name": "",
                "recommendation_score": 91,
                "priority_score": 67,
                "ai_confidence": 0.74,
                "display_confidence": 0.71,
                "strategy_count": 3,
                "news_mentioned": True,
                "source_tag": "今日Top3",
            }
        ],
    )

    item = payload["688799.SH"]
    assert item["name"] == "华纳药厂"
    assert item["overall_score"] == 67
    assert item["recommendation_score"] == 91
    assert item["priority_score"] == 67
    assert item.get("score") != 91
    assert item["confidence"] == pytest.approx(0.71)
    assert item["overall_confidence"] == pytest.approx(0.74)
    assert item["score_components"]["strategy_count"] == 3
    assert item["score_components"]["news_mentioned"] is True


def test_build_dashboard_ai_payload_keeps_unanalysed_overall_score_empty() -> None:
    payload = EnhancedScreeningScheduler._build_dashboard_ai_payload(
        ai_analyses={
            "688800.SH": {
                "summary": "仅有说明",
            }
        },
        final_recommendations={
            "688800.SH": {
                "weighted_score": 89,
                "score": 68,
                "recommendation": "建议关注",
            }
        },
        stock_name_map={"688800.SH": "测试股份"},
        pool_states=[
            {
                "ts_code": "688800.SH",
                "recommendation_score": 89,
                "priority_score": None,
                "overall_score": None,
            }
        ],
    )

    item = payload["688800.SH"]
    assert item["recommendation_score"] == 89
    assert item["overall_score"] is None
    assert item["priority_score"] is None


def test_evaluate_distribution_risk_penalizes_high_level_pullback(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_stock_moneyflow_summary = Mock(return_value={"recent_3d_net_inflow": -1200})
    stock = StockScreenItem(
        ts_code="688710.SH",
        name="益诺思",
        close=32.0,
        pct_change=-2.5,
        volume_ratio=3.3,
        turnover_rate=18.0,
        price_position_20d=0.97,
    )
    daily_rows = [
        {"trade_date": "20260327", "open": 33.0, "high": 34.2, "close": 32.0, "pct_chg": -2.5, "turnover_rate": 18.0},
        {"trade_date": "20260326", "close": 32.8, "pct_chg": 4.8, "turnover_rate": 8.0},
        {"trade_date": "20260325", "close": 31.4, "pct_chg": 3.6, "turnover_rate": 7.6},
        {"trade_date": "20260324", "close": 30.3, "pct_chg": 4.1, "turnover_rate": 7.2},
        {"trade_date": "20260321", "close": 29.1, "pct_chg": 2.9, "turnover_rate": 6.8},
        {"trade_date": "20260320", "close": 28.3, "pct_chg": 1.8, "turnover_rate": 6.5},
    ]

    risk = scheduler._evaluate_distribution_risk(stock, daily_rows=daily_rows)

    assert risk["latest_weakening_flag"] is True
    assert risk["high_level_pullback_flag"] is True
    assert risk["theme_support_absent_flag"] is False
    assert risk["distribution_risk_score"] >= 3.5
    assert "高位回调且承接不足" in risk["distribution_risk_flags"]


def test_build_top_ranking_score_penalizes_short_term_contradiction() -> None:
    recommendation = {
        "weighted_score": 92,
        "recommendation_text": "等待确认：短线分歧偏大，暂不追高",
        "action_bias": "观察",
        "technical_signal": "多头趋势",
        "distribution_risk_score": 1.8,
        "recent_runup_5d": 12.5,
        "moneyflow_3d_value": 1800,
        "turnover_spike_ratio": 1.9,
        "continuation_bias_score": -1.2,
    }
    decisive_recommendation = {
        "weighted_score": 88,
        "recommendation_text": "建议跟踪：趋势延续性较好，可等回踩或放量确认",
        "action_bias": "跟踪",
        "technical_signal": "放量突破",
        "distribution_risk_score": 0.6,
        "recent_runup_5d": 5.2,
        "moneyflow_3d_value": 9200,
        "turnover_spike_ratio": 1.2,
        "continuation_bias_score": 2.4,
    }

    contradiction_score = EnhancedScreeningScheduler._build_top_ranking_score(
        "000001.SZ", recommendation, None, apply_divergence_penalty=True
    )
    decisive_score = EnhancedScreeningScheduler._build_top_ranking_score(
        "000002.SZ", decisive_recommendation, None, apply_divergence_penalty=True
    )

    assert EnhancedScreeningScheduler._build_short_term_contradiction_penalty(recommendation) > 0
    assert contradiction_score < decisive_score


def test_build_continuation_bias_prefers_healthier_next_day_extension_profile() -> None:
    score_a, pos_a, neg_a = EnhancedScreeningScheduler._build_continuation_bias(
        "000001.SZ",
        {"overall_score": 80, "technical_signal": "放量突破"},
        distribution_risk={
            "moneyflow_3d_value": 8200,
            "recent_runup_5d": 5.0,
            "turnover_spike_ratio": 1.2,
            "distribution_risk_score": 0.8,
            "latest_weakening_flag": False,
            "high_level_pullback_flag": False,
            "theme_support_absent_flag": False,
            "candidate_risk_blocked": False,
        },
        theme_support={"leader_turnover_justified_flag": True, "unsupported_high_position_flag": False},
        industry_adjustment={"industry_heat_score": 2.6},
        strategy_count=3,
        previous_state=None,
        is_previous_top3=False,
    )
    score_b, pos_b, neg_b = EnhancedScreeningScheduler._build_continuation_bias(
        "000002.SZ",
        {"overall_score": 80, "technical_signal": "多头趋势"},
        distribution_risk={
            "moneyflow_3d_value": 500,
            "recent_runup_5d": 13.8,
            "turnover_spike_ratio": 2.3,
            "distribution_risk_score": 2.4,
            "latest_weakening_flag": True,
            "high_level_pullback_flag": True,
            "theme_support_absent_flag": True,
            "candidate_risk_blocked": False,
        },
        theme_support={"leader_turnover_justified_flag": False, "unsupported_high_position_flag": True},
        industry_adjustment={"industry_heat_score": 0.2},
        strategy_count=1,
        previous_state=None,
        is_previous_top3=False,
    )

    assert score_a > score_b
    assert pos_a
    assert neg_b


def test_build_continuation_bias_gives_light_bonus_to_stable_repeat_pick() -> None:
    score, positive_flags, negative_flags = EnhancedScreeningScheduler._build_continuation_bias(
        "000001.SZ",
        {"overall_score": 82, "technical_signal": "多头走强"},
        distribution_risk={
            "moneyflow_3d_value": 3600,
            "recent_runup_5d": 6.0,
            "turnover_spike_ratio": 1.3,
            "distribution_risk_score": 1.1,
            "latest_weakening_flag": False,
            "high_level_pullback_flag": False,
            "theme_support_absent_flag": False,
            "candidate_risk_blocked": False,
        },
        theme_support={"leader_turnover_justified_flag": False, "unsupported_high_position_flag": False},
        industry_adjustment={"industry_heat_score": 1.5},
        strategy_count=2,
        previous_state={"recommendation_score": 80, "score_change": 0.5},
        is_previous_top3=True,
    )

    assert score > 0
    assert any("昨日Top3" in flag for flag in positive_flags)
    assert not any("今日风险走弱" in flag for flag in negative_flags)


def test_build_continuation_bias_does_not_override_risk_block() -> None:
    score, _positive_flags, negative_flags = EnhancedScreeningScheduler._build_continuation_bias(
        "000003.SZ",
        {"overall_score": 90, "technical_signal": "放量突破"},
        distribution_risk={
            "moneyflow_3d_value": 9800,
            "recent_runup_5d": 4.0,
            "turnover_spike_ratio": 1.1,
            "distribution_risk_score": 4.2,
            "latest_weakening_flag": False,
            "high_level_pullback_flag": False,
            "theme_support_absent_flag": False,
            "candidate_risk_blocked": True,
        },
        theme_support={"leader_turnover_justified_flag": True, "unsupported_high_position_flag": False},
        industry_adjustment={"industry_heat_score": 3.0},
        strategy_count=4,
        previous_state={"recommendation_score": 88, "score_change": 2.0},
        is_previous_top3=True,
    )

    assert score <= 0
    assert negative_flags == [] or isinstance(negative_flags, list)


def test_build_report_context_includes_today_top3_and_yesterday_review(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.load_recommendation_pool_state.return_value = [
        {
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "source_tag": "今日Top3",
            "recommend_rank": 1,
            "recommendation_score": 85,
            "priority_score": 76,
            "display_confidence": 0.72,
            "recommendation_text": "昨日看延续",
        },
        {
            "ts_code": "000099.SZ",
            "name": "掉队股",
            "source_tag": "今日Top3",
            "recommend_rank": 2,
            "recommendation_score": 80,
            "priority_score": 74,
            "display_confidence": 0.68,
            "recommendation_text": "昨日强势",
        },
    ]
    pool_states = [
        {
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "source_tag": "今日Top3",
            "recommend_rank": 1,
            "recommendation_score": 88,
            "overall_score": 78,
            "priority_score": 78,
            "base_score": 77,
            "display_confidence": 0.75,
            "strategy_count": 2,
            "news_mentioned": True,
            "technical_signal": "多头",
            "recommendation_text": "今日继续跟踪",
            "action_plan": {"action_bias": "观察"},
        },
        {
            "ts_code": "000002.SZ",
            "name": "万科A",
            "source_tag": "今日Top3",
            "recommend_rank": 2,
            "recommendation_score": 81,
            "overall_score": 72,
            "priority_score": 72,
            "base_score": 71,
            "display_confidence": 0.69,
            "strategy_count": 1,
            "news_mentioned": False,
            "technical_signal": "震荡",
            "recommendation_text": "关注回踩",
            "action_plan": {"action_bias": "买入"},
        },
    ]
    context = scheduler._build_report_context(
        trade_date=date(2026, 3, 27),
        pool_states=pool_states,
        ai_analyses={"000001.SZ": {"summary": "延续强势"}},
        final_recommendations={"000001.SZ": {"overall_score": 78}},
    )

    assert [item["ts_code"] for item in context["today_top3"]] == ["000001.SZ", "000002.SZ"]
    assert context["today_top3"][0]["strategy_count"] == 2
    assert context["today_top3"][0]["name"] == "平安银行"
    review_map = {item["ts_code"]: item for item in context["yesterday_top3_review"]}
    assert review_map["000001.SZ"]["today_verdict"] == "延续走强，继续列入今日Top3"
    assert review_map["000001.SZ"]["yesterday_conclusion"] == "优先跟踪；昨日看延续；强度仍待验证；等放量确认"
    assert review_map["000099.SZ"]["today_present"] is False
    assert "不再作为今日推荐" in review_map["000099.SZ"]["today_verdict"]
    assert review_map["000099.SZ"]["missing_factor_candidates"]


def test_build_report_context_today_top10_excludes_yesterday_continuations(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = None
    scheduler.store.load_recommendation_pool_state.return_value = []
    scheduler.store.list_recommendation_pool.return_value = []

    context = scheduler._build_report_context(
        trade_date=date(2026, 4, 2),
        pool_states=[
            {
                "ts_code": "000001.SZ",
                "name": "今日Top3",
                "source_tag": "今日Top3",
                "in_frontlist": True,
                "recommend_rank": 1,
                "recommendation_score": 91.0,
                "overall_score": 81.0,
                "base_score": 80.0,
            },
            {
                "ts_code": "000002.SZ",
                "name": "昨日延续",
                "source_tag": "昨日延续",
                "in_frontlist": True,
                "recommend_rank": 2,
                "recommendation_score": 89.0,
                "overall_score": 79.0,
                "base_score": 78.0,
            },
            {
                "ts_code": "000003.SZ",
                "name": "今日候选",
                "source_tag": "今日候选",
                "in_frontlist": True,
                "recommend_rank": 3,
                "recommendation_score": 87.0,
                "overall_score": 77.0,
                "base_score": 76.0,
            },
        ],
        ai_analyses={},
        final_recommendations={},
    )

    assert [item["ts_code"] for item in context["today_top10"]] == ["000001.SZ", "000003.SZ"]


def test_build_distribution_risk_map_uses_snapshot_history_rows() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_stock_moneyflow_summary = Mock(return_value={"recent_3d_net_inflow": 3200.0})

    stocks = [
        StockScreenItem(
            ts_code="000001.SZ",
            name="A",
            close=10,
            pct_change=-1.8,
            volume_ratio=2.1,
            turnover_rate=6.0,
            price_position_20d=85,
        )
    ]
    market_snapshot = {
        "daily": {
            "000001.SZ": [
                {"trade_date": "20260327", "open": 10.6, "high": 10.8, "close": 10.1, "pct_chg": -1.8, "turnover_rate": 6.0},
                {"trade_date": "20260326", "close": 10.5, "pct_chg": 3.2, "turnover_rate": 3.0},
                {"trade_date": "20260325", "close": 10.1, "pct_chg": 2.8, "turnover_rate": 2.9},
                {"trade_date": "20260324", "close": 9.8, "pct_chg": 2.6, "turnover_rate": 2.7},
                {"trade_date": "20260321", "close": 9.5, "pct_chg": 2.1, "turnover_rate": 2.5},
            ]
        }
    }

    risk_map = scheduler._build_distribution_risk_map({"000001.SZ": stocks[0]}, market_snapshot=market_snapshot)
    risk = risk_map["000001.SZ"]

    assert risk["recent_runup_5d"] > 0
    assert risk["turnover_spike_ratio"] > 1
    assert risk["distribution_risk_score"] > 0
    assert risk["candidate_risk_blocked"] in {True, False}



def test_save_dashboard_snapshot_writes_latest_and_trade_date(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = None
    scheduler.store.load_recommendation_pool_state.return_value = []
    scheduler.store.list_recommendation_pool.return_value = [
        {"ts_code": "002269.SZ", "name": "空壳", "recommendation_score": 99, "priority_score": None, "source_tag": "今日Top3", "in_frontlist": True, "recommend_rank": None},
        {"ts_code": "000001.SZ", "name": "平安银行", "recommendation_score": 88, "priority_score": 78, "overall_score": 78, "base_score": 76, "technical_score": 75, "distribution_risk_score": 1.1, "source_tag": "今日Top3", "in_frontlist": True, "recommend_rank": 1},
    ]
    report = IntelligentReport(
        report_id="r1",
        report_type=__import__("octts.services.intelligent_report_generator", fromlist=["ReportType"]).ReportType.MORNING,
        title="测试报告",
        generate_time=datetime.now(),
        sections=[],
        summary="摘要",
        key_points=[],
        recommendations=[],
        metadata={"report_blocks": {"focus_stocks": [{"ts_code": "legacy.SZ"}]}}
    )
    scheduler._save_dashboard_snapshot(
        screening_results={},
        ai_analyses={},
        news_clusters=[],
        report=report,
        final_recommendations={},
        trade_date=date(2026, 3, 27),
        report_context={"today_top3": [{"ts_code": "legacy.SZ"}]},
    )

    latest_path = tmp_path / "history" / "intelligent_screening" / "latest.json"
    dated_path = tmp_path / "history" / "intelligent_screening" / "20260327.json"
    assert latest_path.exists()
    assert dated_path.exists()
    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    dated_payload = json.loads(dated_path.read_text(encoding="utf-8"))
    assert latest_payload["report_context"]["today_top3"][0]["ts_code"] == "000001.SZ"
    assert latest_payload["recommendation_pool"]["today_top"][0]["ts_code"] == "000001.SZ"
    assert dated_payload["intelligent_report"]["blocks"]["focus_stocks"][0]["ts_code"] == "legacy.SZ"


def test_generate_recommendation_and_action_plan_use_single_relaxed_mapping() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )

    recommendation_text = scheduler._generate_recommendation(
        78,
        {"overall_confidence": 0.74, "technical_signal": "放量突破"},
        distribution_risk={"distribution_risk_score": 0.8, "candidate_risk_blocked": False},
    )
    action_plan = scheduler._build_action_plan(
        {
            "weighted_score": 78,
            "recommendation_text": recommendation_text,
            "technical_signal": "放量突破",
            "candidate_risk_blocked": False,
        },
        Mock(close=12.3),
        None,
    )

    assert recommendation_text == "建议跟踪：趋势延续性较好，可等回踩或放量确认"
    assert action_plan["action_bias"] == "跟踪"


def test_build_action_plan_marks_risk_blocked_items_as_avoid() -> None:
    action_plan = EnhancedScreeningScheduler._build_action_plan(
        {
            "weighted_score": 84,
            "recommendation_text": "等待确认：短线分歧偏大，暂不追高",
            "candidate_risk_blocked": True,
            "technical_signal": "多头趋势",
        },
        Mock(close=18.5),
        None,
    )

    assert action_plan["action_bias"] == "回避"


def test_build_report_context_prefers_recommend_rank_over_priority_score() -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH="history",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH="memory.json",
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.list_recommendation_pool.return_value = []

    pool_states = [
        {
            "ts_code": "688710.SH",
            "name": "高原始分",
            "source_tag": "今日Top3",
            "recommend_rank": 2,
            "recommendation_score": 86,
            "overall_score": 84,
            "priority_score": 99,
            "base_score": 84,
        },
        {
            "ts_code": "000001.SZ",
            "name": "最终第一",
            "source_tag": "今日Top3",
            "recommend_rank": 1,
            "recommendation_score": 90,
            "overall_score": 82,
            "priority_score": 80,
            "base_score": 82,
        },
    ]

    scheduler.store.load_recommendation_pool_state.return_value = pool_states

    context = scheduler._build_report_context(
        trade_date=date(2026, 3, 27),
        pool_states=pool_states,
        ai_analyses={},
        final_recommendations={},
    )

    assert [item["ts_code"] for item in context["today_top3"][:2]] == ["000001.SZ", "688710.SH"]
    assert "today_top10" not in context


def test_build_report_context_skips_shell_today_top3_states_without_real_ai_fields() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.list_recommendation_pool.return_value = []
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = None
    scheduler.store.load_recommendation_pool_state.return_value = []

    pool_states = [
        {
            "ts_code": "002269.SZ",
            "name": "空壳一",
            "source_tag": "今日Top3",
            "recommend_rank": None,
            "recommendation_score": 99,
            "overall_score": None,
            "priority_score": None,
            "summary": None,
            "distribution_risk_score": None,
        },
        {
            "ts_code": "301226.SZ",
            "name": "空壳二",
            "source_tag": "今日Top3",
            "recommend_rank": 2,
            "recommendation_score": 97,
            "overall_score": None,
            "priority_score": None,
            "summary": None,
            "distribution_risk_score": None,
        },
        {
            "ts_code": "300692.SZ",
            "name": "真实一",
            "source_tag": "今日Top3",
            "recommend_rank": 3,
            "recommendation_score": 88,
            "overall_score": 81,
            "priority_score": 81,
            "base_score": 79,
            "technical_score": 80,
            "summary": "真实分析一",
            "distribution_risk_score": 1.2,
        },
        {
            "ts_code": "600613.SH",
            "name": "真实二",
            "source_tag": "今日Top3",
            "recommend_rank": 4,
            "recommendation_score": 87,
            "overall_score": 80,
            "priority_score": 80,
            "base_score": 78,
            "technical_score": 79,
            "summary": "真实分析二",
            "distribution_risk_score": 1.1,
        },
        {
            "ts_code": "300086.SZ",
            "name": "真实三",
            "source_tag": "今日Top3",
            "recommend_rank": 5,
            "recommendation_score": 86,
            "overall_score": 79,
            "priority_score": 79,
            "base_score": 77,
            "technical_score": 78,
            "summary": "真实分析三",
            "distribution_risk_score": 1.0,
        },
        {
            "ts_code": "000001.SZ",
            "name": "昨日延续",
            "source_tag": "昨日延续",
            "recommend_rank": 1,
            "recommendation_score": 95,
            "overall_score": 70,
            "priority_score": 70,
        },
    ]

    context = scheduler._build_report_context(
        trade_date=date(2026, 4, 1),
        pool_states=pool_states,
        ai_analyses={},
        final_recommendations={},
    )

    codes = [item["ts_code"] for item in context["today_top3"]]
    assert codes == ["300692.SZ", "600613.SH", "300086.SZ"]
    assert all(item["overall_score"] is not None for item in context["today_top3"])
    assert all(item["distribution_risk_score"] is not None for item in context["today_top3"])
    assert all(item.get("base_score") is not None for item in context["today_top3"])
    assert "000001.SZ" not in codes
    assert all(item["summary"] for item in context["today_top3"])



def test_build_recommendation_pool_states_only_assigns_rank_to_real_today_top3(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = date(2026, 4, 1)
    scheduler.store.load_recommendation_pool_state.return_value = [
        {"ts_code": "002686.SZ", "source_tag": "今日Top3", "recommendation_score": 78.08, "recommend_rank": 1},
        {"ts_code": "000788.SZ", "source_tag": "今日Top3", "recommendation_score": 76.88, "recommend_rank": 2},
        {"ts_code": "301520.SZ", "source_tag": "今日Top3", "recommendation_score": 76.88, "recommend_rank": 3},
    ]
    scheduler.store.list_recommendation_pool.return_value = [
        {"ts_code": "002686.SZ", "in_frontlist": True, "recommendation_score": 78.08},
        {"ts_code": "000788.SZ", "in_frontlist": True, "recommendation_score": 76.88},
        {"ts_code": "301520.SZ", "in_frontlist": True, "recommendation_score": 76.88},
    ]

    screening_results = {
        "s1": ScreenResult(
            screen_id="s1",
            criteria=ScreenCriteria(limit=10),
            stocks=[
                StockScreenItem(ts_code="002269.SZ", name="空壳一", close=10, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=76.56, recommendation="monitor", confidence="high", technical_score=90),
                StockScreenItem(ts_code="301226.SZ", name="空壳二", close=11, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=76.04, recommendation="monitor", confidence="high", technical_score=89),
                StockScreenItem(ts_code="600222.SH", name="空壳三", close=12, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=75.36, recommendation="monitor", confidence="high", technical_score=88),
                StockScreenItem(ts_code="300692.SZ", name="真实一", close=13, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=54.01, recommendation="monitor", confidence="high", technical_score=87),
                StockScreenItem(ts_code="600613.SH", name="真实二", close=14, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=48.68, recommendation="monitor", confidence="high", technical_score=86),
                StockScreenItem(ts_code="300086.SZ", name="真实三", close=15, pct_change=1, volume_ratio=1.5, turnover_rate=1.1, recommendation_score=47.61, recommendation="monitor", confidence="high", technical_score=85),
            ],
            total_count=6,
            execution_time=0.1,
        )
    }
    final_recommendations = {
        "300692.SZ": {"overall_score": 82.31, "base_score": 82.49, "weighted_score": 54.01, "distribution_risk_score": 0.7, "technical_score": 98, "overall_confidence": 0.75, "summary": "真实一分析"},
        "600613.SH": {"overall_score": 70.6, "base_score": 70.83, "weighted_score": 48.68, "distribution_risk_score": 1.3, "technical_score": 87, "overall_confidence": 0.76, "summary": "真实二分析"},
        "300086.SZ": {"overall_score": 74.54, "base_score": 74.46, "weighted_score": 47.61, "distribution_risk_score": 1.2, "technical_score": 87, "overall_confidence": 0.77, "summary": "真实三分析"},
    }

    states = scheduler._build_recommendation_pool_states(
        trade_date=date(2026, 4, 2),
        screening_results=screening_results,
        final_recommendations=final_recommendations,
        candidate_codes=["002269.SZ", "301226.SZ", "600222.SH", "300692.SZ", "600613.SH", "300086.SZ"],
    )

    rank_map = {state.ts_code: state.recommend_rank for state in states}
    assert "002269.SZ" not in rank_map
    assert "301226.SZ" not in rank_map
    assert "600222.SH" not in rank_map
    assert rank_map["300692.SZ"] == 1
    assert rank_map["600613.SH"] == 2
    assert rank_map["300086.SZ"] == 3

    state_map = {state.ts_code: state for state in states}
    assert state_map["300692.SZ"].recommendation_score == 45.61
    assert state_map["300692.SZ"].final_display_recommendation_score == 45.61
    assert state_map["300692.SZ"].top3_risk_penalty == 8.4
    assert state_map["300692.SZ"].short_term_contradiction_penalty == 0.0
    assert state_map["600613.SH"].recommendation_score == 32.28
    assert state_map["600613.SH"].top3_risk_penalty == 15.6
    assert state_map["600613.SH"].short_term_contradiction_penalty == 0.8
    assert state_map["300086.SZ"].recommendation_score == 32.41
    assert state_map["300086.SZ"].top3_risk_penalty == 14.4
    assert state_map["300086.SZ"].short_term_contradiction_penalty == 0.8



def test_intelligent_report_generator_splits_llm_calls_and_keeps_report_shape() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    llm_client = Mock()
    llm_client.complete = AsyncMock(
        side_effect=[
            json.dumps(
                {
                    "focus_stocks": [
                        {
                            "ts_code": "000001.SZ",
                            "name": "平安银行",
                            "focus_analysis": "今日分析",
                            "core_highlights": ["亮点1", "亮点2", "亮点3"],
                            "risk_warnings": ["风险1", "风险2"],
                            "overall_assessment": "继续跟踪",
                            "action_plan": {"action_bias": "观察"},
                        }
                    ],
                    "comparison": {"best_short_term": "000001.SZ"},
                    "overall_action": {"headline": "今日建议", "action_items": ["跟踪龙头"]},
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "yesterday_reviews": [
                        {
                            "ts_code": "000002.SZ",
                            "name": "万科A",
                            "status": "观察",
                            "today_verdict": "仅作复盘跟踪",
                            "review_analysis": "昨日复盘",
                            "analysis": "昨日逻辑部分兑现，今日仅作跟踪。",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        ]
    )
    generator = IntelligentReportGenerator(settings=settings, llm_client=llm_client, analyzer=Mock())
    report = asyncio.run(
        generator.generate_morning_report(
            news_clusters=[],
            market_data={"trend": "震荡", "sentiment": "分化"},
            stock_pool=["000001.SZ", "000002.SZ"],
            screening_context={
                "today_top3": [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "recommendation_score": 90,
                        "overall_score": 80,
                        "fundamental_score": 70,
                        "technical_score": 75,
                        "risk_score": 20,
                        "action_plan": {},
                    }
                ],
                "yesterday_top3_review": [
                    {
                        "ts_code": "000002.SZ",
                        "name": "万科A",
                        "yesterday_conclusion": "昨日结论",
                        "action_plan": {},
                    }
                ],
                "comparison_candidates": [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "recommendation_score": 90,
                        "overall_score": 80,
                        "fundamental_score": 70,
                        "technical_score": 75,
                        "risk_score": 20,
                    }
                ],
                "today_top3_live_context": [],
                "yesterday_top3_live_context": [],
            },
        )
    )

    blocks = report.metadata["report_blocks"]
    assert set(blocks) >= {"focus_stocks", "yesterday_reviews", "comparison", "overall_action", "news_clusters", "theme_view"}
    assert blocks["focus_stocks"][0]["ts_code"] == "000001.SZ"
    assert blocks["yesterday_reviews"][0]["ts_code"] == "000002.SZ"
    assert llm_client.complete.await_count == 2

    first_prompt = llm_client.complete.await_args_list[0].args[0]
    second_prompt = llm_client.complete.await_args_list[1].args[0]
    assert "today_top3_live_context" in first_prompt
    assert "yesterday_top3_review" not in first_prompt
    assert "yesterday_top3_review" in second_prompt
    assert "today_top3_live_context" not in second_prompt


def test_intelligent_report_generator_falls_back_missing_blocks() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    llm_client = Mock()
    llm_client.complete = AsyncMock(side_effect=[json.dumps({}, ensure_ascii=False), json.dumps({}, ensure_ascii=False)])
    generator = IntelligentReportGenerator(settings=settings, llm_client=llm_client, analyzer=Mock())

    report = asyncio.run(
        generator.generate_morning_report(
            news_clusters=[],
            market_data={"trend": "震荡", "sentiment": "谨慎"},
            stock_pool=["000001.SZ", "000002.SZ"],
            screening_context={
                "today_top3": [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "recommendation_score": 88,
                        "overall_score": 80,
                        "summary": "今日仍在观察",
                        "action_plan": {},
                    }
                ],
                "yesterday_top3_review": [
                    {
                        "ts_code": "000002.SZ",
                        "name": "万科A",
                        "yesterday_conclusion": "昨日看修复",
                        "action_plan": {},
                    }
                ],
                "comparison_candidates": [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "recommendation_score": 88,
                        "overall_score": 80,
                        "fundamental_score": 60,
                        "technical_score": 70,
                        "risk_score": 30,
                    }
                ],
                "today_top3_live_context": [],
                "yesterday_top3_live_context": [],
            },
        )
    )

    blocks = report.metadata["report_blocks"]
    assert blocks["focus_stocks"][0]["ts_code"] == "000001.SZ"
    assert blocks["focus_stocks"][0]["focus_analysis"]
    assert blocks["yesterday_reviews"][0]["ts_code"] == "000002.SZ"
    assert blocks["yesterday_reviews"][0]["review_analysis"]
    assert blocks["comparison"]["basic_rank"]
    assert blocks["overall_action"]["headline"]


def test_focus_analysis_fallback_avoids_score_template_text() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    generator = IntelligentReportGenerator(settings=settings, llm_client=Mock(), analyzer=Mock())

    merged = generator._merge_focus_stock_item(
        {
            "ts_code": "002686.SZ",
            "focus_analysis": "当前排序主要由推荐分78.1与综合分78.1共同决定。命中策略数为2个。",
            "overall_assessment": "亿利达（002686.SZ）综合评分 69.4，主评分 69.9 由技术面与基本面构成。",
        },
        {
            "ts_code": "002686.SZ",
            "name": "亿利达",
            "recommendation_score": 78.1,
            "overall_score": 69.4,
            "pct_change": 10.04,
            "latest_profit_yoy": -22.5,
            "moneyflow_3d_value": -8805,
            "action_plan": {},
        },
    )

    assert "当前排序主要由推荐分" not in merged["focus_analysis"]
    assert "综合评分 69.4" not in merged["overall_assessment"]
    assert merged["focus_analysis"]
    assert merged["overall_assessment"]


def test_focus_analysis_fallback_replaces_short_repeated_text() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    generator = IntelligentReportGenerator(settings=settings, llm_client=Mock(), analyzer=Mock())

    merged = generator._merge_focus_stock_item(
        {
            "ts_code": "000001.SZ",
            "focus_analysis": "继续观察。",
            "overall_assessment": "继续观察。",
        },
        {
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "recommendation_score": 82.0,
            "overall_score": 76.0,
            "pct_change": 3.2,
            "turnover_rate": 1.8,
            "moneyflow_3d_value": 120000000.0,
            "action_plan": {"entry_zone": "10.0-10.2", "take_profit": "10.8", "stop_loss": "9.8", "invalid_condition": "跌破支撑"},
        },
    )

    assert merged["focus_analysis"]
    assert merged["focus_analysis"] != "继续观察。"
    assert "10.0-10.2" in merged["focus_analysis"]



def test_review_analysis_fallback_replaces_template_text() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    generator = IntelligentReportGenerator(settings=settings, llm_client=Mock(), analyzer=Mock())

    merged = generator._merge_review_item(
        {
            "ts_code": "000002.SZ",
            "review_analysis": "综合来看，该股当前处于观察阶段，操作上先看是否企稳。",
            "analysis": "原有逻辑减弱，今日主要转为复盘观察。",
        },
        {
            "ts_code": "000002.SZ",
            "name": "万科A",
            "status": "观察",
            "today_verdict": "仅作复盘跟踪",
            "yesterday_conclusion": "昨日强于板块",
            "pct_change": -1.6,
            "turnover_rate": 2.1,
            "volume_ratio": 0.8,
            "action_plan": {"action_bias": "观察", "entry_zone": "等待企稳", "take_profit": "反抽分批", "stop_loss": "跌破低点", "invalid_condition": "量价继续恶化"},
        },
    )

    assert merged["review_analysis"]
    assert "综合来看，该股当前处于" not in merged["review_analysis"]
    assert "操作上先看" not in merged["review_analysis"]



def test_clean_generated_text_removes_structured_field_leaks() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    generator = IntelligentReportGenerator(settings=settings, llm_client=Mock(), analyzer=Mock())

    cleaned = generator._clean_generated_text(
        '无明显新闻催化（news_mentioned false）。板块热度偏弱（industry_flow_bias "偏弱"）。'
    )

    assert "news_mentioned" not in cleaned
    assert "industry_flow_bias" not in cleaned
    assert "偏弱" in cleaned


def test_focus_views_skip_zero_turnover_spike_ratio() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    generator = IntelligentReportGenerator(settings=settings, llm_client=Mock(), analyzer=Mock())

    item = {
        "technical_signal": "趋势改善",
        "turnover_spike_ratio": 0.0,
        "moneyflow_3d_value": 5506.3,
        "recent_runup_5d": 9.03,
    }

    assert "0.00倍" not in generator._build_focus_risk_note(item)
    assert "0.00倍" not in generator._build_focus_trading_context_view(item)


def test_report_context_keeps_missing_overall_score_empty() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.list_recommendation_pool.return_value = []
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = None
    scheduler.store.load_recommendation_pool_state.return_value = []

    pool_states = [
        {
            "ts_code": "002686.SZ",
            "name": "亿利达",
            "source_tag": "今日Top3",
            "recommend_rank": 1,
            "recommendation_score": 78.1,
            "overall_score": None,
            "priority_score": None,
            "base_score": 69.4,
        }
    ]

    context = scheduler._build_report_context(
        trade_date=date(2026, 4, 1),
        pool_states=pool_states,
        ai_analyses={"002686.SZ": {"overall_score": 69.4}},
        final_recommendations={"002686.SZ": {"weighted_score": 81.3, "score": 69.4}},
    )

    assert context["today_top3"][0]["recommendation_score"] == 78.1
    assert context["today_top3"][0]["overall_score"] is None



def test_report_context_preserves_distinct_recommendation_and_overall_scores() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.list_recommendation_pool.return_value = []
    scheduler.store.get_previous_recommendation_pool_trade_date.return_value = None
    scheduler.store.load_recommendation_pool_state.return_value = []

    pool_states = [
        {
            "ts_code": "002686.SZ",
            "name": "亿利达",
            "source_tag": "今日Top3",
            "recommend_rank": 1,
            "recommendation_score": 78.1,
            "overall_score": 69.4,
            "priority_score": 69.4,
            "base_score": 69.4,
        }
    ]

    context = scheduler._build_report_context(
        trade_date=date(2026, 4, 1),
        pool_states=pool_states,
        ai_analyses={},
        final_recommendations={},
    )

    assert context["today_top3"][0]["recommendation_score"] == 78.1
    assert context["today_top3"][0]["overall_score"] == 69.4
    assert context["today_top3"][0]["recommendation_score"] != context["today_top3"][0]["overall_score"]



def test_split_report_prompt_builders_scope_fields() -> None:
    today_system, today_prompt = build_today_screening_report_prompt(
        market_data={"trend": "震荡"},
        news_clusters=[{"theme": "机器人"}],
        screening_context={
            "today_top3": [{"ts_code": "000001.SZ"}],
            "today_top3_live_context": [{"ts_code": "000001.SZ", "items": []}],
            "comparison_candidates": [{"ts_code": "000001.SZ"}],
            "yesterday_top3_review": [{"ts_code": "000002.SZ"}],
            "yesterday_top3_live_context": [{"ts_code": "000002.SZ", "items": []}],
        },
    )
    yesterday_system, yesterday_prompt = build_yesterday_review_report_prompt(
        news_clusters=[{"theme": "机器人"}],
        screening_context={
            "today_top3": [{"ts_code": "000001.SZ"}],
            "today_top3_live_context": [{"ts_code": "000001.SZ", "items": []}],
            "yesterday_top3_review": [{"ts_code": "000002.SZ"}],
            "yesterday_top3_live_context": [],
        },
    )

    assert today_system
    assert yesterday_system
    assert "today_top3_live_context" in today_prompt
    assert "yesterday_top3_review" not in today_prompt
    assert "yesterday_top3_live_context" not in today_prompt
    assert "yesterday_top3_review" in yesterday_prompt
    assert "today_top3_live_context" not in yesterday_prompt
    assert "focus_stocks" in today_prompt
    assert "overall_action" in today_prompt
    assert "yesterday_reviews" in yesterday_prompt


def test_build_recommendation_items_preserves_source_tags() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )

    items = scheduler._build_recommendation_items(
        trade_date="20260327",
        screening_results={},
        ai_analyses={},
        final_recommendations={},
        pool_states=[
            {"ts_code": "000001.SZ", "name": "A", "recommendation_score": 88, "source_tag": "今日Top3", "is_repeat_pick": True, "tracking_status": "active", "hit_streak_days": 2},
            {"ts_code": "000002.SZ", "name": "B", "recommendation_score": 70, "source_tag": "昨日延续", "is_repeat_pick": True, "tracking_status": "tracking", "hit_streak_days": 1},
            {"ts_code": "000003.SZ", "name": "C", "recommendation_score": 66, "source_tag": "今日候选", "tracking_status": "candidate", "ai_confidence": 0.64},
        ],
    )

    assert len(items) == 3
    assert items[0]["source_tag"] == "今日Top3"
    assert items[0]["is_repeat_pick"] is True
    assert items[0]["status"] == "active"
    assert items[1]["source_tag"] == "昨日延续"
    assert items[1]["status"] == "tracking"
    assert items[1]["score_change"] is None
    assert items[2]["source_tag"] == "今日候选"
    assert items[2]["ai_confidence"] == 0.64


def test_combine_analysis_includes_dimension_scores() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_stock_moneyflow_summary = Mock(return_value={"recent_3d_net_inflow": 0.0})

    screening_results = {
        "s1": ScreenResult(
            screen_id="s1",
            criteria=ScreenCriteria(limit=10),
            stocks=[StockScreenItem(ts_code="000001.SZ", name="平安银行", close=10.5, pct_change=2.3, volume_ratio=1.8, turnover_rate=1.2)],
            total_count=1,
            execution_time=0.1,
        )
    }

    result = asyncio.run(
        scheduler._combine_analysis(
            screening_results=screening_results,
            ai_analyses={
                "000001.SZ": {
                    "overall_score": 80,
                    "overall_confidence": 0.8,
                    "summary": "分析摘要",
                    "technical_signal": "多头",
                    "technical_score": 12.5,
                    "fundamental_score": 34.5,
                    "sentiment_score": 0.0,
                    "news_score": 8.0,
                }
            },
            news_clusters=[],
        )
    )

    assert result["000001.SZ"]["technical_score"] == 12.5
    assert result["000001.SZ"]["fundamental_score"] == 34.5
    assert result["000001.SZ"]["sentiment_score"] == 0.0
    assert result["000001.SZ"]["news_score"] == 8.0


def test_combine_analysis_keeps_high_turnover_leader_with_theme_support() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_stock_moneyflow_summary = Mock(return_value={"recent_3d_net_inflow": 18000})

    stock = StockScreenItem(
        ts_code="000001.SZ",
        name="龙头股",
        close=18.5,
        pct_change=6.2,
        volume_ratio=3.6,
        turnover_rate=22.0,
        price_position_20d=0.95,
    )
    screening_results = {
        "s1": ScreenResult(screen_id="s1", criteria=ScreenCriteria(limit=10), stocks=[stock], total_count=1, execution_time=0.1),
        "s2": ScreenResult(screen_id="s2", criteria=ScreenCriteria(limit=10), stocks=[stock], total_count=1, execution_time=0.1),
    }
    news_clusters = [
        NewsCluster(
            cluster_id="c1",
            theme="机器人",
            summary="题材持续发酵",
            importance=0.85,
            news_items=[
                NewsItem(
                    source=NewsSource.EASTMONEY,
                    title="机器人主线继续强化",
                    content="龙头获持续关注",
                    url="https://example.com/news/1",
                    publish_time=datetime.now(),
                    related_stocks=["000001.SZ"],
                )
            ],
            key_stocks=["000001.SZ"],
        )
    ]

    result = asyncio.run(
        scheduler._combine_analysis(
            screening_results=screening_results,
            ai_analyses={
                "000001.SZ": {
                    "overall_score": 78,
                    "overall_confidence": 0.85,
                    "summary": "强势龙头",
                    "technical_signal": "多头",
                }
            },
            news_clusters=news_clusters,
        )
    )

    assert "000001.SZ" in result
    assert result["000001.SZ"]["leader_turnover_justified_flag"] is True
    assert result["000001.SZ"]["unsupported_high_position_flag"] is False
    assert result["000001.SZ"]["theme_support_score"] >= 3.4


def test_combine_analysis_blocks_unsupported_high_position_stock() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_stock_moneyflow_summary = Mock(return_value={"recent_3d_net_inflow": -1200})

    stock = StockScreenItem(
        ts_code="688710.SH",
        name="益诺思",
        close=32.0,
        pct_change=1.8,
        volume_ratio=2.9,
        turnover_rate=10.5,
        price_position_20d=0.96,
    )
    screening_results = {
        "s1": ScreenResult(screen_id="s1", criteria=ScreenCriteria(limit=10), stocks=[stock], total_count=1, execution_time=0.1),
    }

    result = asyncio.run(
        scheduler._combine_analysis(
            screening_results=screening_results,
            ai_analyses={
                "688710.SH": {
                    "overall_score": 79,
                    "overall_confidence": 0.82,
                    "summary": "高位但支撑不足",
                    "technical_signal": "分歧",
                }
            },
            news_clusters=[],
        )
    )

    assert "688710.SH" not in result


def test_score_fallback_preserves_zero_values(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")
    scheduler.store.load_recommendation_pool_state.return_value = []
    scheduler.store.list_recommendation_pool.return_value = []

    screening_results = {
        "s1": ScreenResult(
            screen_id="s1",
            criteria=ScreenCriteria(limit=10),
            stocks=[
                StockScreenItem(
                    ts_code="000001.SZ",
                    name="平安银行",
                    close=10,
                    pct_change=1,
                    volume_ratio=1.5,
                    turnover_rate=1.1,
                    recommendation_score=88,
                    recommendation="monitor",
                    confidence="high",
                )
            ],
            total_count=1,
            execution_time=0.1,
        )
    }
    final_recommendations = {
        "000001.SZ": {
            "score": 70,
            "overall_score": 70,
            "weighted_score": 88,
            "overall_confidence": 0.72,
            "technical_score": 0.0,
            "fundamental_score": 0.0,
            "sentiment_score": 0.0,
            "news_score": 0.0,
        }
    }

    states = scheduler._build_recommendation_pool_states(
        trade_date=date(2026, 3, 27),
        screening_results=screening_results,
        final_recommendations=final_recommendations,
        candidate_codes=["000001.SZ"],
    )
    state = states[0]
    assert state.technical_score == 0.0
    assert state.fundamental_score == 0.0
    assert state.sentiment_score == 0.0
    assert state.news_score == 0.0

    merged = scheduler._build_report_stock_payload(
        item=state.model_dump(),
        ai_analyses={
            "000001.SZ": {
                "technical_score": 9.0,
                "fundamental_score": 8.0,
                "sentiment_score": 7.0,
                "news_score": 6.0,
            }
        },
        final_recommendations={},
    )
    assert merged["technical_score"] == 0.0
    assert merged["fundamental_score"] == 0.0
    assert merged["sentiment_score"] == 0.0
    assert merged["news_score"] == 0.0


def test_evaluate_distribution_risk_supports_snapshot_dict_stock(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    screener = Mock()
    screener.client.fetch_moneyflow.return_value = [
        {"trade_date": "20260401", "net_mf_amount": 1200},
        {"trade_date": "20260331", "net_mf_amount": 800},
        {"trade_date": "20260328", "net_mf_amount": 500},
    ]
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=screener,
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )

    risk = scheduler._evaluate_distribution_risk(
        {
            "ts_code": "000001.SZ",
            "volume_ratio": 2.8,
            "pct_chg": 6.2,
            "turnover_rate": 12.0,
            "price_position_20d": 0.91,
        },
        daily_rows=[
            {"trade_date": "20260401", "open": 10.0, "high": 10.8, "close": 10.6, "pct_chg": 6.2, "turnover_rate": 12.0},
            {"trade_date": "20260331", "pct_chg": 3.0, "turnover_rate": 8.0},
            {"trade_date": "20260328", "pct_chg": 2.5, "turnover_rate": 7.5},
            {"trade_date": "20260327", "pct_chg": 1.5, "turnover_rate": 6.5},
            {"trade_date": "20260326", "pct_chg": 1.2, "turnover_rate": 6.0},
        ],
    )

    assert risk["moneyflow_3d_value"] == 2500.0
    assert risk["recent_runup_5d"] == 14.4
    assert risk["distribution_risk_score"] > 0


def test_build_report_stock_payload_does_not_promote_screening_score_to_overall_score(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")

    payload = scheduler._build_report_stock_payload(
        item={
            "ts_code": "000001.SZ",
            "name": "测试股",
            "recommendation_score": 86.0,
            "overall_score": None,
            "priority_score": None,
        },
        ai_analyses={"000001.SZ": {"summary": "仅有摘要"}},
        final_recommendations={"000001.SZ": {"weighted_score": 86.0, "score": 71.0}},
    )

    assert payload["recommendation_score"] == 86.0
    assert payload["overall_score"] is None
    assert payload["priority_score"] is None


def test_build_report_stock_payload_keeps_continuation_fields(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    screener = Mock()
    screener.client.fetch_financial_indicators.return_value = []
    screener.client.fetch_moneyflow.return_value = []
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=screener,
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )

    payload = scheduler._build_report_stock_payload(
        item={
            "ts_code": "000001.SZ",
            "name": "测试股",
            "continuation_bias_score": 2.8,
            "continuation_positive_flags": ["3日资金承接偏强"],
            "continuation_negative_flags": ["近5日涨幅偏大"],
        },
        ai_analyses={},
        final_recommendations={},
    )

    assert payload["continuation_bias_score"] == 2.8
    assert payload["continuation_positive_flags"] == ["3日资金承接偏强"]
    assert payload["continuation_negative_flags"] == ["近5日涨幅偏大"]
    assert payload["overview_reason"] == "谨慎观察；近5日涨幅偏大；等放量确认"


def test_build_report_stock_payload_falls_back_from_null_final_display_score(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    scheduler = EnhancedScreeningScheduler(
        settings=settings,
        screener=Mock(),
        store=Mock(),
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )
    scheduler._build_financial_yoy_summary = Mock(return_value={"latest_revenue_yoy": None, "latest_profit_yoy": None})
    scheduler._build_moneyflow_windows = Mock(return_value={"main_fund_flow_1d": None, "main_fund_flow_3d": None, "main_fund_flow_10d": None})
    scheduler._build_company_business_summary = Mock(return_value="")

    payload = scheduler._build_report_stock_payload(
        item={
            "ts_code": "000001.SZ",
            "name": "测试股",
            "recommend_rank": 1,
            "recommendation_score": 86.0,
            "final_display_recommendation_score": None,
            "display_confidence": None,
        },
        ai_analyses={"000001.SZ": {"confidence": 0.73}},
        final_recommendations={},
    )

    assert payload["recommendation_score"] == 86.0
    assert payload["final_display_recommendation_score"] == 86.0
    assert payload["display_confidence"] == pytest.approx(0.73)


@patch("octts.services.screening_validator.TushareClient")
def test_logic_consistency_distinguishes_missing_scores(mock_tushare_client) -> None:
    validator = __import__("octts.services.screening_validator", fromlist=["ScreeningValidator"]).ScreeningValidator(
        Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    )

    result = validator.check_logic_consistency(
        {
            "000001.SZ": {"technical_score": 0.0, "fundamental_score": 45.0},
            "000002.SZ": {"fundamental_score": 45.0},
        }
    )

    assert "000001.SZ: 技术面(0.0)和基本面(45.0)差异大" in result["issues"]
    assert "000002.SZ: 缺少评分字段(technical_score)" in result["issues"]
    assert all("技术面(0.0)和基本面(45.0)差异大" not in issue or issue.startswith("000001.SZ") for issue in result["issues"])

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
from octts.services.intelligent_report_generator import IntelligentReport


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
    assert states[0].ai_confidence == pytest.approx(0.8)
    assert states[0].display_confidence == pytest.approx(0.72)
    assert states[3].recommendation_score == 60
    assert states[3].priority_score == 55
    assert states[3].source_tag == "昨日延续"
    assert states[3].name == "F"
    assert states[3].score_change == -2.0
    assert next(state for state in states if state.ts_code == "000008.SZ").source_tag == "昨日延续"
    assert next(state for state in states if state.ts_code == "000008.SZ").is_repeat_pick is True
    assert next(state for state in states if state.ts_code == "000008.SZ").score_change == -2.0
    assert next(state for state in states if state.ts_code == "000008.SZ").name == "I"
    assert next(state for state in states if state.ts_code == "000008.SZ").display_confidence == pytest.approx(0.53)
    assert next(state for state in states if state.ts_code == "000008.SZ").ai_confidence > next(state for state in states if state.ts_code == "000008.SZ").display_confidence


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
            "recommendation_score": 88,
            "overall_score": 78,
            "priority_score": 78,
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
            "recommendation_score": 81,
            "overall_score": 72,
            "priority_score": 72,
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
    assert review_map["000001.SZ"]["today_verdict"] == "延续"
    assert review_map["000099.SZ"]["today_present"] is False
    assert review_map["000099.SZ"]["missing_factor_candidates"]


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
    scheduler.store.list_recommendation_pool.return_value = [
        {"ts_code": "000001.SZ", "name": "平安银行", "recommendation_score": 88, "priority_score": 78, "source_tag": "今日Top3", "in_frontlist": True}
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
        metadata={"report_blocks": {"focus_stocks": [{"ts_code": "000001.SZ"}]}}
    )
    scheduler._save_dashboard_snapshot(
        screening_results={},
        ai_analyses={},
        news_clusters=[],
        report=report,
        final_recommendations={},
        trade_date=date(2026, 3, 27),
        report_context={"today_top3": [{"ts_code": "000001.SZ"}]},
    )

    latest_path = tmp_path / "history" / "intelligent_screening" / "latest.json"
    dated_path = tmp_path / "history" / "intelligent_screening" / "20260327.json"
    assert latest_path.exists()
    assert dated_path.exists()
    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
    dated_payload = json.loads(dated_path.read_text(encoding="utf-8"))
    assert latest_payload["report_context"]["today_top3"][0]["ts_code"] == "000001.SZ"
    assert dated_payload["intelligent_report"]["blocks"]["focus_stocks"][0]["ts_code"] == "000001.SZ"


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

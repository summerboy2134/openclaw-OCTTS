import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

from apscheduler.schedulers.background import BackgroundScheduler

from fastapi.testclient import TestClient
from starlette.datastructures import State

from octts.api import _build_stock_intelligent_insight, app, get_active_intelligent_screening_job, get_intelligent_screening_job
from octts.config import Settings
from octts.schemas.backtest import BacktestMetrics, BacktestResult
from octts.schemas.report import (
    DecisionValidation,
    HistoricalAnalysisRecord,
    MemorySummary,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
)
from octts.schemas.screener import ScreenCriteria, ScreenResult, StockScreenItem
from octts.services.history_store import FileHistoryStore
from octts.services.memory_store import FileMemoryStore
from octts.services.position_store import FilePositionStore
from octts.services.stock_screener import StockScreener
from octts.services.intelligent_screening_job_manager import IntelligentScreeningJobManager
from octts.services.stock_screening_scheduler import StockScreeningScheduler
from octts.services.enhanced_screening_scheduler import EnhancedScreeningScheduler


def _build_record() -> HistoricalAnalysisRecord:
    snapshot = PriceSnapshot(
        ts_code="600000.SH",
        trade_date="20260309",
        close=10.2,
        high=10.3,
        low=10.0,
    )
    report = StructuredAnalysis(
        ts_code="600000.SH",
        phase="review",
        trend_judgement="等待向上突破",
        previous_view_status="initial",
        operation_advice="靠近支撑时分批关注",
        risk_warning=["若跌破 9.8 需止损"],
        observation_points=["关注 10.5 压力位"],
        summary_markdown="等待突破。",
        decision=TradingDecision(
            signal="buy",
            rationale="支撑有效且量能温和修复。",
            entry_zone=PriceZone(low=10.0, high=10.2),
            stop_loss=9.8,
            take_profit=[10.5],
            invalidation_condition="放量跌破 9.8",
            holding_horizon="swing",
            confidence_score=0.72,
            risk_reward_ratio=1.8,
            evidence=["支撑位有效"],
        ),
        memory=MemorySummary(
            ts_code="600000.SH",
            phase="review",
            trend_bias="bullish",
            capital_flow_view="主力资金小幅回流",
            confidence_score=0.72,
            summary="等待突破确认",
        ),
    )
    return HistoricalAnalysisRecord(
        record_id="r1",
        request_id="req1",
        generated_at=report.memory.generated_at,
        snapshot=snapshot,
        report=report,
        validation=DecisionValidation(status="entered", note="已进入建议区间。", entry_triggered=True),
    )


def _seed_screening_result(database_url: str) -> None:
    from octts.models.screening_models import DatabaseManager

    manager = DatabaseManager(database_url)
    manager.save_screening_result(
        "oversold_bounce",
        ScreenResult(
            screen_id="screen-db-1",
            criteria=ScreenCriteria(limit=10),
            stocks=[
                StockScreenItem(
                    ts_code="000001.SZ",
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
        ),
    )


def test_dashboard_data_returns_latest_cards(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())
    position_store = FilePositionStore(str(tmp_path / "positions.json"))
    position_store.set_status("600000.SH", "holding")

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="",
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/dashboard/data")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["cards"]) == 1
    assert payload["cards"][0]["ts_code"] == "600000.SH"
    assert payload["cards"][0]["position_status"] == "holding"
    assert payload["default_stock_pool"] == []
    assert "openclaw_status" in payload


def test_dashboard_data_includes_intelligent_screening_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="600000.SH",
            OCTTS_HISTORY_FILE_PATH=str(tmp_path / "history.json"),
            OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )
    monkeypatch.setattr(
        "octts.api._load_intelligent_dashboard_payload",
        lambda settings: {
            "generated_at": "2026-03-24T10:00:00",
            "screening_results": {
                "strategy_count": 4,
                "total_stocks": 28,
                "final_recommendations": 3,
                "frontlist_count": 3,
                "shadow_count": 0,
                "candidate_count": 5,
                "today_top_count": 2,
                "continuation_count": 1,
            },
            "recommendation_pool": {
                "frontlist": [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "priority_score": 92,
                        "recommendation_score": 88,
                        "in_frontlist": True,
                        "tracking_status": "active",
                        "llm_focus_level": "high",
                        "hit_streak_days": 3,
                        "miss_streak_days": 0,
                        "technical_signal": "多头趋势",
                        "recommendation_text": "建议跟踪",
                        "ai_confidence": 0.82,
                        "source_tag": "今日Top3",
                        "is_repeat_pick": True,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "name": "万科A",
                        "priority_score": 80,
                        "recommendation_score": 72,
                        "in_frontlist": True,
                        "tracking_status": "tracking",
                        "llm_focus_level": "medium",
                        "hit_streak_days": 1,
                        "miss_streak_days": 0,
                        "technical_signal": "延续观察",
                        "recommendation_text": "继续观察",
                        "ai_confidence": 0.61,
                        "source_tag": "昨日延续",
                        "is_repeat_pick": False,
                    }
                ],
                "shadow": [],
                "shadow_symbols": [],
                "today_top": [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "priority_score": 92,
                        "recommendation_score": 88,
                        "technical_signal": "多头趋势",
                        "recommendation_text": "建议跟踪",
                        "ai_confidence": 0.82,
                        "source_tag": "今日Top3",
                    }
                ],
                "yesterday_continuations": [
                    {
                        "ts_code": "000002.SZ",
                        "name": "万科A",
                        "priority_score": 80,
                        "recommendation_score": 72,
                        "technical_signal": "延续观察",
                        "recommendation_text": "继续观察",
                        "ai_confidence": 0.61,
                        "source_tag": "昨日延续",
                    }
                ],
            },
            "ai_analyses": {
                "000001.SZ": {
                    "name": "平安银行",
                    "score": 88,
                    "overall_confidence": 0.82,
                    "technical_signal": "多头趋势",
                    "recommendation": "建议跟踪",
                    "summary": "银行板块修复",
                    "technical_summary": "量价共振",
                    "fundamental_summary": "估值稳健",
                    "sentiment_summary": "情绪改善",
                    "news_summary": "银行板块活跃",
                }
            },
            "news_clusters": [{"theme": "银行板块活跃", "key_stocks": ["000001.SZ"]}],
            "intelligent_report": {
                "title": "智能选股报告",
                "summary": "这里是摘要",
                "blocks": {
                    "focus_stocks": [{"ts_code": "000001.SZ", "core_highlights": ["趋势延续"]}],
                    "yesterday_reviews": [{"ts_code": "000002.SZ", "status": "延续"}],
                    "comparison": {"best_short_term": "000001.SZ"},
                    "overall_action": {"headline": "保持节奏", "action_items": ["聚焦 Top3"]},
                },
            },
            "recommendation_summary": {
                "stats": {
                    "window_count": 2,
                    "win_rate_5d": 0.5,
                }
            },
        },
    )

    client = TestClient(app)
    response = client.get("/dashboard/data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["intelligent_screening"]["strategy_count"] == 4
    assert payload["intelligent_screening"]["final_recommendations"] == 3
    assert payload["intelligent_screening"]["news_cluster_count"] == 1
    assert payload["intelligent_screening"]["top_recommendations"][0]["ts_code"] == "000001.SZ"
    assert payload["intelligent_screening"]["top_recommendations"][0]["score"] == 92
    assert payload["intelligent_screening"]["top_recommendations"][0]["overall_score"] == 92
    assert payload["intelligent_screening"]["top_recommendations"][0]["recommendation_score"] == 88
    assert payload["intelligent_screening"]["top_recommendations"][0]["priority_score"] == 92
    assert payload["intelligent_screening"]["top_recommendations"][0]["source_tag"] == "今日Top3"
    assert payload["intelligent_screening"]["top_recommendations"][0]["is_repeat_pick"] is True
    assert payload["intelligent_screening"]["tracked_count"] == 2
    assert payload["intelligent_screening"]["win_rate_5d"] == 0.5
    assert payload["intelligent_screening"]["today_top_count"] == 2
    assert payload["intelligent_screening"]["continuation_count"] == 1
    assert payload["intelligent_screening"]["shadow_count"] == 0
    assert len(payload["intelligent_screening"]["top_recommendations"]) == 2
    assert payload["intelligent_screening"]["report_summary"] == "这里是摘要"


def test_dashboard_route_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "OCTTS Dashboard" in response.text
    assert 'id="runIntelligentScreeningButton"' in response.text
    assert 'id="backtestForm"' in response.text
    assert 'id="backtestTemplateSelect"' in response.text
    assert "回撤曲线" in response.text


def test_root_redirects_to_dashboard() -> None:
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard"


def test_favicon_returns_empty_response() -> None:
    client = TestClient(app)
    response = client.get("/favicon.ico")

    assert response.status_code == 204
    assert response.text == ""


def test_stock_detail_data_returns_symbol_payload(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())
    position_store = FilePositionStore(str(tmp_path / "positions.json"))
    position_store.set_status("600000.SH", "holding")

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
            OPENCLAW_GATEWAY_URL="http://127.0.0.1:18789",
            OPENCLAW_HOOKS_ENABLED=True,
        ),
    )

    client = TestClient(app)
    response = client.get("/stocks/600000.SH/data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"]["ts_code"] == "600000.SH"
    assert payload["openclaw_status"]["connected"] is True
    assert payload["position_status"] == "holding"
    assert payload["intelligent_screening_insight"]["ts_code"] == "600000.SH"


def test_stock_detail_data_includes_intelligent_screening_insight(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
            OCTTS_HISTORY_DIR=str(tmp_path / "historydir"),
        ),
    )
    monkeypatch.setattr(
        "octts.api._load_intelligent_dashboard_payload",
        lambda settings, trade_date=None: {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "600000.SH", "source_tag": "今日Top3", "recommendation_score": 88, "priority_score": 77, "ai_confidence": 0.8}],
                "today_top": [{"ts_code": "600000.SH", "action_plan": {"entry_zone": "10-10.2", "take_profit": "10.8", "stop_loss": "9.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破支撑"}}],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600000.SH", "recommendation_score": 88, "overall_score": 77, "display_confidence": 0.8, "recommendation_text": "继续观察", "technical_signal": "量价共振", "action_plan": {"action_bias": "观察"}}],
                "yesterday_top3_review": [{"ts_code": "600000.SH", "today_verdict": "延续", "previous_recommendation_score": 82}],
            },
            "intelligent_report": {
                "blocks": {
                    "focus_stocks": [{"ts_code": "600000.SH", "core_highlights": ["趋势延续"], "risk_warnings": ["量能待确认"], "overall_assessment": "适合继续跟踪", "action_plan": {"entry_zone": "10-10.2", "take_profit": "10.8", "stop_loss": "9.8", "invalid_condition": "跌破支撑"}}],
                    "yesterday_reviews": [{"ts_code": "600000.SH", "today_verdict": "延续", "status": "延续"}],
                }
            },
        },
    )

    client = TestClient(app)
    response = client.get("/stocks/600000.SH/data")
    assert response.status_code == 200
    payload = response.json()
    insight = payload["intelligent_screening_insight"]
    assert insight["in_today_top3"] is True
    assert insight["core_highlights"] == ["趋势延续"]
    assert insight["overall_assessment"] == "适合继续跟踪"
    assert insight["action_plan"]["entry_zone"] == "10-10.2"
    assert insight["action_plan"]["take_profit"] == "10.8"
    assert insight["action_plan"]["stop_loss"] == "9.8"
    assert insight["action_plan"]["invalid_condition"] == "跌破支撑"
    assert insight["yesterday_vs_today"]["today_verdict"] == "延续"


def test_build_stock_intelligent_insight_uses_review_only_action_plan_fallback() -> None:
    insight = _build_stock_intelligent_insight(
        "000002.SZ",
        {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "000002.SZ", "source_tag": "昨日延续", "recommendation_score": 72, "priority_score": 70, "ai_confidence": 0.61}],
                "yesterday_continuations": [{"ts_code": "000002.SZ", "source_tag": "昨日延续", "technical_signal": "转入跟踪"}],
            },
            "report_context": {"yesterday_top3_review": [{"ts_code": "000002.SZ", "review_status": "减仓观察"}]},
            "intelligent_report": {
                "blocks": {
                    "yesterday_reviews": [{
                        "ts_code": "000002.SZ",
                        "today_verdict": "跌破承接，考虑离场",
                        "review_status": "减仓观察",
                        "action_plan": {
                            "take_profit": "反抽到 12.6 附近止盈",
                            "stop_loss": "跌破 11.8 离场",
                            "invalid_condition": "午后不能重新站回 12.0",
                        },
                    }]
                }
            },
        },
    )

    assert insight["in_yesterday_review"] is True
    assert insight["action_plan"]["take_profit"] == "反抽到 12.6 附近止盈"
    assert insight["action_plan"]["stop_loss"] == "跌破 11.8 离场"
    assert insight["action_plan"]["invalid_condition"] == "午后不能重新站回 12.0"
    assert insight["yesterday_vs_today"]["today_verdict"] == "跌破承接，考虑离场"
    assert insight["yesterday_vs_today"]["review_status"] == "减仓观察"


def test_intelligent_screening_page_restores_sections_and_data_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json"),
    )
    monkeypatch.setattr(
        "octts.api._build_recommendation_dashboard_payload",
        lambda settings: {
            "generated_at": "2026-03-30T09:15:00",
            "screening_results": {
                "strategy_count": 4,
                "total_stocks": 28,
                "final_recommendations": 3,
            },
            "recommendation_pool": {
                "frontlist": [],
                "today_top": [{"ts_code": "000001.SZ", "name": "平安银行", "recommendation_score": 88, "priority_score": 92, "source_tag": "今日Top3", "technical_signal": "多头趋势", "recommendation_text": "建议跟踪", "ai_confidence": 0.82, "action_plan": {"entry_zone": "10.1-10.3", "stop_loss": "9.8", "take_profit": "10.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破 9.8"}}],
                "yesterday_continuations": [{"ts_code": "000002.SZ", "name": "万科A", "recommendation_score": 72, "priority_score": 80, "source_tag": "昨日延续", "technical_signal": "延续观察", "recommendation_text": "继续观察", "ai_confidence": 0.61, "action_plan": {"take_profit": "反抽 8.8 附近止盈", "stop_loss": "跌破 8.1 离场", "invalid_condition": "全天弱于地产板块", "holding_horizon": "2-3个交易日"}}],
            },
            "ai_analyses": {
                "000001.SZ": {
                    "name": "平安银行",
                    "overall_score": 88,
                    "overall_confidence": 0.82,
                    "technical_signal": "多头趋势",
                    "recommendation": "建议跟踪",
                    "summary": "银行板块修复",
                    "technical_summary": "量价共振",
                    "fundamental_summary": "估值稳健",
                    "sentiment_summary": "情绪改善",
                    "news_summary": "银行板块活跃",
                    "key_points": ["趋势延续", "量能修复"],
                }
            },
            "news_clusters": [{"theme": "银行板块活跃", "key_stocks": ["000001.SZ"]}],
            "intelligent_report": {
                "title": "智能选股报告",
                "summary": "这里是摘要",
                "blocks": {
                    "focus_stocks": [{"ts_code": "000001.SZ", "core_highlights": ["趋势延续"], "action_plan": {"entry_zone": "10.1-10.3", "stop_loss": "9.8", "take_profit": "10.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破 9.8"}, "overall_assessment": "优先等回踩再介入"}],
                    "yesterday_reviews": [{"ts_code": "000002.SZ", "status": "延续", "today_verdict": "不能转强则离场", "action_plan": {"take_profit": "反抽 8.8 附近止盈", "stop_loss": "跌破 8.1 离场", "invalid_condition": "全天弱于地产板块", "holding_horizon": "2-3个交易日"}, "analysis": "反弹力度不足，先看能否守住支撑"}],
                    "comparison": {"best_short_term": "000001.SZ"},
                    "overall_action": {"headline": "保持节奏", "action_items": ["聚焦 Top3"]},
                },
            },
            "recommendation_summary": {"stats": {"window_count": 2, "win_rate_5d": 0.5}},
            "recommendation_methodology": {"strategy_count": 4, "candidate_selection": ["多策略共振"], "score_formula": ["按综合分排序"]},
        },
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening")

    assert response.status_code == 200
    assert "今日 Top3" in response.text
    assert "昨日 Top3 今日复盘 / 昨日延续" in response.text
    assert "重点个股分析" in response.text
    assert "智能报告" in response.text
    assert "银行板块修复" in response.text
    assert "量价共振" in response.text
    assert "估值稳健" in response.text
    assert "情绪改善" in response.text
    assert "银行板块活跃" in response.text
    assert "买入区间" in response.text
    assert "止损位" in response.text
    assert "第一止盈位" in response.text
    assert "失效条件" in response.text
    assert "今日结论" in response.text
    assert "转弱止损/离场" in response.text
    assert "离场触发条件" in response.text
    assert "保持节奏" in response.text


def test_intelligent_screening_page_shows_empty_win_rate_and_total_when_no_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json"),
    )
    monkeypatch.setattr(
        "octts.api._build_recommendation_dashboard_payload",
        lambda settings: {
            "generated_at": None,
            "screening_results": {"strategy_count": None, "total_stocks": None, "final_recommendations": 0},
            "recommendation_pool": {"frontlist": [], "today_top": [], "yesterday_continuations": []},
            "ai_analyses": {},
            "news_clusters": [],
            "intelligent_report": {},
            "recommendation_summary": {"stats": {}},
            "recommendation_methodology": {"strategy_count": 6, "tracking_metrics": ["5日胜率定义为 return_5d > 0"]},
        },
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening")

    assert response.status_code == 200
    assert "暂无本次运行数据" in response.text
    assert ">--<" in response.text
    assert "若未开启数据库或暂无已验证样本，则显示 --" in response.text


def test_stock_detail_page_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/stocks/600000.SH")

    assert response.status_code == 200
    assert "600000.SH 单股详情" in response.text
    assert 'id="positionStatusSelect"' in response.text
    assert 'id="reanalyzeSymbolButton"' in response.text


def test_update_position_status_persists_selection(tmp_path, monkeypatch) -> None:
    position_path = tmp_path / "positions.json"
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_POSITION_FILE_PATH=str(position_path),
            OCTTS_STOCK_POOL="",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.put("/positions/600000.sh", json={"position_status": "holding"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ts_code"] == "600000.SH"
    assert payload["position_status"] == "holding"
    assert payload["stock_pool"] == ["600000.SH"]
    assert FilePositionStore(str(position_path)).get_status("600000.SH") == "holding"
    assert "600000.SH" in payload["stock_pool"]


def test_openclaw_status_endpoint_uses_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_AUTOMATION_ENABLED=True,
            OCTTS_AUTOMATION_TIMEZONE="Asia/Shanghai",
            OPENCLAW_AGENT_ID="octts",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH="memory/latest_memory.json",
        ),
    )

    client = TestClient(app)
    response = client.get("/openclaw/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["automation_enabled"] is True
    assert payload["automation_timezone"] == "Asia/Shanghai"
    assert len(payload["automation_slots"]) == 1
    assert payload["automation_slots"][0]["phase"] == "review"


def test_backtest_endpoint_returns_result(monkeypatch) -> None:
    class DummyBacktestEngine:
        def run(self, request):
            assert request.phase == "review"
            return BacktestResult(
                phase="review",
                stock_pool=["600000.SH"],
                start_date="20260101",
                end_date="20260110",
                initial_cash=100000,
                ending_cash=101000,
                metrics=BacktestMetrics(trade_count=1, total_return=0.01),
            )

    monkeypatch.setattr("octts.api._build_backtest_engine", lambda: DummyBacktestEngine())

    client = TestClient(app)
    response = client.post(
        "/backtest",
        json={
            "stock_pool": ["600000.SH"],
            "start_date": "20260101",
            "end_date": "20260110",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["phase"] == "review"
    assert payload["metrics"]["trade_count"] == 1


def test_add_stock_pool_item_persists_to_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OCTTS_STOCK_POOL=600000.SH\n", encoding="utf-8")

    monkeypatch.setattr("octts.api._env_file_path", lambda: env_path)
    monkeypatch.setattr("octts.api._clear_settings_cache", lambda: None)
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="600000.SH",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.post("/stock-pool", json={"ts_code": "000001.sz"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["stock_pool"] == ["600000.SH", "000001.SZ"]
    assert "OCTTS_STOCK_POOL=600000.SH,000001.SZ" in env_path.read_text(encoding="utf-8")


def test_remove_stock_pool_item_persists_to_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OCTTS_STOCK_POOL=600000.SH,000001.SZ\n", encoding="utf-8")

    monkeypatch.setattr("octts.api._env_file_path", lambda: env_path)
    monkeypatch.setattr("octts.api._clear_settings_cache", lambda: None)
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="600000.SH,000001.SZ",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.delete("/stock-pool/000001.SZ")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stock_pool"] == ["600000.SH"]
    assert "OCTTS_STOCK_POOL=600000.SH" in env_path.read_text(encoding="utf-8")


def test_clear_symbol_analysis_data_removes_history_and_memory(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history"
    memory_path = tmp_path / "memory.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())
    memory_store = FileMemoryStore(str(memory_path))
    memory_store.set(_build_record().report.memory)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(history_path),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(memory_path),
        ),
    )

    client = TestClient(app)
    response = client.delete("/analysis-data/600000.SH")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cleared_symbols"] == ["600000.SH"]
    assert payload["removed_records"] == 1
    assert FileHistoryStore(str(history_path)).list_records("600000.SH") == []
    assert FileMemoryStore(str(memory_path)).get("600000.SH") is None


def test_clear_all_analysis_data_removes_everything(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history"
    memory_path = tmp_path / "memory.json"
    store = FileHistoryStore(str(history_path))
    first = _build_record()
    second = _build_record()
    second.report.ts_code = "000001.SZ"
    second.snapshot.ts_code = "000001.SZ"
    second.report.memory.ts_code = "000001.SZ"
    store.append(first)
    store.append(second)

    memory_store = FileMemoryStore(str(memory_path))
    memory_store.set(first.report.memory)
    memory_store.set(second.report.memory)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(history_path),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(memory_path),
        ),
    )

    client = TestClient(app)
    response = client.delete("/analysis-data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cleared_all"] is True
    assert payload["removed_records"] == 2
    assert payload["removed_memory_items"] == 2
    assert FileHistoryStore(str(history_path)).list_latest() == []
    assert FileMemoryStore(str(memory_path)).get("600000.SH") is None
    assert FileMemoryStore(str(memory_path)).get("000001.SZ") is None


def test_screening_history_reads_database_results(tmp_path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'screening.db'}"
    _seed_screening_result(database_url)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_USE_DATABASE=True,
            OCTTS_DATABASE_URL=database_url,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/screen/history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 1
    assert payload["history"][0]["strategy"] == "oversold_bounce"
    assert payload["history"][0]["total_stocks"] == 1


def test_stock_performance_reads_database_results(tmp_path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'screening.db'}"
    _seed_screening_result(database_url)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_USE_DATABASE=True,
            OCTTS_DATABASE_URL=database_url,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/stock/000001.SZ/performance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ts_code"] == "000001.SZ"
    assert payload["total_appearances"] == 1
    assert payload["appearances"][0]["strategy"] == "oversold_bounce"


def test_screen_result_reads_database_when_cache_is_empty(tmp_path, monkeypatch) -> None:
    database_url = f"sqlite:///{tmp_path / 'screening.db'}"
    _seed_screening_result(database_url)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_USE_DATABASE=True,
            OCTTS_DATABASE_URL=database_url,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/screen/results/screen-db-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["screen_id"] == "screen-db-1"
    assert payload["total_count"] == 1
    assert payload["stocks"][0]["ts_code"] == "000001.SZ"


def test_build_intelligent_overview_payload_prefers_recommendation_score() -> None:
    from octts.api import _build_intelligent_overview_payload

    payload = {
        "ai_analyses": {
            "000001.SZ": {
                "name": "平安银行",
                "score": 92,
                "priority_score": 92,
                "recommendation_score": 58.8,
                "overall_score": 58.8,
                "overall_confidence": 0.81,
                "technical_signal": "多头趋势",
                "recommendation": "建议跟踪",
                "summary": "推荐分数应对外统一。",
            },
            "000002.SZ": {
                "name": "万科A",
                "score": 70,
                "priority_score": 70,
                "recommendation_score": 52.0,
                "overall_score": 52.0,
                "technical_signal": "等待确认",
                "recommendation": "继续观察",
                "summary": "未命中 AI 置信度时应使用展示池回填。",
            },
        },
        "recommendation_pool": {
            "frontlist": [
                {
                    "ts_code": "000001.SZ",
                    "name": "平安银行",
                    "priority_score": 92,
                    "recommendation_score": 79.4,
                    "recommendation_text": "建议跟踪",
                    "source_tag": "今日Top3",
                    "ai_confidence": 0.81,
                },
                {
                    "ts_code": "000002.SZ",
                    "name": "万科A",
                    "priority_score": 70,
                    "recommendation_score": 52.0,
                    "recommendation_text": "继续观察",
                    "source_tag": "今日候选",
                    "ai_confidence": 0.64,
                }
            ],
            "shadow": [],
            "shadow_symbols": [],
        },
        "screening_results": {},
        "intelligent_report": {},
        "recommendation_summary": {},
    }

    result = _build_intelligent_overview_payload(payload)

    assert result["top_recommendations"][0]["score"] == 58.8
    assert result["top_recommendations"][0]["recommendation_score"] == 58.8
    assert result["top_recommendations"][0]["priority_score"] == 92
    assert result["top_recommendations"][1]["source_tag"] == "今日候选"
    assert result["top_recommendations"][1]["confidence"] == 0.64



def test_configure_logging_uses_daily_rotation(monkeypatch, tmp_path) -> None:
    from octts.api import _configure_logging

    settings = Settings(
        OCTTS_HISTORY_FILE_PATH=str(tmp_path / "history" / "history.json"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    root_logger.handlers = [
        handler for handler in root_logger.handlers if getattr(handler, "baseFilename", None)
    ]

    try:
        monkeypatch.setattr("octts.api.get_settings", lambda: settings)
        _configure_logging()

        log_path = str((tmp_path / "history" / "logs" / "app.log").resolve())
        handler = next(
            handler for handler in logging.getLogger().handlers if getattr(handler, "baseFilename", None) == log_path
        )

        assert handler.when.upper() == "MIDNIGHT"
        assert handler.backupCount == 30
        assert handler.suffix == "%Y-%m-%d"
    finally:
        for handler in list(root_logger.handlers):
            if handler not in original_handlers:
                root_logger.removeHandler(handler)
                handler.close()
        root_logger.handlers = original_handlers



def test_intelligent_screening_page_reads_saved_snapshot(tmp_path, monkeypatch) -> None:
    snapshot_dir = tmp_path / "history" / "intelligent_screening"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-26T09:35:22",
                "screening_results": {"total_stocks": 12},
                "ai_analyses": {
                    "000001.SZ": {
                        "name": "平安银行",
                        "score": 88,
                        "overall_confidence": 0.82,
                        "recommendation": "建议继续跟踪",
                        "summary": "真实快照摘要",
                        "technical_summary": "量价配合较好",
                        "technical_signal": "多头趋势",
                        "key_points": ["技术评分高"],
                    }
                },
                "news_clusters": [
                    {
                        "theme": "银行板块活跃",
                        "importance": 0.8,
                        "summary": "资金回流银行板块",
                        "key_stocks": ["000001.SZ"],
                        "news_items": [{"title": "测试新闻"}],
                    }
                ],
                "intelligent_report": {
                    "title": "真实智能报告",
                    "summary": "这里应该展示真实快照",
                    "sections": [{"title": "重点", "content": "测试内容"}],
                },
                "recommendation_summary": {
                    "new_recommendations": [
                        {
                            "ts_code": "000001.SZ",
                            "name": "平安银行",
                            "recommend_score": 88,
                            "ai_confidence": 0.82,
                            "strategy_count": 3,
                            "recommendation_text": "建议继续跟踪",
                        }
                    ],
                    "tracking_recommendations": [
                        {
                            "ts_code": "000001.SZ",
                            "name": "平安银行",
                            "status": "tracking",
                            "tracking_days": 3,
                            "entry_price": 10.0,
                            "latest_price": 10.6,
                            "return_1d": 0.02,
                            "return_3d": 0.04,
                            "return_5d": 0.06,
                            "max_drawdown_10d": -0.02,
                        }
                    ],
                    "stats": {
                        "tracked_count": 1,
                        "win_rate_5d": 0.6,
                        "average_return_5d": 0.05,
                        "average_vs_benchmark_5d": 0.03,
                        "benchmark_win_rate_5d": 0.55,
                        "profit_loss_ratio_5d": 1.8,
                        "repeat_recommendations": [
                            {"ts_code": "000001.SZ", "recommendation_count": 2, "average_return_5d": 0.05}
                        ],
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening")

    assert response.status_code == 200
    assert "真实智能报告" in response.text
    assert "这里应该展示真实快照" in response.text
    assert "最近生成时间：" in response.text
    assert "推荐列表" in response.text


def test_intelligent_screening_payload_comes_from_report_exporter(tmp_path, monkeypatch) -> None:
    payload = {
        "generated_at": "2026-03-26T09:35:22",
        "screening_results": {"frontlist_count": 1},
        "recommendation_pool": {"frontlist": [], "shadow": [], "shadow_symbols": []},
        "ai_analyses": {"000001.SZ": {"name": "平安银行", "summary": "统一快照摘要", "technical_signal": "多头", "score": 88}},
        "news_clusters": [],
        "intelligent_report": {"title": "统一来源报告", "summary": "统一来源摘要"},
        "recommendation_summary": {"frontlist": ["000001.SZ"]},
        "recommendation_methodology": {"strategy_count": 1},
    }

    class FakeExporter:
        def build_intelligent_screening_payload(self):
            return payload

    monkeypatch.setattr("octts.api._build_report_exporter_with_settings", lambda settings: FakeExporter())
    monkeypatch.setattr("octts.api._load_intelligent_dashboard_payload", lambda settings: dict(payload))
    monkeypatch.setattr("octts.api._load_recommendation_summary", lambda settings: payload["recommendation_summary"])
    monkeypatch.setattr("octts.api._build_recommendation_methodology_payload", lambda settings: payload["recommendation_methodology"])
    result = __import__("octts.api", fromlist=["_build_recommendation_dashboard_payload"])._build_recommendation_dashboard_payload(Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json")))

    assert result["generated_at"] == payload["generated_at"]
    assert result["ai_analyses"]["000001.SZ"]["summary"] == "统一快照摘要"
    assert result["intelligent_report"]["summary"] == "统一来源摘要"
    assert result["recommendation_summary"]["frontlist"] == ["000001.SZ"]


def test_intelligent_screening_page_tolerates_none_fields_in_snapshot(tmp_path, monkeypatch) -> None:
    snapshot_dir = tmp_path / "history" / "intelligent_screening"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "latest.json").write_text(
        json.dumps(
            {
                "screening_results": {"total_stocks": 1},
                "ai_analyses": {
                    "000001.SZ": {
                        "name": None,
                        "score": 88,
                        "overall_confidence": 0.82,
                        "recommendation": None,
                        "summary": None,
                        "technical_summary": None,
                        "technical_signal": None,
                        "key_points": ["技术评分高"],
                        "final_decision": None,
                    }
                },
                "news_clusters": [],
                "intelligent_report": {
                    "title": "容错报告",
                    "summary": "允许空字段",
                    "sections": [],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening")

    assert response.status_code == 200
    assert "容错报告" in response.text
    assert "暂无最近运行时间" in response.text
    assert "推荐列表" in response.text
    assert "octts:intelligent-screening:pending-job-id" in response.text
    assert 'params.get("job_id")' in response.text
    assert 'fetch(`${jobsApiBase}/active`)' in response.text
    assert "setJobState(job)" in response.text


def test_intelligent_screening_job_endpoints_report_progress(tmp_path, monkeypatch) -> None:
    class FakeScheduler:
        def __init__(self, settings, progress_callback=None):
            self.settings = settings
            self.progress_callback = progress_callback

        async def run_intelligent_screening(self):
            del self.settings
            if self.progress_callback:
                await self.progress_callback(
                    {
                        "status": "running",
                        "current_step": 1,
                        "total_steps": 3,
                        "step_name": "新闻采集",
                        "progress_percent": 18,
                        "message": "正在采集最新市场新闻...",
                        "details": {"news_count": 12},
                    }
                )
            await asyncio.sleep(0.02)
            if self.progress_callback:
                await self.progress_callback(
                    {
                        "status": "running",
                        "current_step": 2,
                        "total_steps": 3,
                        "step_name": "AI 深度分析",
                        "progress_percent": 72,
                        "message": "AI 分析进度：1/2",
                        "details": {
                            "current_symbol": "000001.SZ",
                            "completed_items": 1,
                            "total_items": 2,
                        },
                    }
                )
            await asyncio.sleep(0.02)
            return {
                "success": True,
                "current_step": 3,
                "total_steps": 3,
                "screened_stocks": 8,
                "final_recommendations": 2,
                "frontlist_count": 2,
                "tracking_pool_count": 0,
                "report_id": "report-job-1",
            }

    monkeypatch.setattr("octts.api.EnhancedScreeningScheduler", FakeScheduler)
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    with TestClient(app) as client:
        response = client.post("/screen/intelligent/jobs")
        assert response.status_code == 202
        payload = response.json()
        assert payload["created"] is True
        assert payload["job_id"]

        job_id = payload["job_id"]
        saw_progress = False
        final_payload = None

        for _ in range(40):
            status_response = client.get(f"/screen/intelligent/jobs/{job_id}")
            assert status_response.status_code == 200
            status_payload = status_response.json()
            if status_payload["progress_percent"] >= 18:
                saw_progress = True
            if status_payload["status"] == "succeeded":
                final_payload = status_payload
                break
            time.sleep(0.02)

        assert saw_progress is True
        assert final_payload is not None
        assert final_payload["progress_percent"] == 100
        assert final_payload["result"]["final_recommendations"] == 2
        assert final_payload["message"] == "智能选股完成：前台推荐 2 只，跟踪池 0 只。"


def test_automation_scheduler_uses_intelligent_screening_runner(monkeypatch) -> None:
    scheduler_mock = Mock()
    scheduler_mock.run_intelligent_screening = AsyncMock(return_value={"success": True})
    factory = Mock(return_value=scheduler_mock)

    from octts.services.automation_scheduler import _run_scheduled_screening

    _run_scheduled_screening(screening_scheduler_factory=factory)

    factory.assert_called_once_with()
    scheduler_mock.run_intelligent_screening.assert_awaited_once()



def test_api_build_screening_scheduler_returns_enhanced_scheduler(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    scheduler = __import__("octts.api", fromlist=["_build_screening_scheduler"])._build_screening_scheduler()

    assert isinstance(scheduler, EnhancedScreeningScheduler)



def test_daily_screening_scheduler_writes_compat_intelligent_snapshot(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    strategies = StockScreener.get_presets()[:1]
    snapshot = {
        "trade_date": "20240325",
        "created_at": "2026-03-25T00:00:00",
        "stocks": [],
        "daily_basic": {},
        "daily": {},
    }

    client = Mock()
    client.get_or_build_screening_snapshot.return_value = snapshot
    screener = Mock()
    screener.client = client
    screener._get_latest_trade_date.return_value = "20240325"
    screener.screen.return_value = ScreenResult(
        screen_id="screen-1",
        criteria=strategies[0].criteria,
        stocks=[
            StockScreenItem(
                ts_code="000001.SZ",
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
        execution_time=0.01,
    )

    store = Mock()
    store.save_screening_result = AsyncMock()
    report_service = Mock()
    report_service.generate_daily_report = AsyncMock(
        return_value={
            "report_id": "screening_20260326_191614",
            "report_time": "2026-03-26T19:16:14",
            "strategy_results": [
                {
                    "strategy_name": strategies[0].name,
                    "top_stocks": [
                        {
                            "ts_code": "000001.SZ",
                            "name": "平安银行",
                            "pct_change": 2.3,
                        }
                    ],
                }
            ],
        }
    )

    scheduler = StockScreeningScheduler(
        settings,
        screener=screener,
        store=store,
        report_service=report_service,
    )

    with patch.object(scheduler, "_get_active_strategies", return_value=strategies):
        asyncio.run(scheduler.run_daily_screening())

    snapshot_path = tmp_path / "history" / "intelligent_screening" / "latest.json"
    assert snapshot_path.exists()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["generated_at"] == "2026-03-26T19:16:14"
    assert payload["screening_results"]["source"] == "daily_screening_compat"
    assert payload["ai_analyses"]["000001.SZ"]["name"] == "平安银行"



def test_stock_screening_scheduler_reuses_shared_snapshot_once(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    strategies = StockScreener.get_presets()[:4]
    snapshot = {
        "trade_date": "20240325",
        "created_at": "2026-03-25T00:00:00",
        "stocks": [],
        "daily_basic": {},
        "daily": {},
    }

    client = Mock()
    client.get_or_build_screening_snapshot.return_value = snapshot
    screener = Mock()
    screener.client = client
    screener._get_latest_trade_date.return_value = "20240325"
    screener.screen.side_effect = [
        ScreenResult(
            screen_id=f"screen-{i}",
            criteria=strategy.criteria,
            stocks=[],
            total_count=0,
            execution_time=0.01,
        )
        for i, strategy in enumerate(strategies, start=1)
    ]

    store = Mock()
    store.save_screening_result = AsyncMock()
    report_service = Mock()
    report_service.generate_daily_report = AsyncMock(return_value={"report_id": "report-1"})

    scheduler = StockScreeningScheduler(
        settings,
        screener=screener,
        store=store,
        report_service=report_service,
    )

    with patch.object(scheduler, "_get_active_strategies", return_value=strategies):
        result = asyncio.run(scheduler.run_daily_screening())

    assert result["strategies_run"] == 4
    assert result["total_stocks"] == 0
    client.get_or_build_screening_snapshot.assert_called_once_with("20240325")
    assert screener.screen.call_count == 4
    for call in screener.screen.call_args_list:
        assert call.kwargs["trade_date"] == "20240325"
        assert call.kwargs["market_snapshot"] is snapshot

    report_service.generate_daily_report.assert_awaited_once()



def test_enhanced_scheduler_reuses_shared_snapshot_once(tmp_path) -> None:
    settings = Settings(
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    strategies = StockScreener.get_presets()[:4]
    snapshot = {
        "trade_date": "20240325",
        "created_at": "2026-03-25T00:00:00",
        "stocks": [],
        "daily_basic": {},
        "daily": {},
    }

    client = Mock()
    client.get_or_build_screening_snapshot.return_value = snapshot
    screener = Mock()
    screener.client = client
    screener._get_latest_trade_date.return_value = "20240325"
    screener.screen.side_effect = [
        ScreenResult(
            screen_id=f"screen-{i}",
            criteria=strategy.criteria,
            stocks=[],
            total_count=0,
            execution_time=0.01,
        )
        for i, strategy in enumerate(strategies, start=1)
    ]

    store = Mock()
    store.save_screening_result = AsyncMock()

    scheduler = EnhancedScreeningScheduler(
        settings,
        screener=screener,
        store=store,
        analyzer=Mock(),
        news_aggregator=Mock(),
        report_generator=Mock(),
    )

    with patch.object(scheduler, "_get_active_strategies", return_value=strategies):
        results, trade_date = asyncio.run(scheduler._run_screening_strategies())

    assert trade_date == "20240325"
    assert len(results) == 4
    client.get_or_build_screening_snapshot.assert_called_once_with("20240325")
    assert screener.screen.call_count == 4
    for call in screener.screen.call_args_list:
        assert call.kwargs["trade_date"] == "20240325"
        assert call.kwargs["market_snapshot"] is snapshot


def test_active_intelligent_screening_job_endpoint_reads_shared_snapshot_when_memory_empty(tmp_path) -> None:
    manager = IntelligentScreeningJobManager(str(tmp_path / "history"))
    job = {
        "job_id": "shared-job-1",
        "status": "running",
        "created_at": "2026-03-27T10:00:00",
        "started_at": "2026-03-27T10:00:01",
        "finished_at": None,
        "progress_percent": 42,
        "current_step": 2,
        "total_steps": 5,
        "step_name": "AI 深度分析",
        "message": "跨进程任务运行中",
        "details": {"current_symbol": "000001.SZ"},
        "result": None,
        "error": None,
        "is_active": True,
    }
    snapshot_dir = tmp_path / "history" / "intelligent_screening_jobs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "active.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    (snapshot_dir / "shared-job-1.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

    app_like = type("AppLike", (), {"state": State()})()
    app_like.state.intelligent_screening_job_manager = manager
    request = Mock()
    request.app = app_like

    payload = asyncio.run(get_active_intelligent_screening_job(request))

    assert payload["job"] is not None
    assert payload["job"]["job_id"] == "shared-job-1"
    assert payload["job"]["progress_percent"] == 42



def test_intelligent_screening_job_detail_reads_shared_snapshot_when_memory_empty(tmp_path) -> None:
    manager = IntelligentScreeningJobManager(str(tmp_path / "history"))
    job = {
        "job_id": "shared-job-2",
        "status": "running",
        "created_at": "2026-03-27T10:00:00",
        "started_at": "2026-03-27T10:00:01",
        "finished_at": None,
        "progress_percent": 55,
        "current_step": 3,
        "total_steps": 5,
        "step_name": "生成推荐",
        "message": "共享快照可查询详情",
        "details": {},
        "result": None,
        "error": None,
        "is_active": True,
    }
    snapshot_dir = tmp_path / "history" / "intelligent_screening_jobs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "shared-job-2.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

    app_like = type("AppLike", (), {"state": State()})()
    app_like.state.intelligent_screening_job_manager = manager
    request = Mock()
    request.app = app_like

    payload = asyncio.run(get_intelligent_screening_job("shared-job-2", request))

    assert payload["job_id"] == "shared-job-2"
    assert payload["message"] == "共享快照可查询详情"



def test_active_intelligent_screening_job_endpoint_falls_back_to_latest_running_snapshot(tmp_path) -> None:
    manager = IntelligentScreeningJobManager(str(tmp_path / "history"))
    snapshot_dir = tmp_path / "history" / "intelligent_screening_jobs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    older_job = {
        "job_id": "shared-job-old",
        "status": "running",
        "created_at": "2026-03-27T09:00:00",
        "started_at": "2026-03-27T09:00:01",
        "finished_at": None,
        "progress_percent": 20,
        "current_step": 1,
        "total_steps": 4,
        "step_name": "技术选股",
        "message": "旧任务仍在运行",
        "details": {},
        "result": None,
        "error": None,
        "is_active": True,
    }
    newer_job = {
        "job_id": "shared-job-new",
        "status": "running",
        "created_at": "2026-03-27T10:00:00",
        "started_at": "2026-03-27T10:00:01",
        "finished_at": None,
        "progress_percent": 68,
        "current_step": 3,
        "total_steps": 4,
        "step_name": "AI 深度分析",
        "message": "新任务运行中",
        "details": {},
        "result": None,
        "error": None,
        "is_active": True,
    }
    (snapshot_dir / "shared-job-old.json").write_text(json.dumps(older_job, ensure_ascii=False), encoding="utf-8")
    (snapshot_dir / "shared-job-new.json").write_text(json.dumps(newer_job, ensure_ascii=False), encoding="utf-8")

    app_like = type("AppLike", (), {"state": State()})()
    app_like.state.intelligent_screening_job_manager = manager
    request = Mock()
    request.app = app_like

    payload = asyncio.run(get_active_intelligent_screening_job(request))

    assert payload["job"] is not None
    assert payload["job"]["job_id"] == "shared-job-new"
    assert payload["job"]["progress_percent"] == 68



def test_start_job_marks_stale_shared_active_snapshot_before_creating_new_job(tmp_path) -> None:
    manager = IntelligentScreeningJobManager(str(tmp_path / "history"))
    snapshot_dir = tmp_path / "history" / "intelligent_screening_jobs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    stale_job = {
        "job_id": "stale-job-1",
        "status": "running",
        "created_at": "2026-03-27T07:00:00",
        "started_at": "2026-03-27T07:00:01",
        "finished_at": None,
        "progress_percent": 12,
        "current_step": 1,
        "total_steps": 6,
        "step_name": "新闻采集",
        "message": "旧任务残留中",
        "details": {},
        "result": None,
        "error": None,
        "is_active": True,
    }
    (snapshot_dir / "active.json").write_text(json.dumps(stale_job, ensure_ascii=False), encoding="utf-8")
    (snapshot_dir / "stale-job-1.json").write_text(json.dumps(stale_job, ensure_ascii=False), encoding="utf-8")

    async def runner(_progress_callback):
        await asyncio.sleep(0.01)
        return {
            "current_step": 1,
            "total_steps": 1,
            "frontlist_count": 0,
            "tracking_pool_count": 0,
        }

    payload = asyncio.run(manager.start_job(runner))

    assert payload["created"] is True
    assert payload["job"]["job_id"] != "stale-job-1"
    stale_snapshot = json.loads((snapshot_dir / "stale-job-1.json").read_text(encoding="utf-8"))
    assert stale_snapshot["status"] == "failed"
    assert stale_snapshot["is_active"] is False
    assert stale_snapshot["message"] == "检测到陈旧任务快照，已标记为失效。"



def test_intelligent_screening_job_endpoint_reuses_running_job(tmp_path, monkeypatch) -> None:
    class SlowFakeScheduler:
        run_count = 0

        def __init__(self, settings, progress_callback=None):
            self.settings = settings
            self.progress_callback = progress_callback

        async def run_intelligent_screening(self):
            del self.settings
            type(self).run_count += 1
            if self.progress_callback:
                await self.progress_callback(
                    {
                        "status": "running",
                        "current_step": 1,
                        "total_steps": 2,
                        "step_name": "技术选股",
                        "progress_percent": 25,
                        "message": "正在执行技术选股策略...",
                        "details": {},
                    }
                )
            await asyncio.sleep(0.08)
            return {
                "success": True,
                "current_step": 2,
                "total_steps": 2,
                "screened_stocks": 5,
                "final_recommendations": 1,
                "frontlist_count": 1,
                "tracking_pool_count": 0,
                "report_id": "report-job-2",
            }

    monkeypatch.setattr("octts.api.EnhancedScreeningScheduler", SlowFakeScheduler)
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    with TestClient(app) as client:
        first_response = client.post("/screen/intelligent/jobs")
        second_response = client.post("/screen/intelligent/jobs")

        assert first_response.status_code == 202
        assert second_response.status_code == 202

        first_payload = first_response.json()
        second_payload = second_response.json()

        assert first_payload["created"] is True
        assert second_payload["created"] is False
        assert second_payload["job_id"] == first_payload["job_id"]

        for _ in range(40):
            status_response = client.get(f"/screen/intelligent/jobs/{first_payload['job_id']}")
            assert status_response.status_code == 200
            if status_response.json()["status"] == "succeeded":
                break
            time.sleep(0.02)

        assert SlowFakeScheduler.run_count == 1

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from apscheduler.schedulers.background import BackgroundScheduler

from fastapi.testclient import TestClient
from starlette.datastructures import State

from octts.api import _build_intelligent_overview_payload, _build_stock_intelligent_insight, _load_intelligent_dashboard_payload, app, get_active_intelligent_screening_job, get_intelligent_screening_job, get_settings
from octts.ui.intelligent_screening_dashboard import render_intelligent_screening_dashboard
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
from octts.services.intelligent_screening_job_manager import IntelligentScreeningJob, IntelligentScreeningJobManager
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


def test_build_intelligent_overview_payload_prefers_today_top10_rank_over_frontlist_priority() -> None:
    payload = _build_intelligent_overview_payload(
        {
            "report_context": {
                "today_top10": [
                    {"ts_code": "000001.SZ", "name": "最终第一", "recommend_rank": 1, "recommendation_score": 90, "overall_score": 82, "priority_score": 80, "source_tag": "今日Top3"},
                    {"ts_code": "688710.SH", "name": "原始分更高", "recommend_rank": 2, "recommendation_score": 86, "overall_score": 84, "priority_score": 99, "source_tag": "今日Top3"},
                ],
                "today_top3": [{"ts_code": "legacy.SZ", "name": "legacy"}],
            },
            "recommendation_pool": {
                "frontlist": [
                    {"ts_code": "688710.SH", "name": "原始分更高", "recommendation_score": 86, "priority_score": 99},
                    {"ts_code": "000001.SZ", "name": "最终第一", "recommendation_score": 90, "priority_score": 80},
                ]
            },
            "screening_results": {},
            "recommendation_summary": {"stats": {}},
        }
    )

    assert [item["ts_code"] for item in payload["top_recommendations"][:2]] == ["000001.SZ", "688710.SH"]
    assert [item["ts_code"] for item in payload["today_top3"][:2]] == ["000001.SZ", "688710.SH"]



def test_build_intelligent_overview_payload_uses_sorted_today_top10_as_authority() -> None:
    payload = _build_intelligent_overview_payload(
        {
            "report_context": {
                "today_top10": [
                    {"ts_code": "000003.SZ", "name": "C", "recommendation_score": 81, "overall_score": 79, "priority_score": 78, "source_tag": "今日候选"},
                    {"ts_code": "000001.SZ", "name": "A", "recommendation_score": 95, "overall_score": 90, "priority_score": 91, "source_tag": "今日候选"},
                    {"ts_code": "000004.SZ", "name": "D", "recommendation_score": 70, "overall_score": 75, "priority_score": 74, "source_tag": "今日候选"},
                    {"ts_code": "000002.SZ", "name": "B", "recommendation_score": 88, "overall_score": 86, "priority_score": 87, "source_tag": "今日候选"},
                ],
                "today_top3": [
                    {"ts_code": "999999.SZ", "name": "legacy"},
                ],
            },
            "recommendation_pool": {
                "frontlist": [
                    {"ts_code": "888888.SZ", "name": "fallback", "recommendation_score": 99, "priority_score": 99}
                ]
            },
            "screening_results": {},
            "recommendation_summary": {"stats": {}},
        }
    )

    codes = [item["ts_code"] for item in payload["top_recommendations"]]
    assert codes == ["000003.SZ", "000001.SZ", "000004.SZ", "000002.SZ"]
    assert [item["ts_code"] for item in payload["today_top10"]] == codes
    assert [item["ts_code"] for item in payload["today_top3"]] == codes[:3]



def test_build_intelligent_overview_payload_uses_today_top3_when_today_top10_missing() -> None:
    payload = _build_intelligent_overview_payload(
        {
            "report_context": {
                "today_top3": [
                    {"ts_code": "300692.SZ", "name": "真实一", "recommendation_score": 54.01, "overall_score": 82.31, "priority_score": 82.31, "source_tag": "今日Top3"},
                    {"ts_code": "600613.SH", "name": "真实二", "recommendation_score": 48.68, "overall_score": 70.6, "priority_score": 70.6, "source_tag": "今日Top3"},
                    {"ts_code": "300086.SZ", "name": "真实三", "recommendation_score": 47.61, "overall_score": 74.54, "priority_score": 74.54, "source_tag": "今日Top3"},
                ],
            },
            "recommendation_pool": {
                "frontlist": [
                    {"ts_code": "002269.SZ", "name": "空壳一", "recommendation_score": 76.56, "priority_score": 0.0, "source_tag": "今日Top3"},
                    {"ts_code": "301226.SZ", "name": "空壳二", "recommendation_score": 76.04, "priority_score": 0.0, "source_tag": "今日Top3"},
                    {"ts_code": "600222.SH", "name": "空壳三", "recommendation_score": 75.36, "priority_score": 0.0, "source_tag": "今日Top3"},
                ],
            },
            "screening_results": {},
            "recommendation_summary": {"stats": {}},
        }
    )

    assert [item["ts_code"] for item in payload["today_top3"]] == ["300692.SZ", "600613.SH", "300086.SZ"]
    assert [item["ts_code"] for item in payload["top_recommendations"][:3]] == ["300692.SZ", "600613.SH", "300086.SZ"]



def test_build_stock_intelligent_insight_prefers_focus_assessment_for_contradiction_case() -> None:
    insight = _build_stock_intelligent_insight(
        "301157.SZ",
        {
            "report_context": {
                "today_top3": [{
                    "ts_code": "301157.SZ",
                    "recommendation_score": 77.6,
                    "overall_score": 77.6,
                    "recommendation_text": "谨慎观察，暂不建议操作",
                    "technical_signal": "多头趋势",
                    "short_term_contradiction_penalty": 6.0,
                    "action_plan": {"action_bias": "观察"},
                }]
            },
            "intelligent_report": {
                "blocks": {
                    "focus_stocks": [{
                        "ts_code": "301157.SZ",
                        "overall_assessment": "更适合边走边看，不宜作为Top3主打。",
                        "core_highlights": ["高位分歧增加"],
                        "action_plan": {"action_bias": "观察", "entry_zone": "63.50附近观察"},
                    }],
                    "yesterday_reviews": [],
                }
            },
            "recommendation_pool": {"frontlist": [], "today_top": [], "yesterday_continuations": []},
        },
    )

    assert insight["in_today_top3"] is True
    assert insight["overall_assessment"] == "更适合边走边看，不宜作为Top3主打。"
    assert insight["action_plan"]["action_bias"] == "观察"


def test_overview_reason_text_avoids_score_dump(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json"),
    )
    monkeypatch.setattr(
        "octts.api._build_recommendation_dashboard_payload",
        lambda settings: {
            "generated_at": "2026-03-30T09:15:00",
            "screening_results": {"strategy_count": 1, "total_stocks": 3, "final_recommendations": 3},
            "recommendation_pool": {
                "frontlist": [],
                "yesterday_continuations": [],
                "report_context": {
                    "today_top10": [
                        {
                            "ts_code": "002644.SZ",
                            "name": "佛慈制药",
                            "recommendation_score": 68.9,
                            "overall_score": 68.9,
                            "priority_score": 69.2,
                            "technical_signal": "有企稳迹象",
                            "market_context_view": "中药方向跟风修复",
                            "distribution_risk_flags": ["资金承接仍偏弱"],
                            "recommendation_text": "谨慎跟踪，等待多维度共振。",
                            "source_tag": "今日候选",
                            "action_plan": {"entry_zone": "等待触发", "stop_loss": "待观察", "take_profit": "待观察", "holding_horizon": "1-5个交易日", "invalid_condition": "走势失真时离场"},
                        }
                    ]
                },
            },
            "ai_analyses": {},
            "news_clusters": [],
            "intelligent_report": {"title": "智能选股报告", "summary": "这里是摘要", "blocks": {"focus_stocks": [], "yesterday_reviews": [], "comparison": {}, "overall_action": {}}},
            "recommendation_summary": {"stats": {}},
            "recommendation_methodology": {},
        },
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening")

    assert response.status_code == 200
    assert "佛慈制药" in response.text
    assert "谨慎观察；有企稳迹象；资金承接仍偏弱；先看承接" in response.text
    assert "综合评分 68.9" not in response.text
    assert "主评分 69.2" not in response.text
    assert "情绪面 +0.0" not in response.text
    assert "新闻面 -0.3" not in response.text



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
    monkeypatch.setattr(
        "octts.api_routes.dashboard_routes.get_settings",
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


def test_stock_detail_data_includes_analysis_context_metrics(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    record = _build_record()
    record.snapshot.close = 12.0
    record.snapshot.high = 12.4
    record.snapshot.low = 11.8
    record.snapshot.vol_ratio = 1.35
    record.snapshot.turnover_rate = 2.4
    record.snapshot.moneyflow_summary = {
        "net_mf_amount": 1200.0,
        "buy_lg_amount": 900.0,
        "sell_lg_amount": 500.0,
        "buy_elg_amount": 700.0,
        "sell_elg_amount": 250.0,
    }
    record.snapshot.daily_summary = [
        {
            "trade_date": (datetime(2026, 2, 18) + timedelta(days=offset)).strftime("%Y%m%d"),
            "open": 10.0,
            "high": 10.5 + offset * 0.01,
            "low": 9.5 + offset * 0.01,
            "close": 10.0,
        }
        for offset in range(19)
    ]
    store = FileHistoryStore(str(history_path))
    store.append(record)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )
    monkeypatch.setattr(
        "octts.api_routes.dashboard_routes.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_POSITION_FILE_PATH=str(tmp_path / "positions.json"),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/stocks/600000.SH/data")

    assert response.status_code == 200
    analysis_context = response.json()["symbol"]["analysis_context"]
    assert analysis_context["technical"]["ma20"] == 10.1
    assert round(analysis_context["technical"]["distance_to_ma20_pct"], 2) == 18.81
    assert analysis_context["technical"]["vol_ratio"] == 1.35
    assert analysis_context["technical"]["turnover_rate"] == 2.4
    assert analysis_context["technical"]["recent_5d_low"] == 9.65
    assert analysis_context["technical"]["recent_5d_high"] == 12.4
    assert analysis_context["moneyflow"]["net_mf_amount"] == 1200.0
    assert analysis_context["moneyflow"]["buy_lg_amount"] == 900.0
    assert analysis_context["moneyflow"]["sell_elg_amount"] == 250.0


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
        "octts.api_routes.dashboard_routes.get_settings",
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
                "today_top3_live_context": [{"ts_code": "600000.SH", "name": "浦发银行", "query": "浦发银行", "items": [{"title": "浦发银行披露一季报预告", "summary": "业绩边际改善", "source": "东方财富", "url": "https://example.com/live1", "publish_time": "2026-03-30 09:35", "category": "公告"}]}],
                "yesterday_top3_live_context": [],
            },
            "intelligent_report": {
                "blocks": {
                    "focus_stocks": [{
                        "ts_code": "600000.SH",
                        "core_highlights": ["趋势延续"],
                        "risk_warnings": ["量能待确认"],
                        "overall_assessment": "适合继续跟踪",
                        "focus_analysis": "这里是重点分析",
                        "market_context_view": "市场环境向好",
                        "market_performance_view": "股价表现稳定",
                        "catalyst_and_capital_view": "资金承接改善",
                        "fundamental_view": "基本面稳健",
                        "trading_context_view": "节奏以观察低吸为主",
                        "action_plan": {"entry_zone": "10-10.2", "take_profit": "10.8", "stop_loss": "9.8", "invalid_condition": "跌破支撑"},
                    }],
                    "yesterday_reviews": [{"ts_code": "600000.SH", "today_verdict": "延续", "status": "延续"}],
                }
            },
        },
    )
    monkeypatch.setattr(
        "octts.api_routes.dashboard_routes.load_intelligent_dashboard_payload",
        lambda settings, trade_date=None: {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "600000.SH", "source_tag": "今日Top3", "recommendation_score": 88, "priority_score": 77, "ai_confidence": 0.8}],
                "today_top": [{"ts_code": "600000.SH", "action_plan": {"entry_zone": "10-10.2", "take_profit": "10.8", "stop_loss": "9.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破支撑"}}],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600000.SH", "recommendation_score": 88, "overall_score": 77, "display_confidence": 0.8, "recommendation_text": "继续观察", "technical_signal": "量价共振", "action_plan": {"action_bias": "观察"}}],
                "yesterday_top3_review": [{"ts_code": "600000.SH", "today_verdict": "延续", "previous_recommendation_score": 82}],
                "today_top3_live_context": [{"ts_code": "600000.SH", "name": "浦发银行", "query": "浦发银行", "items": [{"title": "浦发银行披露一季报预告", "summary": "业绩边际改善", "source": "东方财富", "url": "https://example.com/live1", "publish_time": "2026-03-30 09:35", "category": "公告"}]}],
                "yesterday_top3_live_context": [],
            },
            "intelligent_report": {
                "blocks": {
                    "focus_stocks": [{
                        "ts_code": "600000.SH",
                        "core_highlights": ["趋势延续"],
                        "risk_warnings": ["量能待确认"],
                        "overall_assessment": "适合继续跟踪",
                        "focus_analysis": "这里是重点分析",
                        "market_context_view": "市场环境向好",
                        "market_performance_view": "股价表现稳定",
                        "catalyst_and_capital_view": "资金承接改善",
                        "fundamental_view": "基本面稳健",
                        "trading_context_view": "节奏以观察低吸为主",
                        "action_plan": {"entry_zone": "10-10.2", "take_profit": "10.8", "stop_loss": "9.8", "invalid_condition": "跌破支撑"},
                    }],
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
    assert insight["focus_analysis"] == "这里是重点分析"
    assert insight["market_context_view"] == "市场环境向好"
    assert insight["market_performance_view"] == "股价表现稳定"
    assert insight["catalyst_and_capital_view"] == "资金承接改善"
    assert insight["fundamental_view"] == "基本面稳健"
    assert insight["trading_context_view"] == "节奏以观察低吸为主"
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
                "frontlist": [
                    {"ts_code": "000009.SZ", "name": "后备股", "recommendation_score": 99, "priority_score": 99, "source_tag": "今日Top3"}
                ],
                "today_top": [{"ts_code": "000009.SZ", "name": "后备股", "recommendation_score": 99, "priority_score": 99, "source_tag": "今日Top3"}],
                "yesterday_continuations": [{"ts_code": "000002.SZ", "name": "万科A", "recommendation_score": 72, "priority_score": 80, "source_tag": "昨日延续", "technical_signal": "延续观察", "recommendation_text": "继续观察", "ai_confidence": 0.61, "action_plan": {"take_profit": "反抽 8.8 附近止盈", "stop_loss": "跌破 8.1 离场", "invalid_condition": "全天弱于地产板块", "holding_horizon": "2-3个交易日"}}],
                "report_context": {
                    "today_top10": [
                        {"ts_code": "000001.SZ", "name": "平安银行", "recommendation_score": 88, "priority_score": 92, "overall_score": 90, "source_tag": "今日候选", "technical_signal": "多头趋势", "recommendation_text": "建议跟踪", "ai_confidence": 0.82, "action_plan": {"entry_zone": "10.1-10.3", "stop_loss": "9.8", "take_profit": "10.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破 9.8"}},
                        {"ts_code": "000003.SZ", "name": "招商银行", "recommendation_score": 84, "priority_score": 85, "overall_score": 84, "source_tag": "今日候选"},
                        {"ts_code": "000004.SZ", "name": "宁波银行", "recommendation_score": 80, "priority_score": 82, "overall_score": 81, "source_tag": "今日候选"},
                        {"ts_code": "000005.SZ", "name": "兴业银行", "recommendation_score": 78, "priority_score": 79, "overall_score": 78, "source_tag": "今日候选"},
                    ],
                    "today_top3": [{"ts_code": "999999.SZ", "name": "旧Top3"}],
                },
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
    assert "重点个股" in response.text
    assert "今日新闻" in response.text
    assert "买入区间" in response.text
    assert "止损位" in response.text
    assert "第一止盈位" in response.text
    assert "失效条件" in response.text
    assert "转弱止损/离场" in response.text
    assert "离场触发条件" in response.text
    assert "后备股" not in response.text
    assert "今日候选" not in response.text
    assert 'href="/intelligent-screening?tab=recommendations"' not in response.text



def test_focus_tab_uses_focus_title_and_hides_score_breakdown(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json"),
    )
    monkeypatch.setattr(
        "octts.api._build_recommendation_dashboard_payload",
        lambda settings: {
            "generated_at": "2026-03-30T09:15:00",
            "screening_results": {"strategy_count": 1, "total_stocks": 3, "final_recommendations": 1},
            "recommendation_pool": {"frontlist": [], "yesterday_continuations": [], "report_context": {"today_top10": []}},
            "ai_analyses": {},
            "news_clusters": [],
            "intelligent_report": {
                "title": "智能选股报告",
                "summary": "这里是摘要",
                "blocks": {
                    "focus_stocks": [
                        {
                            "ts_code": "301157.SZ",
                            "name": "华塑科技",
                            "recommendation_score": 77.6,
                            "overall_score": 77.6,
                            "display_confidence": 0.80,
                            "source_tag": "今日Top3",
                            "focus_analysis": "日内先冲高后回落，说明高位分歧开始增大。业务上仍有电池安全管理方向支撑，但短线更要看资金回流和板块共振是否继续。若后续承接转弱，回撤压力会明显上升。",
                            "market_performance_view": "今日收于63.50元，涨跌幅+3.83%。盘中冲高后震荡回落，短线开始出现分歧。",
                            "catalyst_and_capital_view": "资金面上，当日主力资金-1662，近3日主力资金-3504，说明追价资金并不坚决。",
                            "fundamental_view": "公司主营聚焦电池安全管理系统，覆盖后备电池、动力铅蓄电池与储能锂电BMS。",
                            "trading_context_view": "前期拉升后进入高波动阶段，短线更看承接而不是单日涨幅。",
                            "market_context_view": "板块资金风格偏中性，联动性一般。",
                            "core_highlights": ["进入重点跟踪名单"],
                            "risk_warnings": ["高位分歧加大后，回撤可能放大"],
                            "overall_assessment": "更适合边走边看，不宜只凭分数追高。",
                            "action_plan": {"action_bias": "观察", "entry_zone": "63.50附近观察", "take_profit": "66.67附近分批止盈", "stop_loss": "61.59", "invalid_condition": "量价结构走弱"},
                        }
                    ],
                    "yesterday_reviews": [],
                    "comparison": {},
                    "overall_action": {},
                },
            },
            "recommendation_summary": {"stats": {}},
            "recommendation_methodology": {},
        },
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening?tab=focus")

    assert response.status_code == 200
    assert "<h2 style=\"margin-bottom: 24px;\">重点个股</h2>" in response.text
    assert "评分拆解" not in response.text
    assert "当前排序主要由推荐分" not in response.text
    assert "命中策略数为" not in response.text
    assert "日内先冲高后回落，说明高位分歧开始增大" in response.text
    assert "公司主营聚焦电池安全管理系统" in response.text
    assert "高位分歧加大后，回撤可能放大" in response.text



def test_review_analysis_prefers_recent_market_dynamics_over_score_dump(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json"),
    )
    monkeypatch.setattr(
        "octts.api._build_recommendation_dashboard_payload",
        lambda settings: {
            "generated_at": "2026-03-30T09:15:00",
            "screening_results": {"strategy_count": 1, "total_stocks": 3, "final_recommendations": 1},
            "recommendation_pool": {
                "frontlist": [],
                "yesterday_continuations": [
                    {
                        "ts_code": "300585.SZ",
                        "name": "奥联电子",
                        "status": "延续",
                        "today_verdict": "延续跟踪，但已降至昨日延续区",
                        "yesterday_conclusion": "昨日结论缺失",
                        "analysis": "高位回落后仍未完全走坏，但强度相比昨日已明显下降。",
                        "pct_change": 1.85,
                        "turnover_rate": 18.4,
                        "volume_ratio": 1.36,
                        "amplitude": 7.2,
                        "recent_runup_5d": 12.6,
                        "action_plan": {"action_bias": "观察", "entry_zone": "16.50附近观察", "take_profit": "17.32附近分批止盈", "stop_loss": "16.00", "invalid_condition": "趋势改善"},
                    }
                ],
                "report_context": {"today_top10": []},
            },
            "ai_analyses": {},
            "news_clusters": [],
            "intelligent_report": {
                "title": "智能选股报告",
                "summary": "这里是摘要",
                "blocks": {
                    "focus_stocks": [],
                    "yesterday_reviews": [
                        {
                            "ts_code": "300585.SZ",
                            "name": "奥联电子",
                            "status": "延续",
                            "today_verdict": "延续跟踪，但已降至昨日延续区",
                            "yesterday_conclusion": "昨日结论缺失",
                            "analysis": "高位回落后仍未完全走坏，但强度相比昨日已明显下降。",
                            "strength_change": "当前状态为“延续”，推荐分较昨日回落23.0，主要变化来自奥联电子（300585.SZ）综合评分 70.9，主评分 71.1 由技术面与基本面构成。",
                            "strength_change_fallback": "推荐分较昨日明显回落，说明强度已从昨日Top3降级到延续跟踪区，后续重点看承接是否继续转弱。",
                            "review_analysis": "昨日逻辑是“昨日结论缺失”，今天的复盘结论是“延续跟踪，但已降至昨日延续区”。从结果看，奥联电子（300585.SZ）综合评分 70.9，主评分 71.1 由技术面与基本面构成，情绪面 +0.1、新闻面 -0.3 作为辅助修正。",
                            "action_plan": {"action_bias": "观察", "entry_zone": "16.50附近观察", "take_profit": "17.32附近分批止盈", "stop_loss": "16.00", "invalid_condition": "趋势改善"},
                        }
                    ],
                    "comparison": {},
                    "overall_action": {},
                },
            },
            "recommendation_summary": {"stats": {}},
            "recommendation_methodology": {},
        },
    )

    client = TestClient(app)
    response = client.get("/intelligent-screening?tab=focus")

    assert response.status_code == 200
    assert "300585.SZ" in response.text
    assert "昨日结论缺失" in response.text
    assert "推荐分较昨日回落23.0" not in response.text
    assert "综合评分 70.9" not in response.text
    assert "主评分 71.1" not in response.text



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


def test_build_stock_intelligent_insight_includes_continuation_fields() -> None:
    insight = _build_stock_intelligent_insight(
        "600000.SH",
        {
            "report_context": {
                "today_top3": [{
                    "ts_code": "600000.SH",
                    "recommendation_score": 88,
                    "overall_score": 77,
                    "continuation_bias_score": 2.6,
                    "continuation_positive_flags": ["3日资金承接偏强", "昨日Top3延续未走坏"],
                    "continuation_negative_flags": ["近5日涨幅偏大"],
                }]
            },
            "recommendation_pool": {"frontlist": [], "today_top": [], "yesterday_continuations": []},
            "intelligent_report": {"blocks": {"focus_stocks": [], "yesterday_reviews": []}},
        },
    )

    assert insight["continuation_bias_score"] == 2.6
    assert insight["continuation_positive_flags"] == ["3日资金承接偏强", "昨日Top3延续未走坏"]
    assert insight["continuation_negative_flags"] == ["近5日涨幅偏大"]


def test_build_stock_intelligent_insight_exposes_focus_view_fields() -> None:
    insight = _build_stock_intelligent_insight(
        "600613.SH",
        {
            "report_context": {
                "today_top3": [{
                    "ts_code": "600613.SH",
                    "recommendation_score": 28.41,
                    "overall_score": 70.6,
                }]
            },
            "recommendation_pool": {"frontlist": [], "today_top": [], "yesterday_continuations": []},
            "intelligent_report": {
                "blocks": {
                    "focus_stocks": [{
                        "ts_code": "600613.SH",
                        "score_rationale": "排序逻辑解释",
                        "fundamental_view": "基本面说明",
                        "market_context_view": "市场环境说明",
                        "trading_context_view": "交易节奏说明",
                        "market_performance_view": "市场表现说明",
                        "catalyst_and_capital_view": "催化与资金说明",
                        "focus_analysis": "完整重点分析",
                        "core_highlights": ["亮点一"],
                        "risk_warnings": ["风险一"],
                        "overall_assessment": "综合评价",
                    }],
                    "yesterday_reviews": [],
                }
            },
        },
    )

    assert insight["score_rationale"] == "排序逻辑解释"
    assert insight["fundamental_view"] == "基本面说明"
    assert insight["market_context_view"] == "市场环境说明"
    assert insight["trading_context_view"] == "交易节奏说明"
    assert insight["market_performance_view"] == "市场表现说明"
    assert insight["catalyst_and_capital_view"] == "催化与资金说明"
    assert insight["focus_analysis"] == "完整重点分析"
    assert insight["core_highlights"] == ["亮点一"]
    assert insight["risk_warnings"] == ["风险一"]


def test_build_intelligent_overview_payload_prefers_authoritative_today_top3_over_legacy_frontlist() -> None:
    payload = _build_intelligent_overview_payload(
        {
            "report_context": {
                "today_top10": [
                    {"ts_code": "300692.SZ", "name": "真实一", "recommend_rank": 3, "recommendation_score": 88, "overall_score": 81, "priority_score": 81, "source_tag": "今日Top3"},
                    {"ts_code": "600613.SH", "name": "真实二", "recommend_rank": 4, "recommendation_score": 87, "overall_score": 80, "priority_score": 80, "source_tag": "今日Top3"},
                    {"ts_code": "300086.SZ", "name": "真实三", "recommend_rank": 5, "recommendation_score": 86, "overall_score": 79, "priority_score": 79, "source_tag": "今日Top3"},
                ],
                "today_top3": [{"ts_code": "legacy.SZ", "name": "旧Top3"}],
            },
            "recommendation_pool": {
                "frontlist": [
                    {"ts_code": "002269.SZ", "name": "空壳", "recommendation_score": 99, "priority_score": 0, "source_tag": "今日Top3"},
                    {"ts_code": "300692.SZ", "name": "真实一", "recommendation_score": 88, "priority_score": 81, "source_tag": "今日Top3"},
                ],
                "today_top": [{"ts_code": "002269.SZ", "name": "空壳", "source_tag": "今日Top3"}],
            },
            "screening_results": {},
            "recommendation_summary": {"stats": {}},
        }
    )

    assert [item["ts_code"] for item in payload["today_top3"]] == ["300692.SZ", "600613.SH", "300086.SZ"]
    assert [item["ts_code"] for item in payload["top_recommendations"][:3]] == ["300692.SZ", "600613.SH", "300086.SZ"]


def test_build_intelligent_overview_payload_derives_today_top3_from_final_today_top10() -> None:
    payload = _build_intelligent_overview_payload(
        {
            "report_context": {
                "today_top10": [
                    {"ts_code": "000001.SZ", "name": "A", "recommendation_score": 91, "overall_score": 80, "priority_score": 80},
                    {"ts_code": "000002.SZ", "name": "B", "recommendation_score": 87, "overall_score": 79, "priority_score": 79},
                    {"ts_code": "000003.SZ", "name": "C", "recommendation_score": 85, "overall_score": 78, "priority_score": 78},
                    {"ts_code": "000004.SZ", "name": "D", "recommendation_score": 83, "overall_score": 77, "priority_score": 77},
                ],
                "today_top3": [
                    {"ts_code": "legacy-1.SZ", "name": "旧一"},
                    {"ts_code": "legacy-2.SZ", "name": "旧二"},
                    {"ts_code": "legacy-3.SZ", "name": "旧三"},
                ],
            },
            "recommendation_pool": {"frontlist": [{"ts_code": "front.SZ", "name": "前排", "recommendation_score": 99}]},
            "screening_results": {},
            "recommendation_summary": {"stats": {}},
        }
    )

    assert [item["ts_code"] for item in payload["today_top10"]] == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    assert [item["ts_code"] for item in payload["today_top3"]] == ["000001.SZ", "000002.SZ", "000003.SZ"]


def test_build_intelligent_overview_payload_keeps_today_top3_authoritative_when_top10_has_candidates_first() -> None:
    payload = _build_intelligent_overview_payload(
        {
            "report_context": {
                "today_top3": [
                    {"ts_code": "002107.SZ", "name": "沃华医药", "recommendation_score": 55.8, "overall_score": 83.1, "source_tag": "今日Top3"},
                    {"ts_code": "688010.SH", "name": "福光股份", "recommendation_score": 49.2, "overall_score": 66.9, "source_tag": "今日Top3"},
                    {"ts_code": "600272.SH", "name": "开开实业", "recommendation_score": 42.1, "overall_score": 65.6, "source_tag": "今日Top3"},
                ],
                "today_top10": [
                    {"ts_code": "603182.SH", "name": "嘉华股份", "recommendation_score": 76.56, "overall_score": None, "source_tag": "今日候选"},
                    {"ts_code": "688618.SH", "name": "三旺通信", "recommendation_score": 74.76, "overall_score": None, "source_tag": "今日候选"},
                    {"ts_code": "002107.SZ", "name": "沃华医药", "recommendation_score": 55.8, "overall_score": 83.1, "source_tag": "今日Top3"},
                    {"ts_code": "688010.SH", "name": "福光股份", "recommendation_score": 49.2, "overall_score": 66.9, "source_tag": "今日Top3"},
                    {"ts_code": "600272.SH", "name": "开开实业", "recommendation_score": 42.1, "overall_score": 65.6, "source_tag": "今日Top3"},
                ],
            },
            "recommendation_pool": {"frontlist": []},
        }
    )

    assert [item["ts_code"] for item in payload["today_top3"]] == ["002107.SZ", "688010.SH", "600272.SH"]
    assert [item["ts_code"] for item in payload["today_top10"][:2]] == ["603182.SH", "688618.SH"]


def _build_dashboard_payload_for_order_tests() -> dict:
    return {
        "screening_results": {},
        "recommendation_pool": {
            "frontlist": [
                {"ts_code": "FRONT.SZ", "name": "前排干扰", "recommendation_score": 99, "priority_score": 99},
            ],
            "yesterday_continuations": [],
        },
        "ai_analyses": {},
        "news_clusters": [],
        "intelligent_report": {
            "blocks": {
                "focus_stocks": [
                    {"ts_code": "000001.SZ", "focus_analysis": "甲分析", "overall_assessment": "甲总结"},
                    {"ts_code": "000002.SZ", "focus_analysis": "乙分析", "overall_assessment": "乙总结"},
                    {"ts_code": "000003.SZ", "focus_analysis": "丙分析", "overall_assessment": "丙总结"},
                ],
                "yesterday_reviews": [],
                "comparison": {},
                "overall_action": {},
            }
        },
        "recommendation_summary": {},
        "recommendation_methodology": {},
        "report_context": {
            "today_top3": [
                {"ts_code": "000001.SZ", "name": "甲", "recommendation_score": 91, "overall_score": 81, "recommend_rank": 1, "source_tag": "今日Top3"},
                {"ts_code": "000002.SZ", "name": "乙", "recommendation_score": 89, "overall_score": 80, "recommend_rank": 2, "source_tag": "今日Top3"},
                {"ts_code": "000003.SZ", "name": "丙", "recommendation_score": 87, "overall_score": 79, "recommend_rank": 3, "source_tag": "今日Top3"},
            ],
            "today_top10": [
                {"ts_code": "000001.SZ", "name": "甲", "recommendation_score": 91, "overall_score": 81, "recommend_rank": 1, "source_tag": "今日Top3"},
                {"ts_code": "000002.SZ", "name": "乙", "recommendation_score": 89, "overall_score": 80, "recommend_rank": 2, "source_tag": "今日Top3"},
                {"ts_code": "000003.SZ", "name": "丙", "recommendation_score": 87, "overall_score": 79, "recommend_rank": 3, "source_tag": "今日Top3"},
                {"ts_code": "000004.SZ", "name": "丁", "recommendation_score": 85, "overall_score": 78, "recommend_rank": 4, "source_tag": "今日候选"},
            ],
            "yesterday_top3_review": [
                {"ts_code": "000777.SZ", "name": "复盘股", "status": "观察", "today_verdict": "继续复盘", "review_analysis": "完整复盘内容", "analysis": "完整复盘内容"},
            ],
        },
    }


def test_render_intelligent_screening_dashboard_keeps_overview_top3_order_and_yesterday_review_visibility() -> None:
    payload = _build_dashboard_payload_for_order_tests()
    html = render_intelligent_screening_dashboard(**payload, active_tab="overview")

    today_top3_pos = [html.index(code) for code in ["000001.SZ", "000002.SZ", "000003.SZ"]]
    assert today_top3_pos == sorted(today_top3_pos)
    assert "000777.SZ" in html
    assert "完整复盘内容" in html


def test_render_intelligent_screening_dashboard_keeps_focus_tab_top3_order() -> None:
    payload = _build_dashboard_payload_for_order_tests()
    html = render_intelligent_screening_dashboard(**payload, active_tab="focus")

    focus_pos = [html.index(code) for code in ["000001.SZ", "000002.SZ", "000003.SZ"]]
    assert focus_pos == sorted(focus_pos)


def test_render_intelligent_screening_dashboard_keeps_focus_report_only_entries_visible() -> None:
    payload = _build_dashboard_payload_for_order_tests()
    payload["report_context"]["today_top3"] = []
    html = render_intelligent_screening_dashboard(**payload, active_tab="focus")

    assert "甲分析" in html
    assert "乙分析" in html
    assert "丙分析" in html


def test_render_intelligent_screening_dashboard_prefers_authoritative_top3_and_score_fallback() -> None:
    payload = _build_dashboard_payload_for_order_tests()
    payload["recommendation_pool"]["frontlist"] = [
        {"ts_code": "FRONT1.SZ", "name": "前排干扰一", "recommendation_score": 99, "priority_score": 99},
        {"ts_code": "FRONT2.SZ", "name": "前排干扰二", "recommendation_score": 98, "priority_score": 98},
        {"ts_code": "FRONT3.SZ", "name": "前排干扰三", "recommendation_score": 97, "priority_score": 97},
    ]
    payload["report_context"]["today_top10"][0]["final_display_recommendation_score"] = None

    focus_html = render_intelligent_screening_dashboard(**payload, active_tab="focus")

    assert "FRONT1.SZ" not in focus_html
    assert "甲分析" in focus_html
    focus_pos = [focus_html.index(code) for code in ["000001.SZ", "000002.SZ", "000003.SZ"]]
    assert focus_pos == sorted(focus_pos)


def test_render_intelligent_screening_dashboard_preserves_report_focus_analysis_when_top10_has_candidates_first() -> None:
    payload = _build_dashboard_payload_for_order_tests()
    payload["report_context"]["today_top3"] = [
        {"ts_code": "002107.SZ", "name": "沃华医药", "recommendation_score": 55.8, "overall_score": 83.1, "source_tag": "今日Top3"},
        {"ts_code": "688010.SH", "name": "福光股份", "recommendation_score": 49.2, "overall_score": 66.9, "source_tag": "今日Top3"},
        {"ts_code": "600272.SH", "name": "开开实业", "recommendation_score": 42.1, "overall_score": 65.6, "source_tag": "今日Top3"},
    ]
    payload["report_context"]["today_top10"] = [
        {"ts_code": "603182.SH", "name": "嘉华股份", "recommendation_score": 76.56, "source_tag": "今日候选"},
        {"ts_code": "688618.SH", "name": "三旺通信", "recommendation_score": 74.76, "source_tag": "今日候选"},
        {"ts_code": "002107.SZ", "name": "沃华医药", "recommendation_score": 55.8, "overall_score": 83.1, "source_tag": "今日Top3"},
        {"ts_code": "688010.SH", "name": "福光股份", "recommendation_score": 49.2, "overall_score": 66.9, "source_tag": "今日Top3"},
        {"ts_code": "600272.SH", "name": "开开实业", "recommendation_score": 42.1, "overall_score": 65.6, "source_tag": "今日Top3"},
    ]
    payload["intelligent_report"]["blocks"]["focus_stocks"] = [
        {"ts_code": "002107.SZ", "focus_analysis": "沃华深度分析", "overall_assessment": "沃华总结"},
        {"ts_code": "688010.SH", "focus_analysis": "福光深度分析", "overall_assessment": "福光总结"},
        {"ts_code": "600272.SH", "focus_analysis": "开开深度分析", "overall_assessment": "开开总结"},
    ]

    html = render_intelligent_screening_dashboard(**payload, active_tab="focus")

    assert "沃华深度分析" in html
    assert "福光深度分析" in html
    assert "开开深度分析" in html
    assert "603182.SH" not in html
    assert "688618.SH" not in html


def test_render_intelligent_screening_dashboard_invalid_recommendations_tab_falls_back_to_overview() -> None:
    payload = _build_dashboard_payload_for_order_tests()
    html = render_intelligent_screening_dashboard(**payload, active_tab="recommendations")

    assert "今日 Top3" in html
    assert "昨日 Top3 今日复盘 / 昨日延续" in html


def test_build_stock_intelligent_insight_marks_only_authoritative_today_top3_as_top3() -> None:
    insight = _build_stock_intelligent_insight(
        "002269.SZ",
        {
            "report_context": {
                "today_top3": [{"ts_code": "300692.SZ", "recommendation_score": 88, "overall_score": 81}],
            },
            "recommendation_pool": {
                "frontlist": [{"ts_code": "002269.SZ", "source_tag": "今日Top3", "recommendation_score": 99, "priority_score": 0}],
                "today_top": [{"ts_code": "002269.SZ", "source_tag": "今日Top3"}],
                "yesterday_continuations": [],
            },
            "intelligent_report": {"blocks": {"focus_stocks": [{"ts_code": "300692.SZ", "overall_assessment": "真实分析"}], "yesterday_reviews": []}},
        },
    )

    assert insight["in_today_top3"] is False
    assert insight["overall_assessment"] is None


def test_build_stock_intelligent_insight_prefers_display_confidence() -> None:
    insight = _build_stock_intelligent_insight(
        "600000.SH",
        {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "600000.SH", "source_tag": "今日Top3", "recommendation_score": 88, "priority_score": 77, "ai_confidence": 0.45}],
                "today_top": [{"ts_code": "600000.SH", "ai_confidence": 0.52}],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600000.SH", "recommendation_score": 88, "overall_score": 77, "display_confidence": 0.8, "overall_confidence": 0.66}],
            },
            "intelligent_report": {"blocks": {}},
        },
    )

    assert insight["display_confidence"] == 0.8
    assert insight["overall_confidence"] == 0.8
    assert insight["ai_confidence"] == 0.45
    assert insight["confidence"] == 0.8


def test_build_stock_intelligent_insight_falls_back_to_overall_confidence() -> None:
    insight = _build_stock_intelligent_insight(
        "600001.SH",
        {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "600001.SH", "source_tag": "今日Top3", "recommendation_score": 81, "priority_score": 75, "ai_confidence": 0.41}],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600001.SH", "recommendation_score": 81, "overall_score": 75, "overall_confidence": 0.73}],
            },
            "intelligent_report": {"blocks": {}},
        },
    )

    assert insight["display_confidence"] == 0.73
    assert insight["overall_confidence"] == 0.73
    assert insight["ai_confidence"] == 0.41
    assert insight["confidence"] == 0.73


def test_build_stock_intelligent_insight_falls_back_to_ai_confidence() -> None:
    insight = _build_stock_intelligent_insight(
        "600002.SH",
        {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "600002.SH", "source_tag": "今日候选", "recommendation_score": 76, "priority_score": 71, "ai_confidence": 0.62}],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600002.SH", "recommendation_score": 76, "overall_score": 71}],
            },
            "intelligent_report": {"blocks": {}},
        },
    )

    assert insight["display_confidence"] is None
    assert insight["overall_confidence"] == 0.62
    assert insight["ai_confidence"] == 0.62
    assert insight["confidence"] == 0.62


def test_build_stock_intelligent_insight_keeps_missing_confidence_as_none() -> None:
    insight = _build_stock_intelligent_insight(
        "600003.SH",
        {
            "recommendation_pool": {
                "frontlist": [{"ts_code": "600003.SH", "source_tag": "今日候选", "recommendation_score": 70, "priority_score": 68}],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600003.SH", "recommendation_score": 70, "overall_score": 68}],
            },
            "intelligent_report": {"blocks": {}},
        },
    )

    assert insight["display_confidence"] is None
    assert insight["overall_confidence"] is None
    assert insight["ai_confidence"] is None
    assert insight["confidence"] is None


def test_build_stock_intelligent_insight_preserves_zero_value_fields() -> None:
    insight = _build_stock_intelligent_insight(
        "600004.SH",
        {
            "recommendation_pool": {
                "frontlist": [
                    {
                        "ts_code": "600004.SH",
                        "source_tag": "今日候选",
                        "recommendation_score": 61,
                        "priority_score": 67,
                        "distribution_risk_score": 0.0,
                        "moneyflow_3d_value": 0.0,
                        "turnover_spike_ratio": 0.0,
                        "recent_runup_5d": 0.0,
                        "continuation_bias_score": 0.0,
                        "strategy_count": 0,
                    }
                ],
            },
            "report_context": {
                "today_top3": [{"ts_code": "600004.SH", "recommendation_score": 61, "overall_score": 67}],
            },
            "intelligent_report": {"blocks": {}},
        },
    )

    assert insight["strategy_count"] == 0
    assert insight["continuation_bias_score"] == 0.0
    assert insight["distribution_risk_score"] == 0.0
    assert insight["moneyflow_3d_value"] == 0.0
    assert insight["turnover_spike_ratio"] == 0.0
    assert insight["recent_runup_5d"] == 0.0


def test_load_intelligent_dashboard_payload_normalizes_legacy_recommendation_text_and_action_bias(tmp_path, monkeypatch) -> None:
    history_dir = tmp_path / "history"
    snapshot_dir = history_dir / "intelligent_screening"
    snapshot_dir.mkdir(parents=True)
    snapshot_dir.joinpath("latest.json").write_text(
        json.dumps(
            {
                "snapshot_type": "intelligent_screening",
                "generated_at": "2026-04-01T09:00:00",
                "recommendation_pool": {
                    "frontlist": [
                        {
                            "ts_code": "688710.SH",
                            "recommendation_text": "谨慎：暂不建议操作",
                            "action_plan": {"action_bias": "观察"},
                            "candidate_risk_blocked": False,
                        },
                        {
                            "ts_code": "000001.SZ",
                            "recommendation_text": "推荐：技术面良好，可适当关注",
                            "action_plan": {},
                            "candidate_risk_blocked": False,
                        },
                    ]
                },
                "report_context": {
                    "today_top3": [
                        {
                            "ts_code": "688710.SH",
                            "recommendation_text": "谨慎：暂不建议操作",
                            "action_plan": {"action_bias": "观察"},
                            "candidate_risk_blocked": True,
                        }
                    ]
                },
                "intelligent_report": {"blocks": {"focus_stocks": []}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(history_dir),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    payload = _load_intelligent_dashboard_payload(
        Settings(
            OCTTS_HISTORY_DIR_PATH=str(history_dir),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        )
    )

    assert payload["recommendation_pool"]["frontlist"][0]["recommendation_text"] == "等待确认：具备一定弹性，先观察是否形成一致性"
    assert payload["recommendation_pool"]["frontlist"][0]["action_plan"]["action_bias"] == "观察"
    assert payload["recommendation_pool"]["frontlist"][1]["recommendation_text"] == "建议跟踪：强度尚可，关注盘中承接与量价确认"
    assert payload["recommendation_pool"]["frontlist"][1]["action_plan"]["action_bias"] == "关注买点"
    assert payload["report_context"]["today_top3"][0]["recommendation_text"] == "等待确认：短线分歧偏大，暂不追高"
    assert payload["report_context"]["today_top3"][0]["action_plan"]["action_bias"] == "回避"


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
    assert result["top_recommendations"][0]["display_confidence"] is None
    assert result["top_recommendations"][0]["overall_confidence"] == 0.81
    assert result["top_recommendations"][0]["ai_confidence"] == 0.81
    assert result["top_recommendations"][0]["confidence"] == 0.81
    assert result["top_recommendations"][0]["continuation_bias_score"] is None
    assert result["top_recommendations"][0]["continuation_positive_flags"] == []
    assert result["top_recommendations"][0]["continuation_negative_flags"] == []
    assert result["top_recommendations"][1]["source_tag"] == "今日候选"
    assert result["top_recommendations"][1]["display_confidence"] is None
    assert result["top_recommendations"][1]["overall_confidence"] is None
    assert result["top_recommendations"][1]["ai_confidence"] == 0.64
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
    assert "真实智能报告" not in response.text
    assert "这里应该展示真实快照" not in response.text
    assert "最近生成时间：" in response.text
    assert "今日新闻" in response.text


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
    assert "容错报告" not in response.text
    assert "暂无最近运行时间" in response.text
    assert "今日新闻" in response.text
    assert "octts:intelligent-screening:pending-job-id" in response.text
    assert 'params.get("job_id")' in response.text
    assert 'fetch(`${jobsApiBase}/active`)' in response.text
    assert "setJobState(job)" in response.text


def test_intelligent_screening_job_endpoints_report_progress(tmp_path, monkeypatch) -> None:
    class FakeScheduler:
        thread_names = []

        def __init__(self, settings, progress_callback=None):
            self.settings = settings
            self.progress_callback = progress_callback

        async def run_intelligent_screening(self):
            del self.settings
            type(self).thread_names.append(threading.current_thread().name)
            if self.progress_callback:
                self.progress_callback(
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
                self.progress_callback(
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
        assert FakeScheduler.thread_names
        assert all(name != threading.current_thread().name for name in FakeScheduler.thread_names)


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



def test_cleanup_finished_jobs_removes_expired_job_snapshot_files(tmp_path) -> None:
    manager = IntelligentScreeningJobManager(str(tmp_path / "history"), retention_seconds=1)
    snapshot_dir = tmp_path / "history" / "intelligent_screening_jobs"
    expired_job = IntelligentScreeningJob(
        job_id="expired-job-1",
        status="succeeded",
        created_at=datetime.utcnow() - timedelta(seconds=10),
        finished_at=datetime.utcnow() - timedelta(seconds=5),
    )
    recent_job = IntelligentScreeningJob(
        job_id="recent-job-1",
        status="failed",
        created_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
    )

    manager._jobs[expired_job.job_id] = expired_job
    manager._jobs[recent_job.job_id] = recent_job
    manager._persist_job_snapshot_locked(expired_job)
    manager._persist_job_snapshot_locked(recent_job)

    manager._cleanup_finished_jobs_locked()

    assert not (snapshot_dir / "expired-job-1.json").exists()
    assert (snapshot_dir / "recent-job-1.json").exists()



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
                self.progress_callback(
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
        first_payload = first_response.json()

        for _ in range(20):
            active_response = client.get("/screen/intelligent/jobs/active")
            assert active_response.status_code == 200
            active_job = active_response.json()["job"]
            if active_job and active_job["job_id"] == first_payload["job_id"]:
                break
            time.sleep(0.01)

        second_response = client.post("/screen/intelligent/jobs")

        assert first_response.status_code == 202
        assert second_response.status_code == 202

        second_payload = second_response.json()

        assert first_payload["created"] is True
        assert second_payload["created"] is False
        assert second_payload["job_id"] == first_payload["job_id"]
        assert active_job is not None
        assert active_job["job_id"] == first_payload["job_id"]

        observed_progress = active_job["progress_percent"]
        for _ in range(20):
            active_response = client.get("/screen/intelligent/jobs/active")
            assert active_response.status_code == 200
            active_job = active_response.json()["job"]
            if active_job is not None:
                observed_progress = max(observed_progress, active_job["progress_percent"])
            if observed_progress >= 25:
                break
            time.sleep(0.01)
        assert observed_progress >= 25

        for _ in range(40):
            status_response = client.get(f"/screen/intelligent/jobs/{first_payload['job_id']}")
            assert status_response.status_code == 200
            if status_response.json()["status"] == "succeeded":
                break
            time.sleep(0.02)

        assert SlowFakeScheduler.run_count == 1

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile

from octts.config import Settings
from octts.schemas.report import (
    DecisionValidation,
    HistoricalAnalysisRecord,
    MemorySummary,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
)
from octts.services.news_aggregator import NewsAggregator, NewsItem, NewsSource, StockLiveNewsItem
from octts.services.position_store import FilePositionStore
from octts.services.report_exporter import ReportExporter
from octts.services.history_store import FileHistoryStore

UTC = timezone.utc


class StubLLMClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if "主题聚类分析" in prompt:
            return '{"clusters": [{"theme": "银行催化", "news_indices": [1, 2, 3], "importance": 0.8, "key_stocks": ["600000.SH"], "summary": "板块催化"}]}'
        return '{"1": {"importance": 0.9, "sentiment": 0.4, "stocks": ["600000.SH"]}}'


def _build_record(ts_code: str) -> HistoricalAnalysisRecord:
    generated_at = datetime(2026, 3, 30, 10, 0, tzinfo=UTC)
    snapshot = PriceSnapshot(ts_code=ts_code, name=f"Name-{ts_code}", trade_date="20260330", close=10.2, high=10.5, low=9.9)
    report = StructuredAnalysis(
        ts_code=ts_code,
        phase="review",
        trend_judgement="趋势观察",
        previous_view_status="initial",
        operation_advice="观察关键位",
        risk_warning=["量能不足"],
        observation_points=["关注压力位"],
        summary_markdown=f"{ts_code} summary",
        decision=TradingDecision(
            signal="buy",
            rationale="形态仍可跟踪。",
            entry_zone=PriceZone(low=10.0, high=10.3),
            stop_loss=9.8,
            take_profit=[10.8],
            invalidation_condition="跌破止损位",
            holding_horizon="swing",
            confidence_score=0.7,
            risk_reward_ratio=1.6,
            evidence=["结构未破坏"],
        ),
        memory=MemorySummary(
            ts_code=ts_code,
            generated_at=generated_at,
            phase="review",
            trend_bias="bullish",
            capital_flow_view="资金中性偏强",
            confidence_score=0.7,
            summary="继续观察",
        ),
    )
    return HistoricalAnalysisRecord(
        record_id=f"record-{ts_code}",
        request_id="req-1",
        generated_at=generated_at,
        snapshot=snapshot,
        report=report,
        validation=DecisionValidation(status="watching_entry", note="等待触发入场。"),
    )


def test_export_latest_intelligent_screening_zip_reuses_action_and_verdict_fields(tmp_path) -> None:
    settings = Settings(
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        OCTTS_HISTORY_DIR_PATH=str(tmp_path / "history"),
    )
    history_store = FileHistoryStore(str(tmp_path / "history-store"))
    position_store = FilePositionStore(str(tmp_path / "positions.json"))
    history_store.append(_build_record("600000.SH"))

    snapshot_dir = tmp_path / "history" / "intelligent_screening"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "latest.json").write_text(
        """
        {
          "generated_at": "2026-03-30T10:00:00+00:00",
          "screening_results": {"strategy_count": 4, "total_stocks": 12, "final_recommendations": 2},
          "recommendation_pool": {
            "frontlist": [{"ts_code": "600000.SH", "source_tag": "今日Top3", "recommendation_score": 88, "priority_score": 86, "ai_confidence": 0.8}],
            "today_top": [{"ts_code": "600000.SH", "name": "浦发银行", "source_tag": "今日Top3", "recommendation_score": 88, "priority_score": 86, "ai_confidence": 0.8, "technical_signal": "量价共振", "recommendation_text": "建议跟踪", "action_plan": {"entry_zone": "10.0-10.2", "stop_loss": "9.8", "take_profit": "10.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破 9.8"}}],
            "yesterday_continuations": [{"ts_code": "000001.SZ", "name": "平安银行", "source_tag": "昨日延续", "recommendation_score": 75, "priority_score": 72, "ai_confidence": 0.7, "action_plan": {"take_profit": "反抽 12.6 止盈", "stop_loss": "跌破 11.8 离场", "invalid_condition": "午后不能重新站回 12.0", "holding_horizon": "2-3个交易日"}}]
          },
          "ai_analyses": {},
          "news_clusters": [],
          "report_context": {
            "today_top3": [{"ts_code": "600000.SH", "technical_signal": "量价共振", "recommendation_text": "建议跟踪"}],
            "yesterday_top3_review": [{"ts_code": "000001.SZ", "today_verdict": "不能转强则离场", "review_status": "减仓观察"}],
            "today_top3_live_context": [{"ts_code": "600000.SH", "name": "浦发银行", "query": "浦发银行", "items": [{"title": "浦发银行披露一季报预告", "summary": "业绩边际改善", "source": "东方财富", "url": "https://example.com/live1", "publish_time": "2026-03-30 09:35", "category": "公告"}]}],
            "yesterday_top3_live_context": [{"ts_code": "000001.SZ", "name": "平安银行", "query": "平安银行", "items": [{"title": "平安银行盘中异动", "summary": "资金回流", "source": "东方财富", "url": "https://example.com/live2", "publish_time": "2026-03-30 10:05", "category": "新闻"}]}]
          },
          "intelligent_report": {
            "title": "智能选股报告",
            "summary": "摘要",
            "blocks": {
              "focus_stocks": [{"ts_code": "600000.SH", "name": "浦发银行", "overall_assessment": "优先等回踩", "action_plan": {"entry_zone": "10.0-10.2", "stop_loss": "9.8", "take_profit": "10.8", "holding_horizon": "3个交易日", "invalid_condition": "跌破 9.8"}}],
              "yesterday_reviews": [{"ts_code": "000001.SZ", "name": "平安银行", "today_verdict": "不能转强则离场", "review_status": "减仓观察", "analysis": "反弹力度不足", "action_plan": {"take_profit": "反抽 12.6 止盈", "stop_loss": "跌破 11.8 离场", "invalid_condition": "午后不能重新站回 12.0", "holding_horizon": "2-3个交易日"}}],
              "comparison": {},
              "overall_action": {"headline": "保持节奏", "action_items": ["聚焦 Top3"]}
            }
          }
        }
        """,
        encoding="utf-8",
    )

    archive_name, archive_bytes = ReportExporter(
        settings=settings,
        history_store=history_store,
        position_store=position_store,
    ).export_latest_intelligent_screening_zip()

    assert archive_name.endswith(".zip")
    with ZipFile(BytesIO(archive_bytes)) as archive:
        index_html = archive.read("index.html").decode("utf-8")
        dashboard_html = archive.read("dashboard.html").decode("utf-8")
        detail_html = archive.read("stocks/600000.SH.html").decode("utf-8")
    assert "<!DOCTYPE html>" in index_html
    assert "./dashboard.html" in index_html
    assert "./index.html" in index_html
    assert "../dashboard.html" in detail_html
    assert "600000.SH" in detail_html


def test_report_exporter_methodology_matches_news_bonus(tmp_path) -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"))
    methodology = ReportExporter(
        settings=settings,
        history_store=FileHistoryStore(str(tmp_path / "history")),
        position_store=FilePositionStore(str(tmp_path / "positions.json")),
    ).build_intelligent_screening_payload()["recommendation_methodology"]

    joined = "\n".join(methodology.get("score_formula") or [])
    assert "加 3 分" in joined
    assert "新闻加分" in joined
    assert "最终分数" in joined


async def _run_news_aggregation_checks() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    llm = StubLLMClient()
    aggregator = NewsAggregator(settings, llm_client=llm)
    news_items = [
        NewsItem(source=NewsSource.CAILIAN, title="银行板块获政策催化", content="600000.SH 获政策支持并伴随量能提升，板块热度提升。", url="u1", publish_time=datetime.now()),
        NewsItem(source=NewsSource.EASTMONEY, title="简讯", content="太短", url="u2", publish_time=datetime.now()),
        NewsItem(source=NewsSource.SINA, title="市场午后震荡点评", content="泛泛而谈，没有明显股票线索。", url="u3", publish_time=datetime.now()),
        NewsItem(source=NewsSource.JINSHI, title="机器人产业链订单增长", content="多家公司订单增长，000001.SZ 获资金关注。", url="u4", publish_time=datetime.now()),
        NewsItem(source=NewsSource.CAILIAN, title="银行板块获政策催化", content="600000.SH 获政策支持并伴随量能提升，板块热度提升。", url="u5", publish_time=datetime.now()),
        NewsItem(source=NewsSource.YICAI, title="地产链政策优化", content="地产链迎来政策优化，000002.SZ 受到带动。", url="u6", publish_time=datetime.now()),
    ]

    analyzed = await aggregator.analyze_importance(news_items)
    clusters = await aggregator.cluster_news(analyzed, min_cluster_size=3)

    assert len([prompt for prompt in llm.calls if "重要性和市场影响" in prompt]) == 1
    assert len([prompt for prompt in llm.calls if "主题聚类分析" in prompt]) == 1
    assert analyzed[1].importance == 0.2
    assert analyzed[2].importance == 0.2
    assert analyzed[4].importance == 0.2
    assert len(clusters) == 1


def test_news_aggregator_prefilter_reduces_importance_calls() -> None:
    import asyncio

    asyncio.run(_run_news_aggregation_checks())


def test_news_aggregator_collect_focus_stock_live_context() -> None:
    import asyncio

    async def _run() -> None:
        settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
        aggregator = NewsAggregator(settings)

        async def _fake_collect_stock_news(*, keyword: str, limit: int = 3):
            assert keyword == "浦发银行"
            assert limit == 2
            return [
                StockLiveNewsItem(
                    title="浦发银行披露一季报预告",
                    summary="业绩边际改善",
                    source="东方财富",
                    url="https://example.com/live1",
                    publish_time="2026-03-30 09:35",
                    category="公告",
                )
            ]

        aggregator.stock_news_collector.collect_stock_news = _fake_collect_stock_news
        payload = await aggregator.collect_focus_stock_live_context(
            today_top3=[{"ts_code": "600000.SH", "name": "浦发银行"}],
            yesterday_top3_review=[],
            per_stock_limit=2,
        )

        assert payload["today_top3_live_context"][0]["ts_code"] == "600000.SH"
        assert payload["today_top3_live_context"][0]["items"]
        assert payload["yesterday_top3_live_context"] == []

    asyncio.run(_run())

from octts.config import Settings
from octts.schemas.report import (
    AnalysisRequest,
    MemorySummary,
    PredictionWindow,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
    TrendBreakdown,
)
from octts.services.analysis_pipeline import AnalysisPipeline, format_reports_as_markdown
from octts.services.history_store import FileHistoryStore
from octts.services.memory_store import FileMemoryStore


class FakeTushareClient:
    def fetch_snapshot(self, *, ts_code: str, phase: str, trade_date: str | None = None) -> PriceSnapshot:
        return PriceSnapshot(
            ts_code=ts_code,
            name="PF Bank",
            trade_date=trade_date or "20260309",
            close=10.2,
            pct_chg=1.5,
        )


class FakeLLMClient:
    def analyze(self, *, system_prompt: str, user_prompt: str) -> StructuredAnalysis:
        del system_prompt, user_prompt
        memory = MemorySummary(
            ts_code="600000.SH",
            phase="review",
            trend_bias="bullish",
            short_term_bias="bullish",
            mid_term_bias="bullish",
            long_term_bias="neutral",
            capital_flow_view="资金延续净流入",
            confidence_score=0.8,
            summary="趋势延续，等待确认放量",
            support_levels=[9.8],
            resistance_levels=[10.5],
            key_risks=["量能不足"],
            next_checkpoints=["关注 10.5 是否有效突破"],
        )
        return StructuredAnalysis(
            ts_code="600000.SH",
            phase="review",
            trend_judgement="震荡上行",
            trend_breakdown=TrendBreakdown(
                short_term="bullish",
                mid_term="bullish",
                long_term="neutral",
                short_term_reason="分时低点抬高。",
                mid_term_reason="日线仍在 20 日均线上方。",
                long_term_reason="周线仍处震荡整理。",
            ),
            previous_view_status="confirmed",
            operation_advice="继续观察，突破后再考虑加仓",
            risk_warning=["量能不足"],
            observation_points=["关注 10.5 压力位"],
            summary_markdown="**趋势延续**，但仍需成交量确认。",
            decision=TradingDecision(
                signal="buy",
                rationale="价格接近支撑且资金流改善。",
                entry_zone=PriceZone(low=10.0, high=10.3),
                stop_loss=9.8,
                take_profit=[10.5, 10.8],
                invalidation_condition="跌破 9.8 且量能放大",
                holding_horizon="swing",
                confidence_score=0.78,
                risk_reward_ratio=2.0,
                evidence=["价格守住支撑位", "资金延续净流入"],
            ),
            prediction_windows=[
                PredictionWindow(window="next_1d", bias="neutral", confidence_score=0.56, rationale="等待放量确认。"),
                PredictionWindow(window="next_3d", bias="bullish", confidence_score=0.68, rationale="支撑未破且资金改善。"),
                PredictionWindow(window="next_5d", bias="bullish", confidence_score=0.71, rationale="中线结构仍偏强。"),
            ],
            memory=memory,
        )


class FlakyLLMClient(FakeLLMClient):
    def analyze(self, *, system_prompt: str, user_prompt: str) -> StructuredAnalysis:
        if "000001.SZ" in user_prompt:
            raise ValueError("malformed structured output")
        return super().analyze(system_prompt=system_prompt, user_prompt=user_prompt)


class FakeWeComClient:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send_markdown(self, content: str) -> None:
        self.messages.append(content)


def test_pipeline_runs_and_persists_memory(tmp_path) -> None:
    settings = Settings(
        OCTTS_STOCK_POOL="600000.SH",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    memory_store = FileMemoryStore(settings.memory_file_path)
    history_store = FileHistoryStore(str(tmp_path / "history.json"))
    wecom_client = FakeWeComClient()

    pipeline = AnalysisPipeline(
        settings=settings,
        tushare_client=FakeTushareClient(),
        llm_client=FakeLLMClient(),
        memory_store=memory_store,
        history_store=history_store,
        wecom_client=wecom_client,
    )

    result = pipeline.run(AnalysisRequest(phase="review"))

    assert result.notification_sent is True
    assert len(result.reports) == 1
    assert result.reports[0].decision.signal == "buy"
    assert memory_store.get("600000.SH") is not None
    assert wecom_client.messages
    assert history_store.list_records("600000.SH")


def test_format_reports_as_markdown_includes_core_sections() -> None:
    report = FakeLLMClient().analyze(system_prompt="", user_prompt="")
    content = format_reports_as_markdown([report])

    assert "# OCTTS 自动分析报告" in content
    assert "600000.SH" in content
    assert "震荡上行" in content
    assert "三层趋势" in content
    assert "未来3个交易日：看多" in content
    assert "交易信号：买入" in content


def test_pipeline_skips_failed_symbol_for_multi_stock_requests(tmp_path) -> None:
    settings = Settings(
        OCTTS_STOCK_POOL="600000.SH,000001.SZ",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    memory_store = FileMemoryStore(settings.memory_file_path)
    history_store = FileHistoryStore(str(tmp_path / "history.json"))

    pipeline = AnalysisPipeline(
        settings=settings,
        tushare_client=FakeTushareClient(),
        llm_client=FlakyLLMClient(),
        memory_store=memory_store,
        history_store=history_store,
    )

    result = pipeline.run(AnalysisRequest(phase="review"))

    assert len(result.reports) == 1
    assert result.reports[0].ts_code == "600000.SH"
    assert len(result.errors) == 1
    assert result.errors[0].ts_code == "000001.SZ"
    assert history_store.list_records("600000.SH")
    assert history_store.list_records("000001.SZ") == []

import json

from octts.config import Settings
from octts.schemas.backtest import BacktestRequest, DailyBar
from octts.schemas.report import (
    MemorySummary,
    PredictionWindow,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
    TrendBreakdown,
)
from octts.services.analysis_pipeline import AnalysisPipeline
from octts.services.backtest_engine import BacktestEngine
from octts.services.history_store import FileHistoryStore
from octts.services.memory_store import FileMemoryStore
from octts.services.position_store import FilePositionStore


class FakeMarketDataClient:
    def __init__(self) -> None:
        self._bars = {
            "600000.SH": [
                DailyBar(ts_code="600000.SH", trade_date="20260105", open=10.0, high=10.2, low=9.9, close=10.0),
                DailyBar(ts_code="600000.SH", trade_date="20260106", open=10.0, high=10.7, low=9.95, close=10.6),
                DailyBar(ts_code="600000.SH", trade_date="20260107", open=10.55, high=10.8, low=10.3, close=10.7),
            ]
        }

    def fetch_trading_dates(self, *, start_date: str, end_date: str) -> list[str]:
        return [item.trade_date for item in self._bars["600000.SH"] if start_date <= item.trade_date <= end_date]

    def fetch_daily_bars(self, *, ts_code: str, start_date: str, end_date: str) -> list[DailyBar]:
        return [item for item in self._bars.get(ts_code, []) if start_date <= item.trade_date <= end_date]

    def fetch_historical_snapshot(self, *, ts_code: str, phase: str, trade_date: str) -> PriceSnapshot:
        del phase
        bar = next(item for item in self._bars[ts_code] if item.trade_date == trade_date)
        return PriceSnapshot(
            ts_code=ts_code,
            trade_date=trade_date,
            open=bar.open,
            close=bar.close,
            high=bar.high,
            low=bar.low,
            daily_summary=[item.model_dump(mode="json") for item in self._bars[ts_code] if item.trade_date <= trade_date],
        )


class FakeLLMClient:
    def analyze(self, *, system_prompt: str, user_prompt: str) -> StructuredAnalysis:
        del system_prompt
        payload = json.loads(user_prompt)
        trade_date = payload["snapshot"]["trade_date"]
        signal = "buy" if trade_date == "20260105" else "avoid"
        memory = MemorySummary(
            ts_code="600000.SH",
            phase="review",
            trend_bias="bullish" if signal == "buy" else "neutral",
            capital_flow_view="资金流平稳",
            confidence_score=0.75 if signal == "buy" else 0.5,
            summary=f"{trade_date} snapshot",
        )
        return StructuredAnalysis(
            ts_code="600000.SH",
            phase="review",
            trend_judgement="等待确认",
            trend_breakdown=TrendBreakdown(short_term="bullish", mid_term="neutral", long_term="neutral"),
            previous_view_status="initial",
            operation_advice="等待确认",
            risk_warning=[],
            observation_points=[],
            summary_markdown="回测测试",
            decision=TradingDecision(
                signal=signal,
                rationale="测试信号",
                entry_zone=PriceZone(low=9.8, high=10.2),
                stop_loss=9.7,
                take_profit=[10.5],
                invalidation_condition="跌破止损",
                holding_horizon="swing",
                confidence_score=0.75,
                risk_reward_ratio=2.0,
                evidence=["单元测试"],
            ),
            prediction_windows=[
                PredictionWindow(window="next_1d", bias="bullish", confidence_score=0.7, rationale="测试")
            ],
            memory=memory,
        )


def test_backtest_engine_runs_review_only_strategy(tmp_path) -> None:
    settings = Settings(
        OCTTS_STOCK_POOL="600000.SH",
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
    )
    market_data_client = FakeMarketDataClient()
    pipeline = AnalysisPipeline(
        settings=settings,
        tushare_client=market_data_client,
        llm_client=FakeLLMClient(),
        memory_store=FileMemoryStore(str(tmp_path / "memory.json")),
        history_store=FileHistoryStore(str(tmp_path / "history")),
        position_store=FilePositionStore(str(tmp_path / "positions.json")),
    )
    engine = BacktestEngine(
        pipeline=pipeline,
        market_data_client=market_data_client,
    )

    result = engine.run(
        BacktestRequest(
            start_date="20260105",
            end_date="20260107",
            stock_pool=["600000.SH"],
            initial_cash=100000,
            position_size_pct=0.2,
        )
    )

    assert result.phase == "review"
    assert result.stock_pool == ["600000.SH"]
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].entry_date == "20260106"
    assert result.metrics.trade_count == 1
    assert result.metrics.total_return > 0
    assert len(result.daily_positions) == 3

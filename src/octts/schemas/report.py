from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


AnalysisPhase = Literal["morning", "afternoon", "review"]
TrendBias = Literal["bullish", "neutral", "bearish"]
SignalType = Literal["buy", "hold", "reduce", "sell", "avoid"]
HoldingHorizon = Literal["intraday", "swing", "position"]
PositionStatus = Literal["holding", "watching"]
PredictionWindowType = Literal["next_1d", "next_3d", "next_5d"]
ValidationStatus = Literal[
    "no_signal",
    "watching_setup",
    "watching_entry",
    "tracking_position",
    "entered",
    "take_profit_hit",
    "stop_loss_hit",
    "expired",
]


class PriceSnapshot(BaseModel):
    ts_code: str
    name: Optional[str] = None
    trade_date: Optional[str] = None
    open: Optional[float] = None
    close: Optional[float] = None
    pct_chg: Optional[float] = None
    vol_ratio: Optional[float] = None
    turnover_rate: Optional[float] = None
    amount: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    minute_summary: list[dict[str, Any]] = Field(default_factory=list)
    daily_summary: list[dict[str, Any]] = Field(default_factory=list)
    weekly_summary: list[dict[str, Any]] = Field(default_factory=list)
    moneyflow_summary: dict[str, Any] = Field(default_factory=dict)
    financial_indicators: list[dict[str, Any]] = Field(default_factory=list)
    earnings_express: list[dict[str, Any]] = Field(default_factory=list)


class MemorySummary(BaseModel):
    ts_code: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase: AnalysisPhase
    trend_bias: TrendBias
    short_term_bias: Optional[TrendBias] = None
    mid_term_bias: Optional[TrendBias] = None
    long_term_bias: Optional[TrendBias] = None
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    capital_flow_view: str
    key_risks: list[str] = Field(default_factory=list)
    next_checkpoints: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0, le=1)
    summary: str


class TrendBreakdown(BaseModel):
    short_term: TrendBias = "neutral"
    mid_term: TrendBias = "neutral"
    long_term: TrendBias = "neutral"
    short_term_reason: str = ""
    mid_term_reason: str = ""
    long_term_reason: str = ""


class PredictionWindow(BaseModel):
    window: PredictionWindowType
    bias: TrendBias
    confidence_score: float = Field(ge=0, le=1)
    rationale: str


class PriceZone(BaseModel):
    low: Optional[float] = None
    high: Optional[float] = None


class TradingDecision(BaseModel):
    signal: SignalType
    rationale: str
    entry_zone: Optional[PriceZone] = None
    stop_loss: Optional[float] = None
    take_profit: list[float] = Field(default_factory=list)
    invalidation_condition: str
    holding_horizon: HoldingHorizon
    confidence_score: float = Field(ge=0, le=1)
    risk_reward_ratio: Optional[float] = None
    evidence: list[str] = Field(default_factory=list)


class DecisionValidation(BaseModel):
    status: ValidationStatus
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str
    entry_triggered: bool = False
    target_hit_level: Optional[float] = None
    stop_loss_hit: bool = False
    current_close: Optional[float] = None
    current_high: Optional[float] = None
    current_low: Optional[float] = None


class StructuredAnalysis(BaseModel):
    ts_code: str
    phase: AnalysisPhase
    trend_judgement: str
    trend_breakdown: TrendBreakdown = Field(default_factory=TrendBreakdown)
    previous_view_status: Literal["confirmed", "weakened", "reversed", "initial"]
    operation_advice: str
    risk_warning: list[str] = Field(default_factory=list)
    observation_points: list[str] = Field(default_factory=list)
    summary_markdown: str
    decision: TradingDecision
    prediction_windows: list[PredictionWindow] = Field(default_factory=list)
    memory: MemorySummary


class HistoricalAnalysisRecord(BaseModel):
    record_id: str
    request_id: str
    generated_at: datetime
    snapshot: PriceSnapshot
    report: StructuredAnalysis
    validation: DecisionValidation


class ValidationUpdate(BaseModel):
    record_id: str
    ts_code: str
    previous_status: ValidationStatus
    current_status: ValidationStatus
    note: str
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AnalysisRequest(BaseModel):
    phase: AnalysisPhase
    stock_pool: Optional[list[str]] = None
    trade_date: Optional[str] = None
    notify: bool = True
    force_refresh: bool = False


class SymbolAnalysisError(BaseModel):
    ts_code: str
    phase: AnalysisPhase
    error: str


class AnalysisResult(BaseModel):
    request_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase: AnalysisPhase
    reports: list[StructuredAnalysis]
    notification_sent: bool = False
    validation_updates: list[ValidationUpdate] = Field(default_factory=list)
    errors: list[SymbolAnalysisError] = Field(default_factory=list)
    raw_payloads: dict[str, Any] = Field(default_factory=dict)

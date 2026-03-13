from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc

from pydantic import BaseModel, Field


class DailyBar(BaseModel):
    ts_code: str
    trade_date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    pct_chg: float | None = None
    vol: float | None = None
    amount: float | None = None


class BacktestRequest(BaseModel):
    stock_pool: list[str] | None = None
    start_date: str
    end_date: str
    initial_cash: float = Field(default=100000.0, gt=0)
    position_size_pct: float = Field(default=0.2, gt=0, le=1)
    commission_rate: float = Field(default=0.0003, ge=0)
    slippage_rate: float = Field(default=0.0005, ge=0)
    phase: str = "review"


class BacktestTrade(BaseModel):
    ts_code: str
    signal_date: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str
    return_pct: float
    pnl: float
    holding_days: int


class BacktestDailyPosition(BaseModel):
    trade_date: str
    cash: float
    market_value: float
    equity: float
    open_positions: int


class BacktestMetrics(BaseModel):
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float | None = None
    trade_count: int = 0


class BacktestResult(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    phase: str
    stock_pool: list[str]
    start_date: str
    end_date: str
    initial_cash: float
    ending_cash: float
    trades: list[BacktestTrade] = Field(default_factory=list)
    daily_positions: list[BacktestDailyPosition] = Field(default_factory=list)
    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)

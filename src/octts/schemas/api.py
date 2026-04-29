from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from octts.schemas.report import PositionStatus


class StockPoolItemRequest(BaseModel):
    ts_code: str


class PositionStatusRequest(BaseModel):
    position_status: PositionStatus


class AnalysisActionResponse(BaseModel):
    cleared_symbols: List[str] = Field(default_factory=list)
    cleared_all: bool = False
    removed_records: int = 0
    removed_memory_items: int = 0
    removed_generated_at: Optional[str] = None
    remaining_records: int = 0
    updated_memory: bool = False


class LightweightBacktestRequest(BaseModel):
    start_date: str
    end_date: str
    holding_days: int = Field(default=5, ge=1, le=60)
    top_n: int = Field(default=10, ge=1, le=50)
    commission_rate: float = Field(default=0.0003, ge=0)
    slippage_rate: float = Field(default=0.0005, ge=0)
    strategies: List[str] = Field(default_factory=list)
    stock_pool: List[str] = Field(default_factory=list)

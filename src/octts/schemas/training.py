from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


SHORT_TERM_FEATURE_SCHEMA_VERSION = "v1"
RAW_MARKET_FEATURE_SCHEMA_VERSION = "raw_v1"


class RawMarketTrainingSample(BaseModel):
    feature_schema_version: str = Field(default=RAW_MARKET_FEATURE_SCHEMA_VERSION)
    trade_date: date
    ts_code: str
    entry_price: Optional[float] = None
    close: Optional[float] = None
    pct_change: Optional[float] = None
    turnover_rate: Optional[float] = None
    volume_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    amount: Optional[float] = None
    vol: Optional[float] = None
    return_3d_past: Optional[float] = None
    return_5d_past: Optional[float] = None
    return_10d_past: Optional[float] = None
    volatility_5d: Optional[float] = None
    volatility_10d: Optional[float] = None
    max_drawdown_10d_past: Optional[float] = None
    close_to_ma5: Optional[float] = None
    close_to_ma10: Optional[float] = None
    close_to_ma20: Optional[float] = None
    price_position_20d: Optional[float] = None
    price_position_10d: Optional[float] = None
    avg_turnover_rate_5d: Optional[float] = None
    avg_volume_ratio_5d: Optional[float] = None
    market_return_1d: Optional[float] = None
    market_return_3d: Optional[float] = None
    market_return_5d: Optional[float] = None
    market_up_ratio_1d: Optional[float] = None
    market_up_ratio_3d_avg: Optional[float] = None
    market_up_days_5d: Optional[int] = None
    stock_vs_market_return_1d: Optional[float] = None
    stock_vs_market_return_2d: Optional[float] = None
    stock_vs_market_return_3d: Optional[float] = None
    stock_vs_market_return_5d: Optional[float] = None
    stock_vs_market_return_10d: Optional[float] = None
    pct_change_rank_pct: Optional[float] = None
    turnover_rate_rank_pct: Optional[float] = None
    volume_ratio_rank_pct: Optional[float] = None
    up_days_3d: Optional[int] = None
    up_days_5d: Optional[int] = None
    new_high_gap_20d: Optional[float] = None
    new_high_gap_10d: Optional[float] = None
    new_low_gap_20d: Optional[float] = None
    amount_ratio_1d_5d: Optional[float] = None
    amount_ratio_3d_10d: Optional[float] = None
    turnover_rate_change_1d: Optional[float] = None
    turnover_rate_change_5d: Optional[float] = None
    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    vs_market_1d: Optional[float] = None
    vs_market_3d: Optional[float] = None
    vs_market_5d: Optional[float] = None
    label_up_1d: Optional[bool] = None
    label_up_3d: Optional[bool] = None
    label_up_5d: Optional[bool] = None
    label_vs_market_1d: Optional[bool] = None
    label_vs_market_3d: Optional[bool] = None
    label_vs_market_5d: Optional[bool] = None
    label_strong_1d: Optional[bool] = None


class ShortTermTrainingSample(BaseModel):
    feature_schema_version: str = Field(default=SHORT_TERM_FEATURE_SCHEMA_VERSION)
    trade_date: date
    ts_code: str
    name: Optional[str] = None
    source_tag: Optional[str] = None
    in_frontlist: bool = False
    recommend_rank: Optional[int] = None
    strategy_count: int = 0
    is_repeat_pick: bool = False
    news_mentioned: bool = False
    technical_signal: Optional[str] = None

    entry_price: Optional[float] = None
    close: Optional[float] = None
    pct_change: Optional[float] = None
    volume_ratio: Optional[float] = None
    turnover_rate: Optional[float] = None
    recommendation_score: Optional[float] = None
    overall_score: Optional[float] = None
    technical_score: Optional[float] = None
    fundamental_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    news_score: Optional[float] = None
    base_score: Optional[float] = None
    sentiment_adjustment: Optional[float] = None
    news_adjustment: Optional[float] = None

    industry: Optional[str] = None
    industry_heat_score: Optional[float] = None
    industry_flow_bias: Optional[str] = None

    distribution_risk_score: Optional[float] = None
    distribution_risk_flags: List[str] = Field(default_factory=list)
    moneyflow_3d_value: Optional[float] = None
    recent_large_order_net_inflow: Optional[float] = None
    recent_super_large_order_net_inflow: Optional[float] = None
    turnover_spike_ratio: Optional[float] = None
    recent_runup_5d: Optional[float] = None
    continuation_bias_score: Optional[float] = None
    continuation_positive_flags: List[str] = Field(default_factory=list)
    continuation_negative_flags: List[str] = Field(default_factory=list)
    top3_risk_penalty: Optional[float] = None
    short_term_contradiction_penalty: Optional[float] = None
    late_stage_momentum_flag: bool = False
    candidate_risk_blocked: bool = False

    previous_recommendation_score: Optional[float] = None
    previous_overall_score: Optional[float] = None
    score_change: Optional[float] = None

    action_plan: Dict[str, Any] = Field(default_factory=dict)

    return_1d: Optional[float] = None
    return_3d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    max_drawdown_10d: Optional[float] = None
    benchmark_return_5d: Optional[float] = None
    vs_benchmark_5d: Optional[float] = None
    label_up_1d: Optional[bool] = None

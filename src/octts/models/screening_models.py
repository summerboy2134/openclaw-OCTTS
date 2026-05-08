"""Database models for screening system using SQLAlchemy."""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Iterable

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, Text, JSON, Index, ForeignKey, Table, Date, func, inspect, text, select
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from octts.schemas.screener import ScreenCriteria, ScreenResult, StockScreenItem, TrackedRecommendationState
from octts.schemas.training import ShortTermTrainingSample

Base = declarative_base()


# 多对多关系表
stock_strategy_association = Table(
    'stock_strategy_association',
    Base.metadata,
    Column('stock_result_id', Integer, ForeignKey('stock_screening_results.id')),
    Column('strategy_id', Integer, ForeignKey('screening_strategies.id'))
)


class ScreeningStrategy(Base):
    """选股策略"""
    __tablename__ = 'screening_strategies'

    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    criteria = Column(JSON)
    category = Column(String(50))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    screening_runs = relationship("ScreeningRun", back_populates="strategy")


class ScreeningRun(Base):
    """选股运行记录"""
    __tablename__ = 'screening_runs'

    id = Column(Integer, primary_key=True)
    run_id = Column(String(50), unique=True, nullable=False)
    strategy_id = Column(Integer, ForeignKey('screening_strategies.id'))
    run_date = Column(DateTime, nullable=False)
    total_stocks = Column(Integer, default=0)
    execution_time = Column(Float)
    status = Column(String(20), default='completed')
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    strategy = relationship("ScreeningStrategy", back_populates="screening_runs")
    stock_results = relationship("StockScreeningResult", back_populates="screening_run")

    __table_args__ = (
        Index('idx_run_date', 'run_date'),
        Index('idx_strategy_date', 'strategy_id', 'run_date'),
    )


class StockScreeningResult(Base):
    """股票筛选结果"""
    __tablename__ = 'stock_screening_results'

    id = Column(Integer, primary_key=True)
    screening_run_id = Column(Integer, ForeignKey('screening_runs.id'))
    ts_code = Column(String(20), nullable=False)
    name = Column(String(50))
    close = Column(Float)
    pct_change = Column(Float)
    volume_ratio = Column(Float)
    turnover_rate = Column(Float)
    rsi = Column(Float)
    ma5 = Column(Float)
    ma20 = Column(Float)
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    industry = Column(String(50))
    score = Column(Float)
    match_reasons = Column(JSON)
    rank = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

    screening_run = relationship("ScreeningRun", back_populates="stock_results")
    strategies = relationship("ScreeningStrategy", secondary=stock_strategy_association)
    ai_analysis = relationship("StockAIAnalysis", back_populates="screening_result", uselist=False)

    __table_args__ = (
        Index('idx_ts_code', 'ts_code'),
        Index('idx_run_stock', 'screening_run_id', 'ts_code'),
        Index('idx_score', 'score'),
    )


class StockAIAnalysis(Base):
    """股票AI分析结果"""
    __tablename__ = 'stock_ai_analyses'

    id = Column(Integer, primary_key=True)
    screening_result_id = Column(Integer, ForeignKey('stock_screening_results.id'))
    ts_code = Column(String(20), nullable=False)
    analysis_date = Column(DateTime, nullable=False)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    news_score = Column(Float)
    overall_score = Column(Float)
    overall_confidence = Column(Float)
    technical_analysis = Column(Text)
    fundamental_analysis = Column(Text)
    sentiment_analysis = Column(Text)
    news_analysis = Column(Text)
    recommendation = Column(String(200))
    ai_summary = Column(Text)
    key_points = Column(JSON)
    has_conflict = Column(Boolean, default=False)
    conflict_resolution = Column(Text)
    iteration_count = Column(Integer, default=1)
    supplementary_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)

    screening_result = relationship("StockScreeningResult", back_populates="ai_analysis")

    __table_args__ = (
        Index('idx_ai_ts_code', 'ts_code'),
        Index('idx_ai_date', 'analysis_date'),
        Index('idx_ai_score', 'overall_score'),
    )


class RecommendationRun(Base):
    """智能选股推荐批次"""
    __tablename__ = 'recommendation_runs'

    id = Column(Integer, primary_key=True)
    run_id = Column(String(50), unique=True, nullable=False)
    trade_date = Column(Date, nullable=False)
    generated_at = Column(DateTime, default=datetime.now, nullable=False)
    candidate_count = Column(Integer, default=0)
    final_count = Column(Integer, default=0)
    report_id = Column(String(50))

    items = relationship("RecommendationItem", back_populates="recommendation_run")

    __table_args__ = (
        Index('idx_recommendation_run_trade_date', 'trade_date', unique=True),
        Index('idx_recommendation_run_generated_at', 'generated_at'),
    )


class RecommendationItem(Base):
    """最终推荐股票记录"""
    __tablename__ = 'recommendation_items'

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('recommendation_runs.id'), nullable=False)
    ts_code = Column(String(20), nullable=False)
    name = Column(String(50))
    recommend_rank = Column(Integer)
    recommend_score = Column(Float)
    ai_confidence = Column(Float)
    source_tag = Column(String(50), default='今日Top3')
    is_repeat_pick = Column(Boolean, default=False)
    strategy_count = Column(Integer, default=0)
    news_mentioned = Column(Boolean, default=False)
    technical_signal = Column(String(200))
    recommendation_text = Column(Text)
    status = Column(String(20), default='new')
    tracking_days = Column(Integer, default=0)
    trade_date = Column(Date, nullable=False)
    entry_price = Column(Float)
    score_mode = Column(String(50))
    rerank_pool_rank = Column(Integer)
    rerank_blend_score = Column(Float)
    rerank_model_score = Column(Float)
    rerank_rule_score = Column(Float)
    rerank_rule_weight = Column(Float)
    rerank_model_target = Column(String(50))
    selection_stage = Column(String(50))
    selection_reason = Column(Text)
    selection_reason_components = Column(JSON)
    structured_rank_score = Column(Float)
    structured_rank_position = Column(Integer)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    recommendation_run = relationship("RecommendationRun", back_populates="items")
    performance = relationship("RecommendationPerformance", back_populates="recommendation_item", uselist=False)

    __table_args__ = (
        Index('idx_recommendation_item_run_id', 'run_id'),
        Index('idx_recommendation_item_ts_code', 'ts_code'),
        Index('idx_recommendation_item_status', 'status'),
        Index('idx_recommendation_item_trade_date', 'trade_date'),
    )


class RecommendationPoolState(Base):
    """持续跟踪推荐池状态"""
    __tablename__ = 'recommendation_pool_states'

    id = Column(Integer, primary_key=True)
    ts_code = Column(String(20), nullable=False)
    trade_date = Column(Date, nullable=False)
    name = Column(String(50))
    recommendation_score = Column(Float, default=0.0)
    priority_score = Column(Float, default=0.0)
    overall_score = Column(Float)
    hit_streak_days = Column(Integer, default=0)
    miss_streak_days = Column(Integer, default=0)
    in_frontlist = Column(Boolean, default=False)
    llm_focus_level = Column(String(20), default='low')
    tracking_status = Column(String(20), default='shadow')
    source_tag = Column(String(50), default='今日Top3')
    is_repeat_pick = Column(Boolean, default=False)
    setup_type = Column(String(50))
    risk_level = Column(String(20))
    recommendation = Column(String(50))
    position_status = Column(String(20))
    last_frontlist_date = Column(Date)
    times_entered_frontlist = Column(Integer, default=0)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    news_score = Column(Float)
    base_score = Column(Float)
    sentiment_adjustment = Column(Float)
    news_adjustment = Column(Float)
    score_model = Column(String(50))
    close = Column(Float)
    pct_change = Column(Float)
    volume_ratio = Column(Float)
    turnover_rate = Column(Float)
    ma20 = Column(Float)
    strategy_count = Column(Integer, default=0)
    divergence_score = Column(Float)
    strategy_consistency_label = Column(String(50))
    news_mentioned = Column(Boolean, default=False)
    industry = Column(String(50))
    industry_heat_score = Column(Float)
    industry_flow_bias = Column(String(20))
    distribution_risk_score = Column(Float)
    distribution_risk_flags = Column(JSON)
    moneyflow_3d_value = Column(Float)
    recent_large_order_net_inflow = Column(Float)
    recent_super_large_order_net_inflow = Column(Float)
    turnover_spike_ratio = Column(Float)
    recent_runup_5d = Column(Float)
    continuation_bias_score = Column(Float)
    continuation_positive_flags = Column(JSON)
    continuation_negative_flags = Column(JSON)
    top3_risk_penalty = Column(Float)
    short_term_contradiction_penalty = Column(Float)
    final_display_recommendation_score = Column(Float)
    top3_status = Column(String(20), default='normal')
    top3_reason = Column(Text)
    late_stage_momentum_flag = Column(Boolean, default=False)
    candidate_risk_blocked = Column(Boolean, default=False)
    top3_extreme_risk_blocked = Column(Boolean, default=False)
    top3_extreme_risk_reason = Column(Text)
    ai_confidence = Column(Float)
    display_confidence = Column(Float)
    technical_signal = Column(String(200))
    summary = Column(Text)
    recommendation_text = Column(Text)
    entry_price = Column(Float)
    fundamental_bonus = Column(Float)
    fundamental_bonus_breakdown = Column(JSON)
    recommend_rank = Column(Integer)
    frontlist_rank = Column(Integer)
    rerank_pool_rank = Column(Integer)
    rerank_model_score = Column(Float)
    rerank_rule_score = Column(Float)
    rerank_blend_score = Column(Float)
    rerank_rule_weight = Column(Float)
    rerank_model_target = Column(String(50))
    rerank_selected_for_llm = Column(Boolean, default=False)
    selection_stage = Column(String(50))
    selection_reason = Column(Text)
    selection_reason_components = Column(JSON)
    structured_rank_score = Column(Float)
    structured_rank_position = Column(Integer)
    previous_recommendation_score = Column(Float)
    previous_overall_score = Column(Float)
    previous_confidence = Column(Float)
    score_change = Column(Float)
    today_present = Column(Boolean, default=True)
    absence_reason = Column(Text)
    action_plan = Column(JSON)
    review_status = Column(String(20))
    yesterday_conclusion = Column(Text)
    today_verdict = Column(Text)
    miss_reason_candidates = Column(JSON)
    missing_factor_candidates = Column(JSON)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_recommendation_pool_trade_date', 'trade_date'),
        Index('idx_recommendation_pool_frontlist', 'trade_date', 'in_frontlist'),
        Index('idx_recommendation_pool_priority', 'trade_date', 'priority_score'),
        Index('idx_recommendation_pool_code_date', 'ts_code', 'trade_date'),
    )


class RecommendationPerformance(Base):
    """推荐表现回填"""
    __tablename__ = 'recommendation_performances'

    id = Column(Integer, primary_key=True)
    recommendation_item_id = Column(Integer, ForeignKey('recommendation_items.id'), unique=True, nullable=False)
    entry_price = Column(Float)
    latest_price = Column(Float)
    return_1d = Column(Float)
    return_3d = Column(Float)
    return_5d = Column(Float)
    return_10d = Column(Float)
    max_drawdown_10d = Column(Float)
    benchmark_code = Column(String(20), default='000300.SH')
    benchmark_return_5d = Column(Float)
    vs_benchmark_5d = Column(Float)
    hit_5d = Column(Boolean)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    recommendation_item = relationship("RecommendationItem", back_populates="performance")

    __table_args__ = (
        Index('idx_recommendation_performance_updated_at', 'updated_at'),
    )


class ShortTermTrainingSampleRecord(Base):
    """短线训练样本"""
    __tablename__ = 'short_term_training_samples'

    id = Column(Integer, primary_key=True)
    feature_schema_version = Column(String(20), nullable=False, default='v1')
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    name = Column(String(50))
    source_tag = Column(String(50))
    in_frontlist = Column(Boolean, default=False)
    recommend_rank = Column(Integer)
    strategy_count = Column(Integer, default=0)
    is_repeat_pick = Column(Boolean, default=False)
    news_mentioned = Column(Boolean, default=False)
    technical_signal = Column(String(200))

    entry_price = Column(Float)
    close = Column(Float)
    pct_change = Column(Float)
    volume_ratio = Column(Float)
    turnover_rate = Column(Float)
    recommendation_score = Column(Float)
    overall_score = Column(Float)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    sentiment_score = Column(Float)
    news_score = Column(Float)
    base_score = Column(Float)
    sentiment_adjustment = Column(Float)
    news_adjustment = Column(Float)

    industry = Column(String(50))
    industry_heat_score = Column(Float)
    industry_flow_bias = Column(String(20))

    distribution_risk_score = Column(Float)
    distribution_risk_flags = Column(JSON)
    moneyflow_3d_value = Column(Float)
    recent_large_order_net_inflow = Column(Float)
    recent_super_large_order_net_inflow = Column(Float)
    turnover_spike_ratio = Column(Float)
    recent_runup_5d = Column(Float)
    continuation_bias_score = Column(Float)
    continuation_positive_flags = Column(JSON)
    continuation_negative_flags = Column(JSON)
    top3_risk_penalty = Column(Float)
    short_term_contradiction_penalty = Column(Float)
    late_stage_momentum_flag = Column(Boolean, default=False)
    candidate_risk_blocked = Column(Boolean, default=False)

    previous_recommendation_score = Column(Float)
    previous_overall_score = Column(Float)
    score_change = Column(Float)

    action_plan = Column(JSON)

    return_1d = Column(Float)
    return_3d = Column(Float)
    return_5d = Column(Float)
    return_10d = Column(Float)
    max_drawdown_10d = Column(Float)
    benchmark_return_5d = Column(Float)
    vs_benchmark_5d = Column(Float)
    label_up_1d = Column(Boolean)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_short_term_training_trade_date', 'trade_date'),
        Index('idx_short_term_training_code_date', 'ts_code', 'trade_date', unique=True),
        Index('idx_short_term_training_schema_date', 'feature_schema_version', 'trade_date'),
    )


class MarketTradeCalendar(Base):
    """原始交易日历"""
    __tablename__ = 'market_trade_calendar'

    trade_date = Column(Date, primary_key=True)
    exchange = Column(String(20), nullable=False, default='SSE')
    is_open = Column(Boolean, nullable=False, default=True)
    pretrade_date = Column(Date)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_trade_calendar_exchange_date', 'exchange', 'trade_date'),
    )


class MarketDaily(Base):
    """原始日线行情"""
    __tablename__ = 'market_daily'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    pre_close = Column(Float)
    change = Column(Float)
    pct_chg = Column(Float)
    vol = Column(Float)
    amount = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_daily_trade_date_ts_code', 'trade_date', 'ts_code', unique=True),
        Index('idx_market_daily_ts_code_trade_date', 'ts_code', 'trade_date'),
        Index('idx_market_daily_trade_date', 'trade_date'),
    )


class MarketDailyBasic(Base):
    """原始日线基础指标"""
    __tablename__ = 'market_daily_basic'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    turnover_rate = Column(Float)
    turnover_rate_f = Column(Float)
    volume_ratio = Column(Float)
    pe = Column(Float)
    pe_ttm = Column(Float)
    pb = Column(Float)
    ps = Column(Float)
    ps_ttm = Column(Float)
    dv_ratio = Column(Float)
    dv_ttm = Column(Float)
    total_share = Column(Float)
    float_share = Column(Float)
    free_share = Column(Float)
    total_mv = Column(Float)
    circ_mv = Column(Float)
    close = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_daily_basic_trade_date_ts_code', 'trade_date', 'ts_code', unique=True),
        Index('idx_market_daily_basic_ts_code_trade_date', 'ts_code', 'trade_date'),
        Index('idx_market_daily_basic_trade_date', 'trade_date'),
    )


class MarketAdjFactor(Base):
    """原始复权因子"""
    __tablename__ = 'market_adj_factor'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    adj_factor = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_adj_factor_trade_date_ts_code', 'trade_date', 'ts_code', unique=True),
        Index('idx_market_adj_factor_ts_code_trade_date', 'ts_code', 'trade_date'),
        Index('idx_market_adj_factor_trade_date', 'trade_date'),
    )


class MarketStockBasic(Base):
    """股票静态基础信息"""
    __tablename__ = 'market_stock_basic'

    ts_code = Column(String(20), primary_key=True)
    symbol = Column(String(20))
    name = Column(String(50))
    area = Column(String(50))
    industry = Column(String(50))
    market = Column(String(50))
    list_date = Column(Date)
    delist_date = Column(Date)
    is_hs = Column(String(20))
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_stock_basic_name', 'name'),
        Index('idx_market_stock_basic_industry', 'industry'),
    )


class MarketMoneyflowDaily(Base):
    __tablename__ = 'market_moneyflow_daily'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    buy_sm_vol = Column(Float)
    buy_sm_amount = Column(Float)
    sell_sm_vol = Column(Float)
    sell_sm_amount = Column(Float)
    buy_md_vol = Column(Float)
    buy_md_amount = Column(Float)
    sell_md_vol = Column(Float)
    sell_md_amount = Column(Float)
    buy_lg_vol = Column(Float)
    buy_lg_amount = Column(Float)
    sell_lg_vol = Column(Float)
    sell_lg_amount = Column(Float)
    buy_elg_vol = Column(Float)
    buy_elg_amount = Column(Float)
    sell_elg_vol = Column(Float)
    sell_elg_amount = Column(Float)
    net_mf_vol = Column(Float)
    net_mf_amount = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_moneyflow_trade_date_ts_code', 'trade_date', 'ts_code', unique=True),
        Index('idx_market_moneyflow_ts_code_trade_date', 'ts_code', 'trade_date'),
    )


class MarketTopListDaily(Base):
    __tablename__ = 'market_top_list_daily'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    reason = Column(String(255), nullable=False, default='')
    name = Column(String(50))
    close = Column(Float)
    pct_change = Column(Float)
    turnover_rate = Column(Float)
    amount = Column(Float)
    l_sell = Column(Float)
    l_buy = Column(Float)
    l_amount = Column(Float)
    net_amount = Column(Float)
    net_rate = Column(Float)
    amount_rate = Column(Float)
    float_values = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_top_list_trade_date_ts_code_reason', 'trade_date', 'ts_code', 'reason', unique=True),
        Index('idx_market_top_list_trade_date', 'trade_date'),
    )


class MarketLimitListDaily(Base):
    __tablename__ = 'market_limit_list_daily'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    industry = Column(String(100))
    name = Column(String(50))
    close = Column(Float)
    pct_chg = Column(Float)
    amount = Column(Float)
    limit_amount = Column(Float)
    float_mv = Column(Float)
    total_mv = Column(Float)
    turnover_ratio = Column(Float)
    fd_amount = Column(Float)
    first_time = Column(String(20))
    last_time = Column(String(20))
    open_times = Column(Integer)
    up_stat = Column(String(20))
    limit_times = Column(Float)
    limit = Column(String(10))
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_limit_list_trade_date_ts_code', 'trade_date', 'ts_code', unique=True),
        Index('idx_market_limit_list_trade_date', 'trade_date'),
    )


class MarketIndustryMoneyflowDaily(Base):
    __tablename__ = 'market_industry_moneyflow_daily'

    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    ts_code = Column(String(20), nullable=False)
    industry = Column(String(100))
    lead_stock = Column(String(50))
    close = Column(Float)
    pct_change = Column(Float)
    company_num = Column(Integer)
    pct_change_stock = Column(Float)
    close_price = Column(Float)
    net_buy_amount = Column(Float)
    net_sell_amount = Column(Float)
    net_amount = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_industry_moneyflow_trade_date_ts_code', 'trade_date', 'ts_code', unique=True),
        Index('idx_market_industry_moneyflow_trade_date', 'trade_date'),
    )


class MarketMoneyflowMarketDaily(Base):
    __tablename__ = 'market_moneyflow_market_daily'

    trade_date = Column(Date, primary_key=True)
    close_sh = Column(Float)
    pct_change_sh = Column(Float)
    close_sz = Column(Float)
    pct_change_sz = Column(Float)
    net_amount = Column(Float)
    net_amount_rate = Column(Float)
    buy_elg_amount = Column(Float)
    buy_elg_amount_rate = Column(Float)
    buy_lg_amount = Column(Float)
    buy_lg_amount_rate = Column(Float)
    buy_md_amount = Column(Float)
    buy_md_amount_rate = Column(Float)
    buy_sm_amount = Column(Float)
    buy_sm_amount_rate = Column(Float)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

    __table_args__ = (
        Index('idx_market_moneyflow_market_trade_date', 'trade_date'),
    )


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = "sqlite:///octts_screening.db"

        if database_url.startswith("sqlite"):
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=False
            )
        else:
            self.engine = create_engine(
                database_url,
                pool_size=10,
                pool_recycle=3600,
                echo=False
            )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        Base.metadata.create_all(bind=self.engine)
        self._migrate_sqlite_schema()

    def get_session(self):
        return self.SessionLocal()

    def _migrate_sqlite_schema(self) -> None:
        if self.engine.dialect.name != 'sqlite':
            return

        table_migrations = {
            'recommendation_pool_states': {
                'source_tag': "ALTER TABLE recommendation_pool_states ADD COLUMN source_tag VARCHAR(50) DEFAULT '今日Top3'",
                'is_repeat_pick': "ALTER TABLE recommendation_pool_states ADD COLUMN is_repeat_pick BOOLEAN DEFAULT 0",
                'setup_type': "ALTER TABLE recommendation_pool_states ADD COLUMN setup_type VARCHAR(50)",
                'risk_level': "ALTER TABLE recommendation_pool_states ADD COLUMN risk_level VARCHAR(20)",
                'recommendation': "ALTER TABLE recommendation_pool_states ADD COLUMN recommendation VARCHAR(50)",
                'position_status': "ALTER TABLE recommendation_pool_states ADD COLUMN position_status VARCHAR(20)",
                'last_frontlist_date': "ALTER TABLE recommendation_pool_states ADD COLUMN last_frontlist_date DATE",
                'times_entered_frontlist': "ALTER TABLE recommendation_pool_states ADD COLUMN times_entered_frontlist INTEGER DEFAULT 0",
                'technical_score': "ALTER TABLE recommendation_pool_states ADD COLUMN technical_score FLOAT",
                'fundamental_score': "ALTER TABLE recommendation_pool_states ADD COLUMN fundamental_score FLOAT",
                'sentiment_score': "ALTER TABLE recommendation_pool_states ADD COLUMN sentiment_score FLOAT",
                'news_score': "ALTER TABLE recommendation_pool_states ADD COLUMN news_score FLOAT",
                'base_score': "ALTER TABLE recommendation_pool_states ADD COLUMN base_score FLOAT",
                'sentiment_adjustment': "ALTER TABLE recommendation_pool_states ADD COLUMN sentiment_adjustment FLOAT",
                'news_adjustment': "ALTER TABLE recommendation_pool_states ADD COLUMN news_adjustment FLOAT",
                'score_model': "ALTER TABLE recommendation_pool_states ADD COLUMN score_model VARCHAR(50)",
                'overall_score': "ALTER TABLE recommendation_pool_states ADD COLUMN overall_score FLOAT",
                'close': "ALTER TABLE recommendation_pool_states ADD COLUMN close FLOAT",
                'pct_change': "ALTER TABLE recommendation_pool_states ADD COLUMN pct_change FLOAT",
                'volume_ratio': "ALTER TABLE recommendation_pool_states ADD COLUMN volume_ratio FLOAT",
                'turnover_rate': "ALTER TABLE recommendation_pool_states ADD COLUMN turnover_rate FLOAT",
                'ma20': "ALTER TABLE recommendation_pool_states ADD COLUMN ma20 FLOAT",
                'strategy_count': "ALTER TABLE recommendation_pool_states ADD COLUMN strategy_count INTEGER DEFAULT 0",
                'divergence_score': "ALTER TABLE recommendation_pool_states ADD COLUMN divergence_score FLOAT",
                'strategy_consistency_label': "ALTER TABLE recommendation_pool_states ADD COLUMN strategy_consistency_label VARCHAR(50)",
                'news_mentioned': "ALTER TABLE recommendation_pool_states ADD COLUMN news_mentioned BOOLEAN DEFAULT 0",
                'industry': "ALTER TABLE recommendation_pool_states ADD COLUMN industry VARCHAR(50)",
                'industry_heat_score': "ALTER TABLE recommendation_pool_states ADD COLUMN industry_heat_score FLOAT",
                'industry_flow_bias': "ALTER TABLE recommendation_pool_states ADD COLUMN industry_flow_bias VARCHAR(20)",
                'distribution_risk_score': "ALTER TABLE recommendation_pool_states ADD COLUMN distribution_risk_score FLOAT",
                'distribution_risk_flags': "ALTER TABLE recommendation_pool_states ADD COLUMN distribution_risk_flags JSON",
                'moneyflow_3d_value': "ALTER TABLE recommendation_pool_states ADD COLUMN moneyflow_3d_value FLOAT",
                'recent_large_order_net_inflow': "ALTER TABLE recommendation_pool_states ADD COLUMN recent_large_order_net_inflow FLOAT",
                'recent_super_large_order_net_inflow': "ALTER TABLE recommendation_pool_states ADD COLUMN recent_super_large_order_net_inflow FLOAT",
                'turnover_spike_ratio': "ALTER TABLE recommendation_pool_states ADD COLUMN turnover_spike_ratio FLOAT",
                'recent_runup_5d': "ALTER TABLE recommendation_pool_states ADD COLUMN recent_runup_5d FLOAT",
                'continuation_bias_score': "ALTER TABLE recommendation_pool_states ADD COLUMN continuation_bias_score FLOAT",
                'continuation_positive_flags': "ALTER TABLE recommendation_pool_states ADD COLUMN continuation_positive_flags JSON",
                'continuation_negative_flags': "ALTER TABLE recommendation_pool_states ADD COLUMN continuation_negative_flags JSON",
                'top3_risk_penalty': "ALTER TABLE recommendation_pool_states ADD COLUMN top3_risk_penalty FLOAT",
                'short_term_contradiction_penalty': "ALTER TABLE recommendation_pool_states ADD COLUMN short_term_contradiction_penalty FLOAT",
                'final_display_recommendation_score': "ALTER TABLE recommendation_pool_states ADD COLUMN final_display_recommendation_score FLOAT",
                'top3_status': "ALTER TABLE recommendation_pool_states ADD COLUMN top3_status VARCHAR(20) DEFAULT 'normal'",
                'top3_reason': "ALTER TABLE recommendation_pool_states ADD COLUMN top3_reason TEXT",
                'late_stage_momentum_flag': "ALTER TABLE recommendation_pool_states ADD COLUMN late_stage_momentum_flag BOOLEAN DEFAULT 0",
                'candidate_risk_blocked': "ALTER TABLE recommendation_pool_states ADD COLUMN candidate_risk_blocked BOOLEAN DEFAULT 0",
                'top3_extreme_risk_blocked': "ALTER TABLE recommendation_pool_states ADD COLUMN top3_extreme_risk_blocked BOOLEAN DEFAULT 0",
                'top3_extreme_risk_reason': "ALTER TABLE recommendation_pool_states ADD COLUMN top3_extreme_risk_reason TEXT",
                'ai_confidence': "ALTER TABLE recommendation_pool_states ADD COLUMN ai_confidence FLOAT",
                'display_confidence': "ALTER TABLE recommendation_pool_states ADD COLUMN display_confidence FLOAT",
                'technical_signal': "ALTER TABLE recommendation_pool_states ADD COLUMN technical_signal VARCHAR(200)",
                'summary': "ALTER TABLE recommendation_pool_states ADD COLUMN summary TEXT",
                'recommendation_text': "ALTER TABLE recommendation_pool_states ADD COLUMN recommendation_text TEXT",
                'entry_price': "ALTER TABLE recommendation_pool_states ADD COLUMN entry_price FLOAT",
                'fundamental_bonus': "ALTER TABLE recommendation_pool_states ADD COLUMN fundamental_bonus FLOAT",
                'fundamental_bonus_breakdown': "ALTER TABLE recommendation_pool_states ADD COLUMN fundamental_bonus_breakdown JSON",
                'recommend_rank': "ALTER TABLE recommendation_pool_states ADD COLUMN recommend_rank INTEGER",
                'frontlist_rank': "ALTER TABLE recommendation_pool_states ADD COLUMN frontlist_rank INTEGER",
                'rerank_pool_rank': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_pool_rank INTEGER",
                'rerank_model_score': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_model_score FLOAT",
                'rerank_rule_score': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_rule_score FLOAT",
                'rerank_blend_score': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_blend_score FLOAT",
                'rerank_rule_weight': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_rule_weight FLOAT",
                'rerank_model_target': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_model_target VARCHAR(50)",
                'rerank_selected_for_llm': "ALTER TABLE recommendation_pool_states ADD COLUMN rerank_selected_for_llm BOOLEAN DEFAULT 0",
                'selection_stage': "ALTER TABLE recommendation_pool_states ADD COLUMN selection_stage VARCHAR(50)",
                'selection_reason': "ALTER TABLE recommendation_pool_states ADD COLUMN selection_reason TEXT",
                'selection_reason_components': "ALTER TABLE recommendation_pool_states ADD COLUMN selection_reason_components JSON",
                'structured_rank_score': "ALTER TABLE recommendation_pool_states ADD COLUMN structured_rank_score FLOAT",
                'structured_rank_position': "ALTER TABLE recommendation_pool_states ADD COLUMN structured_rank_position INTEGER",
                'previous_recommendation_score': "ALTER TABLE recommendation_pool_states ADD COLUMN previous_recommendation_score FLOAT",
                'previous_overall_score': "ALTER TABLE recommendation_pool_states ADD COLUMN previous_overall_score FLOAT",
                'previous_confidence': "ALTER TABLE recommendation_pool_states ADD COLUMN previous_confidence FLOAT",
                'score_change': "ALTER TABLE recommendation_pool_states ADD COLUMN score_change FLOAT",
                'today_present': "ALTER TABLE recommendation_pool_states ADD COLUMN today_present BOOLEAN DEFAULT 1",
                'absence_reason': "ALTER TABLE recommendation_pool_states ADD COLUMN absence_reason TEXT",
                'action_plan': "ALTER TABLE recommendation_pool_states ADD COLUMN action_plan JSON",
                'review_status': "ALTER TABLE recommendation_pool_states ADD COLUMN review_status VARCHAR(20)",
                'yesterday_conclusion': "ALTER TABLE recommendation_pool_states ADD COLUMN yesterday_conclusion TEXT",
                'today_verdict': "ALTER TABLE recommendation_pool_states ADD COLUMN today_verdict TEXT",
                'miss_reason_candidates': "ALTER TABLE recommendation_pool_states ADD COLUMN miss_reason_candidates JSON",
                'missing_factor_candidates': "ALTER TABLE recommendation_pool_states ADD COLUMN missing_factor_candidates JSON",
                'updated_at': "ALTER TABLE recommendation_pool_states ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            },
            'recommendation_items': {
                'source_tag': "ALTER TABLE recommendation_items ADD COLUMN source_tag VARCHAR(50) DEFAULT '今日Top3'",
                'is_repeat_pick': "ALTER TABLE recommendation_items ADD COLUMN is_repeat_pick BOOLEAN DEFAULT 0",
                'score_mode': "ALTER TABLE recommendation_items ADD COLUMN score_mode VARCHAR(50)",
                'rerank_pool_rank': "ALTER TABLE recommendation_items ADD COLUMN rerank_pool_rank INTEGER",
                'rerank_blend_score': "ALTER TABLE recommendation_items ADD COLUMN rerank_blend_score FLOAT",
                'rerank_model_score': "ALTER TABLE recommendation_items ADD COLUMN rerank_model_score FLOAT",
                'rerank_rule_score': "ALTER TABLE recommendation_items ADD COLUMN rerank_rule_score FLOAT",
                'rerank_rule_weight': "ALTER TABLE recommendation_items ADD COLUMN rerank_rule_weight FLOAT",
                'rerank_model_target': "ALTER TABLE recommendation_items ADD COLUMN rerank_model_target VARCHAR(50)",
                'selection_stage': "ALTER TABLE recommendation_items ADD COLUMN selection_stage VARCHAR(50)",
                'selection_reason': "ALTER TABLE recommendation_items ADD COLUMN selection_reason TEXT",
                'selection_reason_components': "ALTER TABLE recommendation_items ADD COLUMN selection_reason_components JSON",
                'structured_rank_score': "ALTER TABLE recommendation_items ADD COLUMN structured_rank_score FLOAT",
                'structured_rank_position': "ALTER TABLE recommendation_items ADD COLUMN structured_rank_position INTEGER",
            },
            'short_term_training_samples': {
                'recent_large_order_net_inflow': "ALTER TABLE short_term_training_samples ADD COLUMN recent_large_order_net_inflow FLOAT",
                'recent_super_large_order_net_inflow': "ALTER TABLE short_term_training_samples ADD COLUMN recent_super_large_order_net_inflow FLOAT",
            },
        }

        inspector = inspect(self.engine)
        table_names = set(inspector.get_table_names())

        with self.engine.begin() as connection:
            for table_name, required_columns in table_migrations.items():
                if table_name not in table_names:
                    continue
                existing_columns = {column['name'] for column in inspector.get_columns(table_name)}
                for column_name, ddl in required_columns.items():
                    if column_name in existing_columns:
                        continue
                    connection.execute(text(ddl))
                    existing_columns.add(column_name)

    def upsert_market_trade_calendar(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        exchange: str = 'SSE',
        force_refresh: bool = False,
    ) -> int:
        payload = [self._build_market_trade_calendar_payload(row, exchange=exchange) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketTradeCalendar,
            payload,
            conflict_columns=['trade_date'],
            update_columns=['exchange', 'is_open', 'pretrade_date', 'updated_at'],
            force_refresh=force_refresh,
        )

    def upsert_market_daily(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_daily_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketDaily,
            payload,
            conflict_columns=['trade_date', 'ts_code'],
            update_columns=['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'updated_at'],
            force_refresh=force_refresh,
        )

    def upsert_market_daily_basic(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_daily_basic_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketDailyBasic,
            payload,
            conflict_columns=['trade_date', 'ts_code'],
            update_columns=[
                'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'close', 'updated_at'
            ],
            force_refresh=force_refresh,
        )

    def upsert_market_adj_factor(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_adj_factor_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketAdjFactor,
            payload,
            conflict_columns=['trade_date', 'ts_code'],
            update_columns=['adj_factor', 'updated_at'],
            force_refresh=force_refresh,
        )

    def upsert_market_moneyflow_daily(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_moneyflow_daily_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketMoneyflowDaily,
            payload,
            conflict_columns=['trade_date', 'ts_code'],
            update_columns=[
                'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount',
                'sell_md_vol', 'sell_md_amount', 'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
                'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount', 'net_mf_vol', 'net_mf_amount', 'updated_at'
            ],
            force_refresh=force_refresh,
        )

    def upsert_market_top_list_daily(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_top_list_daily_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketTopListDaily,
            payload,
            conflict_columns=['trade_date', 'ts_code', 'reason'],
            update_columns=['name', 'close', 'pct_change', 'turnover_rate', 'amount', 'l_sell', 'l_buy', 'l_amount', 'net_amount', 'net_rate', 'amount_rate', 'float_values', 'updated_at'],
            force_refresh=force_refresh,
        )

    def upsert_market_limit_list_daily(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_limit_list_daily_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketLimitListDaily,
            payload,
            conflict_columns=['trade_date', 'ts_code'],
            update_columns=['industry', 'name', 'close', 'pct_chg', 'amount', 'limit_amount', 'float_mv', 'total_mv', 'turnover_ratio', 'fd_amount', 'first_time', 'last_time', 'open_times', 'up_stat', 'limit_times', 'limit', 'updated_at'],
            force_refresh=force_refresh,
        )

    def upsert_market_industry_moneyflow_daily(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_industry_moneyflow_daily_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketIndustryMoneyflowDaily,
            payload,
            conflict_columns=['trade_date', 'ts_code'],
            update_columns=['industry', 'lead_stock', 'close', 'pct_change', 'company_num', 'pct_change_stock', 'close_price', 'net_buy_amount', 'net_sell_amount', 'net_amount', 'updated_at'],
            force_refresh=force_refresh,
        )

    def upsert_market_moneyflow_market_daily(self, rows: Iterable[Dict[str, Any]], *, force_refresh: bool = False) -> int:
        payload = [self._build_market_moneyflow_market_daily_payload(row) for row in rows]
        payload = [row for row in payload if row is not None]
        return self._execute_market_upsert(
            MarketMoneyflowMarketDaily,
            payload,
            conflict_columns=['trade_date'],
            update_columns=['close_sh', 'pct_change_sh', 'close_sz', 'pct_change_sz', 'net_amount', 'net_amount_rate', 'buy_elg_amount', 'buy_elg_amount_rate', 'buy_lg_amount', 'buy_lg_amount_rate', 'buy_md_amount', 'buy_md_amount_rate', 'buy_sm_amount', 'buy_sm_amount_rate', 'updated_at'],
            force_refresh=force_refresh,
        )

    def has_market_trade_calendar(self, *, start_date: date, end_date: date, exchange: str = 'SSE') -> bool:
        session = self.get_session()
        try:
            row_count = session.query(func.count(MarketTradeCalendar.trade_date)).filter(
                MarketTradeCalendar.exchange == exchange,
                MarketTradeCalendar.trade_date >= start_date,
                MarketTradeCalendar.trade_date <= end_date,
                MarketTradeCalendar.is_open.is_(True),
            ).scalar() or 0
            return row_count > 0
        finally:
            session.close()

    def has_market_data_for_trade_date(self, *, model, trade_date: date) -> bool:
        session = self.get_session()
        try:
            record = session.execute(select(model).where(model.trade_date == trade_date).limit(1)).scalar_one_or_none()
            return record is not None
        finally:
            session.close()

    def _execute_market_upsert(
        self,
        model,
        payload: List[Dict[str, Any]],
        *,
        conflict_columns: List[str],
        update_columns: List[str],
        force_refresh: bool,
    ) -> int:
        if not payload:
            return 0

        insert_stmt = sqlite_insert(model).values(payload)
        if force_refresh:
            update_mapping = {column: getattr(insert_stmt.excluded, column) for column in update_columns}
            statement = insert_stmt.on_conflict_do_update(index_elements=conflict_columns, set_=update_mapping)
        else:
            statement = insert_stmt.on_conflict_do_nothing(index_elements=conflict_columns)

        with self.engine.begin() as connection:
            result = connection.execute(statement)
        return int(result.rowcount or 0)

    @staticmethod
    def _build_market_trade_calendar_payload(row: Dict[str, Any], *, exchange: str) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('cal_date') or row.get('trade_date'))
        if trade_date_value is None:
            return None
        is_open_raw = row.get('is_open')
        is_open = bool(int(is_open_raw)) if isinstance(is_open_raw, str) and is_open_raw.isdigit() else bool(is_open_raw)
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'exchange': str(row.get('exchange') or exchange or 'SSE'),
            'is_open': is_open,
            'pretrade_date': DatabaseManager._parse_date_value(row.get('pretrade_date')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_daily_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'open': DatabaseManager._safe_float_value(row.get('open')),
            'high': DatabaseManager._safe_float_value(row.get('high')),
            'low': DatabaseManager._safe_float_value(row.get('low')),
            'close': DatabaseManager._safe_float_value(row.get('close')),
            'pre_close': DatabaseManager._safe_float_value(row.get('pre_close')),
            'change': DatabaseManager._safe_float_value(row.get('change')),
            'pct_chg': DatabaseManager._safe_float_value(row.get('pct_chg')),
            'vol': DatabaseManager._safe_float_value(row.get('vol')),
            'amount': DatabaseManager._safe_float_value(row.get('amount')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_daily_basic_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'turnover_rate': DatabaseManager._safe_float_value(row.get('turnover_rate')),
            'turnover_rate_f': DatabaseManager._safe_float_value(row.get('turnover_rate_f')),
            'volume_ratio': DatabaseManager._safe_float_value(row.get('volume_ratio')),
            'pe': DatabaseManager._safe_float_value(row.get('pe')),
            'pe_ttm': DatabaseManager._safe_float_value(row.get('pe_ttm')),
            'pb': DatabaseManager._safe_float_value(row.get('pb')),
            'ps': DatabaseManager._safe_float_value(row.get('ps')),
            'ps_ttm': DatabaseManager._safe_float_value(row.get('ps_ttm')),
            'dv_ratio': DatabaseManager._safe_float_value(row.get('dv_ratio')),
            'dv_ttm': DatabaseManager._safe_float_value(row.get('dv_ttm')),
            'total_share': DatabaseManager._safe_float_value(row.get('total_share')),
            'float_share': DatabaseManager._safe_float_value(row.get('float_share')),
            'free_share': DatabaseManager._safe_float_value(row.get('free_share')),
            'total_mv': DatabaseManager._safe_float_value(row.get('total_mv')),
            'circ_mv': DatabaseManager._safe_float_value(row.get('circ_mv')),
            'close': DatabaseManager._safe_float_value(row.get('close')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_adj_factor_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'adj_factor': DatabaseManager._safe_float_value(row.get('adj_factor')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_moneyflow_daily_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'buy_sm_vol': DatabaseManager._safe_float_value(row.get('buy_sm_vol')),
            'buy_sm_amount': DatabaseManager._safe_float_value(row.get('buy_sm_amount')),
            'sell_sm_vol': DatabaseManager._safe_float_value(row.get('sell_sm_vol')),
            'sell_sm_amount': DatabaseManager._safe_float_value(row.get('sell_sm_amount')),
            'buy_md_vol': DatabaseManager._safe_float_value(row.get('buy_md_vol')),
            'buy_md_amount': DatabaseManager._safe_float_value(row.get('buy_md_amount')),
            'sell_md_vol': DatabaseManager._safe_float_value(row.get('sell_md_vol')),
            'sell_md_amount': DatabaseManager._safe_float_value(row.get('sell_md_amount')),
            'buy_lg_vol': DatabaseManager._safe_float_value(row.get('buy_lg_vol')),
            'buy_lg_amount': DatabaseManager._safe_float_value(row.get('buy_lg_amount')),
            'sell_lg_vol': DatabaseManager._safe_float_value(row.get('sell_lg_vol')),
            'sell_lg_amount': DatabaseManager._safe_float_value(row.get('sell_lg_amount')),
            'buy_elg_vol': DatabaseManager._safe_float_value(row.get('buy_elg_vol')),
            'buy_elg_amount': DatabaseManager._safe_float_value(row.get('buy_elg_amount')),
            'sell_elg_vol': DatabaseManager._safe_float_value(row.get('sell_elg_vol')),
            'sell_elg_amount': DatabaseManager._safe_float_value(row.get('sell_elg_amount')),
            'net_mf_vol': DatabaseManager._safe_float_value(row.get('net_mf_vol')),
            'net_mf_amount': DatabaseManager._safe_float_value(row.get('net_mf_amount')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_top_list_daily_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        reason = str(row.get('reason') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'reason': reason,
            'name': str(row.get('name') or '').strip() or None,
            'close': DatabaseManager._safe_float_value(row.get('close')),
            'pct_change': DatabaseManager._safe_float_value(row.get('pct_change')),
            'turnover_rate': DatabaseManager._safe_float_value(row.get('turnover_rate')),
            'amount': DatabaseManager._safe_float_value(row.get('amount')),
            'l_sell': DatabaseManager._safe_float_value(row.get('l_sell')),
            'l_buy': DatabaseManager._safe_float_value(row.get('l_buy')),
            'l_amount': DatabaseManager._safe_float_value(row.get('l_amount')),
            'net_amount': DatabaseManager._safe_float_value(row.get('net_amount')),
            'net_rate': DatabaseManager._safe_float_value(row.get('net_rate')),
            'amount_rate': DatabaseManager._safe_float_value(row.get('amount_rate')),
            'float_values': DatabaseManager._safe_float_value(row.get('float_values')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_limit_list_daily_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        open_times_value = row.get('open_times')
        try:
            open_times = int(float(open_times_value)) if open_times_value not in (None, '') else None
        except (TypeError, ValueError):
            open_times = None
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'industry': str(row.get('industry') or '').strip() or None,
            'name': str(row.get('name') or '').strip() or None,
            'close': DatabaseManager._safe_float_value(row.get('close')),
            'pct_chg': DatabaseManager._safe_float_value(row.get('pct_chg')),
            'amount': DatabaseManager._safe_float_value(row.get('amount')),
            'limit_amount': DatabaseManager._safe_float_value(row.get('limit_amount')),
            'float_mv': DatabaseManager._safe_float_value(row.get('float_mv')),
            'total_mv': DatabaseManager._safe_float_value(row.get('total_mv')),
            'turnover_ratio': DatabaseManager._safe_float_value(row.get('turnover_ratio')),
            'fd_amount': DatabaseManager._safe_float_value(row.get('fd_amount')),
            'first_time': str(row.get('first_time') or '').strip() or None,
            'last_time': str(row.get('last_time') or '').strip() or None,
            'open_times': open_times,
            'up_stat': str(row.get('up_stat') or '').strip() or None,
            'limit_times': DatabaseManager._safe_float_value(row.get('limit_times')),
            'limit': str(row.get('limit') or '').strip() or None,
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_industry_moneyflow_daily_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        ts_code = str(row.get('ts_code') or '').strip()
        if trade_date_value is None or not ts_code:
            return None
        now = datetime.now()
        company_num_value = row.get('company_num')
        try:
            company_num = int(float(company_num_value)) if company_num_value not in (None, '') else None
        except (TypeError, ValueError):
            company_num = None
        return {
            'trade_date': trade_date_value,
            'ts_code': ts_code,
            'industry': str(row.get('industry') or '').strip() or None,
            'lead_stock': str(row.get('lead_stock') or '').strip() or None,
            'close': DatabaseManager._safe_float_value(row.get('close')),
            'pct_change': DatabaseManager._safe_float_value(row.get('pct_change')),
            'company_num': company_num,
            'pct_change_stock': DatabaseManager._safe_float_value(row.get('pct_change_stock')),
            'close_price': DatabaseManager._safe_float_value(row.get('close_price')),
            'net_buy_amount': DatabaseManager._safe_float_value(row.get('net_buy_amount')),
            'net_sell_amount': DatabaseManager._safe_float_value(row.get('net_sell_amount')),
            'net_amount': DatabaseManager._safe_float_value(row.get('net_amount')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _build_market_moneyflow_market_daily_payload(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date_value = DatabaseManager._parse_date_value(row.get('trade_date'))
        if trade_date_value is None:
            return None
        now = datetime.now()
        return {
            'trade_date': trade_date_value,
            'close_sh': DatabaseManager._safe_float_value(row.get('close_sh')),
            'pct_change_sh': DatabaseManager._safe_float_value(row.get('pct_change_sh')),
            'close_sz': DatabaseManager._safe_float_value(row.get('close_sz')),
            'pct_change_sz': DatabaseManager._safe_float_value(row.get('pct_change_sz')),
            'net_amount': DatabaseManager._safe_float_value(row.get('net_amount')),
            'net_amount_rate': DatabaseManager._safe_float_value(row.get('net_amount_rate')),
            'buy_elg_amount': DatabaseManager._safe_float_value(row.get('buy_elg_amount')),
            'buy_elg_amount_rate': DatabaseManager._safe_float_value(row.get('buy_elg_amount_rate')),
            'buy_lg_amount': DatabaseManager._safe_float_value(row.get('buy_lg_amount')),
            'buy_lg_amount_rate': DatabaseManager._safe_float_value(row.get('buy_lg_amount_rate')),
            'buy_md_amount': DatabaseManager._safe_float_value(row.get('buy_md_amount')),
            'buy_md_amount_rate': DatabaseManager._safe_float_value(row.get('buy_md_amount_rate')),
            'buy_sm_amount': DatabaseManager._safe_float_value(row.get('buy_sm_amount')),
            'buy_sm_amount_rate': DatabaseManager._safe_float_value(row.get('buy_sm_amount_rate')),
            'created_at': now,
            'updated_at': now,
        }

    @staticmethod
    def _parse_date_value(value: Any) -> Optional[date]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text_value = str(value).strip()
        if not text_value:
            return None
        for fmt in ('%Y%m%d', '%Y-%m-%d'):
            try:
                return datetime.strptime(text_value, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_float_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:
            return None
        return numeric

    def save_screening_result(
        self,
        strategy_id: str,
        result: Any,
        ai_analysis: Optional[Dict[str, Any]] = None
    ):
        session = self.get_session()
        try:
            strategy = session.query(ScreeningStrategy).filter_by(
                strategy_id=strategy_id
            ).first()

            if not strategy:
                strategy = ScreeningStrategy(
                    strategy_id=strategy_id,
                    name=strategy_id,
                    criteria=result.criteria.model_dump()
                )
                session.add(strategy)
                session.flush()

            screening_run = ScreeningRun(
                run_id=result.screen_id,
                strategy_id=strategy.id,
                run_date=result.screen_time,
                total_stocks=result.total_count,
                execution_time=result.execution_time,
                status='completed'
            )
            session.add(screening_run)
            session.flush()

            for rank, stock in enumerate(result.stocks, 1):
                stock_result = StockScreeningResult(
                    screening_run_id=screening_run.id,
                    ts_code=stock.ts_code,
                    name=stock.name,
                    close=stock.close,
                    pct_change=stock.pct_change,
                    volume_ratio=stock.volume_ratio,
                    turnover_rate=stock.turnover_rate,
                    rsi=stock.rsi,
                    ma5=stock.ma5,
                    ma20=stock.ma20,
                    market_cap=stock.market_cap,
                    pe_ratio=stock.pe_ratio,
                    industry=stock.industry,
                    score=stock.score,
                    match_reasons=stock.match_reasons,
                    rank=rank
                )
                session.add(stock_result)
                session.flush()

                if ai_analysis and stock.ts_code in ai_analysis:
                    analysis = ai_analysis[stock.ts_code]
                    ai_result = StockAIAnalysis(
                        screening_result_id=stock_result.id,
                        ts_code=stock.ts_code,
                        analysis_date=datetime.now(),
                        technical_score=analysis.get('technical_score'),
                        fundamental_score=analysis.get('fundamental_score'),
                        sentiment_score=analysis.get('sentiment_score'),
                        news_score=analysis.get('news_score'),
                        overall_score=analysis.get('overall_score'),
                        overall_confidence=analysis.get('overall_confidence'),
                        recommendation=analysis.get('recommendation'),
                        ai_summary=analysis.get('summary')
                    )
                    session.add(ai_result)

            session.commit()

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_screening_history(
        self,
        strategy_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            query = session.query(ScreeningRun)

            if strategy_id:
                strategy = session.query(ScreeningStrategy).filter_by(
                    strategy_id=strategy_id
                ).first()
                if not strategy:
                    return []
                query = query.filter_by(strategy_id=strategy.id)

            if start_date:
                query = query.filter(ScreeningRun.run_date >= start_date)

            if end_date:
                query = query.filter(ScreeningRun.run_date <= end_date)

            runs = query.order_by(ScreeningRun.run_date.desc()).limit(limit).all()

            results = []
            for run in runs:
                results.append({
                    'run_id': run.run_id,
                    'strategy': run.strategy.name,
                    'run_date': run.run_date,
                    'total_stocks': run.total_stocks,
                    'execution_time': run.execution_time,
                    'status': run.status
                })

            return results

        finally:
            session.close()

    def get_stock_performance(
        self,
        ts_code: str,
        days: int = 30
    ) -> Dict[str, Any]:
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            results = session.query(StockScreeningResult).join(
                ScreeningRun
            ).filter(
                StockScreeningResult.ts_code == ts_code,
                ScreeningRun.run_date >= cutoff_date
            ).order_by(ScreeningRun.run_date.desc()).all()

            appearances = []
            scores = []
            ai_scores = []

            for result in results:
                appearances.append({
                    'date': result.screening_run.run_date,
                    'strategy': result.screening_run.strategy.name,
                    'rank': result.rank,
                    'score': result.score,
                    'pct_change': result.pct_change
                })

                if result.score:
                    scores.append(result.score)

                if result.ai_analysis and result.ai_analysis.overall_score:
                    ai_scores.append(result.ai_analysis.overall_score)

            return {
                'ts_code': ts_code,
                'total_appearances': len(appearances),
                'average_rank': sum(a['rank'] for a in appearances) / len(appearances) if appearances else 0,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'average_ai_score': sum(ai_scores) / len(ai_scores) if ai_scores else 0,
                'appearances': appearances
            }

        finally:
            session.close()

    def get_screening_result(self, run_id: str) -> Optional[ScreenResult]:
        session = self.get_session()
        try:
            run = session.query(ScreeningRun).filter_by(run_id=run_id).first()
            if run is None:
                return None

            ordered_results = sorted(
                run.stock_results,
                key=lambda item: (item.rank is None, item.rank if item.rank is not None else 0),
            )
            stocks = [
                StockScreenItem(
                    ts_code=item.ts_code,
                    name=item.name or "",
                    close=item.close or 0.0,
                    pct_change=item.pct_change,
                    volume_ratio=item.volume_ratio or 0.0,
                    turnover_rate=item.turnover_rate or 0.0,
                    rsi=item.rsi,
                    ma5=item.ma5,
                    ma10=None,
                    ma20=item.ma20,
                    ma60=None,
                    macd=None,
                    macd_signal=None,
                    macd_histogram=None,
                    price_position_20d=None,
                    trend_status=None,
                    momentum_status=None,
                    technical_score=item.score,
                    market_cap=item.market_cap,
                    pe_ratio=item.pe_ratio,
                    industry=item.industry,
                    score=item.score,
                    match_reasons=item.match_reasons or [],
                )
                for item in ordered_results
            ]

            strategy = run.strategy
            criteria_payload = strategy.criteria if strategy and strategy.criteria else {}
            return ScreenResult(
                screen_id=run.run_id,
                criteria=ScreenCriteria(**criteria_payload),
                stocks=stocks,
                total_count=run.total_stocks or len(stocks),
                screen_time=run.run_date,
                execution_time=run.execution_time or 0.0,
            )
        finally:
            session.close()

    def save_recommendation_run(
        self,
        *,
        run_id: str,
        trade_date: date,
        candidate_count: int,
        final_count: int,
        report_id: Optional[str],
        generated_at: Optional[datetime] = None,
        items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        session = self.get_session()
        try:
            recommendation_run = session.query(RecommendationRun).filter_by(trade_date=trade_date).first()
            generated_time = generated_at or datetime.now()
            if recommendation_run is None:
                recommendation_run = RecommendationRun(
                    run_id=run_id,
                    trade_date=trade_date,
                    generated_at=generated_time,
                    candidate_count=candidate_count,
                    final_count=final_count,
                    report_id=report_id,
                )
                session.add(recommendation_run)
                session.flush()
            else:
                recommendation_run.run_id = run_id
                recommendation_run.generated_at = generated_time
                recommendation_run.candidate_count = candidate_count
                recommendation_run.final_count = final_count
                recommendation_run.report_id = report_id
                session.query(RecommendationPerformance).filter(
                    RecommendationPerformance.recommendation_item_id.in_(
                        session.query(RecommendationItem.id).filter_by(run_id=recommendation_run.id)
                    )
                ).delete(synchronize_session=False)
                session.query(RecommendationItem).filter_by(run_id=recommendation_run.id).delete(synchronize_session=False)
                session.flush()

            created_items = []
            for item in items or []:
                tracking_status = item.get('tracking_status')
                raw_status = item.get('status')
                recommendation_item = RecommendationItem(
                    run_id=recommendation_run.id,
                    ts_code=item.get('ts_code'),
                    name=item.get('name'),
                    recommend_rank=item.get('recommend_rank'),
                    recommend_score=item.get('recommend_score'),
                    ai_confidence=item.get('ai_confidence'),
                    source_tag=item.get('source_tag') or '今日Top3',
                    is_repeat_pick=bool(item.get('is_repeat_pick', False)),
                    strategy_count=item.get('strategy_count', 0),
                    news_mentioned=bool(item.get('news_mentioned', False)),
                    technical_signal=item.get('technical_signal'),
                    recommendation_text=item.get('recommendation_text'),
                    status=self._normalize_recommendation_status(raw_status, tracking_status),
                    tracking_days=item.get('tracking_days', 0),
                    trade_date=item.get('trade_date', trade_date),
                    entry_price=item.get('entry_price'),
                    score_mode=item.get('score_mode'),
                    rerank_pool_rank=item.get('rerank_pool_rank'),
                    rerank_blend_score=item.get('rerank_blend_score'),
                    rerank_model_score=item.get('rerank_model_score'),
                    rerank_rule_score=item.get('rerank_rule_score'),
                    rerank_rule_weight=item.get('rerank_rule_weight'),
                    rerank_model_target=item.get('rerank_model_target'),
                    selection_stage=item.get('selection_stage'),
                    selection_reason=item.get('selection_reason'),
                    selection_reason_components=item.get('selection_reason_components') or {},
                    structured_rank_score=item.get('structured_rank_score'),
                    structured_rank_position=item.get('structured_rank_position'),
                    created_at=item.get('created_at') or recommendation_run.generated_at,
                )
                session.add(recommendation_item)
                session.flush()
                created_items.append({
                    'id': recommendation_item.id,
                    'ts_code': recommendation_item.ts_code,
                    'status': recommendation_item.status,
                })

            session.commit()
            return {
                'id': recommendation_run.id,
                'run_id': recommendation_run.run_id,
                'trade_date': recommendation_run.trade_date.isoformat(),
                'final_count': recommendation_run.final_count,
                'items': created_items,
            }
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_recommendation_pool_states(self, states: List[TrackedRecommendationState]) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            persisted: List[Dict[str, Any]] = []
            for state in states:
                record = session.query(RecommendationPoolState).filter_by(
                    ts_code=state.ts_code,
                    trade_date=state.trade_date,
                ).first()
                if record is None:
                    record = RecommendationPoolState(
                        ts_code=state.ts_code,
                        trade_date=state.trade_date,
                        created_at=datetime.now(),
                    )
                    session.add(record)

                record.name = state.name
                record.recommendation_score = state.recommendation_score
                record.priority_score = state.priority_score
                record.overall_score = state.overall_score
                record.hit_streak_days = state.hit_streak_days
                record.miss_streak_days = state.miss_streak_days
                record.in_frontlist = state.in_frontlist
                record.llm_focus_level = state.llm_focus_level
                record.tracking_status = state.tracking_status
                record.source_tag = state.source_tag
                record.is_repeat_pick = bool(state.is_repeat_pick)
                record.setup_type = state.setup_type
                record.risk_level = state.risk_level
                record.recommendation = state.recommendation
                record.position_status = state.position_status
                record.last_frontlist_date = state.last_frontlist_date
                record.times_entered_frontlist = state.times_entered_frontlist
                record.technical_score = state.technical_score
                record.fundamental_score = state.fundamental_score
                record.sentiment_score = state.sentiment_score
                record.news_score = state.news_score
                record.base_score = state.base_score
                record.sentiment_adjustment = state.sentiment_adjustment
                record.news_adjustment = state.news_adjustment
                record.score_model = state.score_model
                record.close = state.close
                record.pct_change = state.pct_change
                record.volume_ratio = state.volume_ratio
                record.turnover_rate = state.turnover_rate
                record.ma20 = getattr(state, 'ma20', None)
                record.strategy_count = state.strategy_count
                record.divergence_score = getattr(state, 'divergence_score', None)
                record.strategy_consistency_label = getattr(state, 'strategy_consistency_label', None)
                record.news_mentioned = bool(state.news_mentioned)
                record.industry = state.industry
                record.industry_heat_score = state.industry_heat_score
                record.industry_flow_bias = state.industry_flow_bias
                record.distribution_risk_score = state.distribution_risk_score
                record.distribution_risk_flags = list(state.distribution_risk_flags or [])
                record.moneyflow_3d_value = state.moneyflow_3d_value
                record.recent_large_order_net_inflow = getattr(state, 'recent_large_order_net_inflow', None)
                record.recent_super_large_order_net_inflow = getattr(state, 'recent_super_large_order_net_inflow', None)
                record.turnover_spike_ratio = state.turnover_spike_ratio
                record.recent_runup_5d = state.recent_runup_5d
                record.continuation_bias_score = state.continuation_bias_score
                record.continuation_positive_flags = list(state.continuation_positive_flags or [])
                record.continuation_negative_flags = list(state.continuation_negative_flags or [])
                record.top3_risk_penalty = state.top3_risk_penalty
                record.short_term_contradiction_penalty = state.short_term_contradiction_penalty
                record.final_display_recommendation_score = state.final_display_recommendation_score
                record.top3_status = getattr(state, 'top3_status', 'normal') or 'normal'
                record.top3_reason = getattr(state, 'top3_reason', None)
                record.late_stage_momentum_flag = bool(state.late_stage_momentum_flag)
                record.candidate_risk_blocked = bool(state.candidate_risk_blocked)
                record.top3_extreme_risk_blocked = bool(getattr(state, 'top3_extreme_risk_blocked', False))
                record.top3_extreme_risk_reason = getattr(state, 'top3_extreme_risk_reason', None)
                record.ai_confidence = state.ai_confidence
                record.display_confidence = state.display_confidence
                record.technical_signal = state.technical_signal
                record.summary = state.summary
                record.recommendation_text = state.recommendation_text
                record.entry_price = state.entry_price
                record.fundamental_bonus = getattr(state, 'fundamental_bonus', None)
                record.fundamental_bonus_breakdown = dict(getattr(state, 'fundamental_bonus_breakdown', {}) or {})
                record.recommend_rank = state.recommend_rank
                record.frontlist_rank = getattr(state, 'frontlist_rank', None)
                record.rerank_pool_rank = getattr(state, 'rerank_pool_rank', None)
                record.rerank_model_score = getattr(state, 'rerank_model_score', None)
                record.rerank_rule_score = getattr(state, 'rerank_rule_score', None)
                record.rerank_blend_score = getattr(state, 'rerank_blend_score', None)
                record.rerank_rule_weight = getattr(state, 'rerank_rule_weight', None)
                record.rerank_model_target = getattr(state, 'rerank_model_target', None)
                record.rerank_selected_for_llm = bool(getattr(state, 'rerank_selected_for_llm', False))
                record.selection_stage = getattr(state, 'selection_stage', None)
                record.selection_reason = getattr(state, 'selection_reason', None)
                record.selection_reason_components = dict(getattr(state, 'selection_reason_components', {}) or {})
                record.structured_rank_score = getattr(state, 'structured_rank_score', None)
                record.structured_rank_position = getattr(state, 'structured_rank_position', None)
                record.previous_recommendation_score = state.previous_recommendation_score
                record.previous_overall_score = state.previous_overall_score
                record.previous_confidence = state.previous_confidence
                record.score_change = state.score_change
                record.today_present = bool(state.today_present)
                record.absence_reason = state.absence_reason
                record.action_plan = dict(state.action_plan or {})
                record.review_status = state.review_status
                record.yesterday_conclusion = state.yesterday_conclusion
                record.today_verdict = state.today_verdict
                record.miss_reason_candidates = list(state.miss_reason_candidates or [])
                record.missing_factor_candidates = list(state.missing_factor_candidates or [])
                record.updated_at = datetime.now()
                persisted.append(self._serialize_recommendation_pool_state(record))

            session.commit()
            return persisted
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_recommendation_pool_state(self, trade_date: Optional[date] = None) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            target_date = trade_date
            if target_date is None:
                target_date = session.query(func.max(RecommendationPoolState.trade_date)).scalar()
            if target_date is None:
                return []

            rows = session.query(RecommendationPoolState).filter(
                RecommendationPoolState.trade_date == target_date
            ).order_by(
                RecommendationPoolState.recommend_rank.asc().nullslast(),
                RecommendationPoolState.rerank_pool_rank.asc().nullslast(),
                RecommendationPoolState.recommendation_score.desc(),
                RecommendationPoolState.overall_score.desc().nullslast(),
                RecommendationPoolState.priority_score.desc(),
                RecommendationPoolState.ts_code.asc(),
            ).all()
            return [self._serialize_recommendation_pool_state(row) for row in rows]
        finally:
            session.close()

    def get_previous_recommendation_pool_trade_date(self, trade_date: date) -> Optional[date]:
        session = self.get_session()
        try:
            return session.query(func.max(RecommendationPoolState.trade_date)).filter(
                RecommendationPoolState.trade_date < trade_date
            ).scalar()
        finally:
            session.close()

    def list_recommendation_pool(
        self,
        trade_date: Optional[date] = None,
        front_only: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            target_date = trade_date
            if target_date is None:
                target_date = session.query(func.max(RecommendationPoolState.trade_date)).scalar()
            if target_date is None:
                return []

            query = session.query(RecommendationPoolState).filter(
                RecommendationPoolState.trade_date == target_date
            )
            if front_only is not None:
                query = query.filter(RecommendationPoolState.in_frontlist == front_only)
            query = query.order_by(
                RecommendationPoolState.recommend_rank.asc().nullslast(),
                RecommendationPoolState.rerank_pool_rank.asc().nullslast(),
                RecommendationPoolState.recommendation_score.desc(),
                RecommendationPoolState.overall_score.desc().nullslast(),
                RecommendationPoolState.priority_score.desc(),
                RecommendationPoolState.ts_code.asc(),
            )
            if limit is not None:
                query = query.limit(limit)
            return [self._serialize_recommendation_pool_state(row) for row in query.all()]
        finally:
            session.close()

    def list_active_recommendations(self, limit: int = 50) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            query = session.query(RecommendationItem).outerjoin(RecommendationPerformance).filter(
                RecommendationItem.status.in_(['new', 'tracking', 'active'])
            ).order_by(RecommendationItem.trade_date.desc(), RecommendationItem.recommend_rank.asc())
            return [self._serialize_recommendation_item(item) for item in query.limit(limit).all()]
        finally:
            session.close()

    def list_new_recommendations(self, trade_date: Optional[date] = None, limit: int = 20) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            query = session.query(RecommendationItem).outerjoin(RecommendationPerformance).filter(
                RecommendationItem.status == 'new'
            )
            if trade_date is not None:
                query = query.filter(RecommendationItem.trade_date == trade_date)
            query = query.order_by(RecommendationItem.trade_date.desc(), RecommendationItem.recommend_rank.asc())
            return [self._serialize_recommendation_item(item) for item in query.limit(limit).all()]
        finally:
            session.close()

    def list_recommendation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            runs = session.query(RecommendationRun).order_by(RecommendationRun.trade_date.desc(), RecommendationRun.generated_at.desc()).limit(limit).all()
            history = []
            for run in runs:
                history.append({
                    'run_id': run.run_id,
                    'trade_date': run.trade_date.isoformat(),
                    'generated_at': run.generated_at.isoformat() if run.generated_at else None,
                    'candidate_count': run.candidate_count,
                    'final_count': run.final_count,
                    'report_id': run.report_id,
                })
            return history
        finally:
            session.close()

    def list_recommendation_run_items(self, trade_date: Optional[date] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            query = session.query(RecommendationItem).outerjoin(RecommendationPerformance)
            if trade_date is not None:
                query = query.filter(RecommendationItem.trade_date == trade_date)
            query = query.order_by(RecommendationItem.trade_date.asc(), RecommendationItem.recommend_rank.asc().nullslast(), RecommendationItem.id.asc())
            if limit is not None:
                query = query.limit(limit)
            return [self._serialize_recommendation_item(item) for item in query.all()]
        finally:
            session.close()

    def list_pending_performance_updates(self, lookback_days: int = 15, limit: int = 100) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            cutoff_date = datetime.now().date() - timedelta(days=lookback_days)
            query = session.query(RecommendationItem).outerjoin(RecommendationPerformance).filter(
                RecommendationItem.trade_date >= cutoff_date,
                RecommendationItem.status.in_(['new', 'tracking', 'validated', 'active'])
            ).order_by(RecommendationItem.trade_date.desc(), RecommendationItem.id.asc())
            return [self._serialize_recommendation_item(item) for item in query.limit(limit).all()]
        finally:
            session.close()

    def upsert_recommendation_performance(self, recommendation_item_id: int, performance: Dict[str, Any]) -> Dict[str, Any]:
        session = self.get_session()
        try:
            item = session.query(RecommendationItem).filter_by(id=recommendation_item_id).first()
            if item is None:
                raise ValueError(f"Recommendation item not found: {recommendation_item_id}")

            record = session.query(RecommendationPerformance).filter_by(
                recommendation_item_id=recommendation_item_id
            ).first()
            if record is None:
                record = RecommendationPerformance(recommendation_item_id=recommendation_item_id)
                session.add(record)

            for field in [
                'entry_price', 'latest_price', 'return_1d', 'return_3d', 'return_5d',
                'return_10d', 'max_drawdown_10d', 'benchmark_code', 'benchmark_return_5d',
                'vs_benchmark_5d', 'hit_5d'
            ]:
                if field in performance:
                    setattr(record, field, performance.get(field))

            if 'tracking_days' in performance:
                item.tracking_days = performance.get('tracking_days') or 0
            if 'status' in performance and performance.get('status'):
                item.status = performance.get('status')
            elif item.tracking_days >= 10:
                item.status = 'validated'
            elif item.tracking_days >= 1:
                item.status = 'tracking'
            else:
                item.status = 'new'

            if performance.get('entry_price') is not None:
                item.entry_price = performance.get('entry_price')

            session.commit()
            return self._serialize_recommendation_item(item)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_recommendation_summary(self, lookback_days: int = 30, history_limit: int = 20) -> Dict[str, Any]:
        session = self.get_session()
        try:
            cutoff_date = datetime.now().date() - timedelta(days=lookback_days)
            recent_items = session.query(RecommendationItem).outerjoin(RecommendationPerformance).filter(
                RecommendationItem.trade_date >= cutoff_date
            ).all()

            serialized_items = [self._serialize_recommendation_item(item) for item in recent_items]
            validated_3d = [item for item in serialized_items if item.get('return_3d') is not None]
            wins_3d = [item for item in validated_3d if item.get('return_3d', 0) > 0]
            average_return_3d = sum(item.get('return_3d', 0.0) for item in validated_3d) / len(validated_3d) if validated_3d else 0.0
            win_rate_3d = len(wins_3d) / len(validated_3d) if validated_3d else 0.0
            validated = [item for item in serialized_items if item.get('return_5d') is not None]
            wins = [item for item in validated if item.get('return_5d', 0) > 0]
            losses = [item for item in validated if item.get('return_5d', 0) <= 0]
            benchmark_wins = [item for item in validated if (item.get('vs_benchmark_5d') or 0) > 0]
            average_return_5d = sum(item.get('return_5d', 0.0) for item in validated) / len(validated) if validated else 0.0
            average_vs_benchmark_5d = sum(item.get('vs_benchmark_5d', 0.0) or 0.0 for item in validated) / len(validated) if validated else 0.0
            win_rate_5d = len(wins) / len(validated) if validated else 0.0
            benchmark_win_rate_5d = len(benchmark_wins) / len(validated) if validated else 0.0
            positive_average = sum(item.get('return_5d', 0.0) for item in wins) / len(wins) if wins else 0.0
            negative_average = sum(abs(item.get('return_5d', 0.0)) for item in losses) / len(losses) if losses else 0.0
            profit_loss_ratio_5d = positive_average / negative_average if negative_average else None

            repeat_rows = session.query(
                RecommendationItem.ts_code,
                func.count(RecommendationItem.id).label('count'),
                func.avg(RecommendationPerformance.return_5d).label('avg_return_5d'),
                func.avg(RecommendationPerformance.vs_benchmark_5d).label('avg_vs_benchmark_5d'),
            ).outerjoin(RecommendationPerformance).filter(
                RecommendationItem.trade_date >= cutoff_date
            ).group_by(RecommendationItem.ts_code).having(func.count(RecommendationItem.id) > 1).order_by(
                func.count(RecommendationItem.id).desc(), RecommendationItem.ts_code.asc()
            ).limit(10).all()

            history = self.list_recommendation_history(limit=history_limit)
            return {
                'history': history,
                'stats': {
                    'lookback_days': lookback_days,
                    'window_count': len(serialized_items),
                    'validated_count_3d': len(validated_3d),
                    'win_rate_3d': win_rate_3d,
                    'average_return_3d': average_return_3d,
                    'validated_count': len(validated),
                    'win_rate_5d': win_rate_5d,
                    'average_return_5d': average_return_5d,
                    'average_vs_benchmark_5d': average_vs_benchmark_5d,
                    'benchmark_win_rate_5d': benchmark_win_rate_5d,
                    'profit_loss_ratio_5d': profit_loss_ratio_5d,
                    'repeat_recommendations': [
                        {
                            'ts_code': row.ts_code,
                            'recommendation_count': int(row.count or 0),
                            'average_return_5d': float(row.avg_return_5d or 0.0),
                            'average_vs_benchmark_5d': float(row.avg_vs_benchmark_5d or 0.0),
                        }
                        for row in repeat_rows
                    ],
                },
            }
        finally:
            session.close()

    def upsert_short_term_training_samples(self, samples: List[ShortTermTrainingSample]) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            persisted: List[Dict[str, Any]] = []
            for sample in samples:
                record = session.query(ShortTermTrainingSampleRecord).filter_by(
                    trade_date=sample.trade_date,
                    ts_code=sample.ts_code,
                ).first()
                if record is None:
                    record = ShortTermTrainingSampleRecord(
                        trade_date=sample.trade_date,
                        ts_code=sample.ts_code,
                        created_at=datetime.now(),
                    )
                    session.add(record)

                payload = sample.model_dump()
                for key, value in payload.items():
                    if key in {"trade_date", "ts_code"}:
                        continue
                    if key in {"distribution_risk_flags", "continuation_positive_flags", "continuation_negative_flags"}:
                        setattr(record, key, list(value or []))
                    elif key == "action_plan":
                        setattr(record, key, dict(value or {}))
                    else:
                        setattr(record, key, value)
                record.updated_at = datetime.now()
                persisted.append(self._serialize_short_term_training_sample(record))

            session.commit()
            return persisted
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_short_term_training_samples(
        self,
        *,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            query = session.query(ShortTermTrainingSampleRecord)
            if start_date is not None:
                query = query.filter(ShortTermTrainingSampleRecord.trade_date >= start_date)
            if end_date is not None:
                query = query.filter(ShortTermTrainingSampleRecord.trade_date <= end_date)
            query = query.order_by(ShortTermTrainingSampleRecord.trade_date.asc(), ShortTermTrainingSampleRecord.ts_code.asc())
            if limit is not None:
                query = query.limit(limit)
            return [self._serialize_short_term_training_sample(row) for row in query.all()]
        finally:
            session.close()

    def get_short_term_training_sample_summary(self) -> Dict[str, Any]:
        session = self.get_session()
        try:
            rows = session.query(ShortTermTrainingSampleRecord).all()
            labeled_rows = [row for row in rows if row.label_up_1d is not None]
            return {
                "sample_count": len(rows),
                "labeled_count": len(labeled_rows),
                "trade_days": len({row.trade_date for row in rows if row.trade_date is not None}),
                "symbols": len({row.ts_code for row in rows if row.ts_code}),
                "schema_versions": sorted({row.feature_schema_version for row in rows if row.feature_schema_version}),
            }
        finally:
            session.close()

    @staticmethod
    def _serialize_recommendation_pool_state(item: RecommendationPoolState) -> Dict[str, Any]:
        return {
            'id': item.id,
            'ts_code': item.ts_code,
            'trade_date': item.trade_date.isoformat() if item.trade_date else None,
            'name': item.name,
            'recommendation_score': item.recommendation_score,
            'priority_score': item.priority_score,
            'overall_score': item.overall_score,
            'hit_streak_days': item.hit_streak_days,
            'miss_streak_days': item.miss_streak_days,
            'in_frontlist': bool(item.in_frontlist),
            'llm_focus_level': item.llm_focus_level,
            'tracking_status': item.tracking_status,
            'source_tag': item.source_tag or '今日Top3',
            'is_repeat_pick': bool(item.is_repeat_pick),
            'setup_type': item.setup_type,
            'risk_level': item.risk_level,
            'recommendation': item.recommendation,
            'position_status': item.position_status,
            'last_frontlist_date': item.last_frontlist_date.isoformat() if item.last_frontlist_date else None,
            'times_entered_frontlist': item.times_entered_frontlist,
            'technical_score': item.technical_score,
            'fundamental_score': item.fundamental_score,
            'sentiment_score': item.sentiment_score,
            'news_score': item.news_score,
            'base_score': item.base_score,
            'sentiment_adjustment': item.sentiment_adjustment,
            'news_adjustment': item.news_adjustment,
            'score_model': item.score_model,
            'close': item.close,
            'pct_change': item.pct_change,
            'volume_ratio': item.volume_ratio,
            'turnover_rate': item.turnover_rate,
            'ma20': getattr(item, 'ma20', None),
            'strategy_count': item.strategy_count,
            'divergence_score': item.divergence_score,
            'strategy_consistency_label': item.strategy_consistency_label,
            'news_mentioned': bool(item.news_mentioned),
            'industry': item.industry,
            'industry_heat_score': item.industry_heat_score,
            'industry_flow_bias': item.industry_flow_bias,
            'distribution_risk_score': item.distribution_risk_score,
            'distribution_risk_flags': list(item.distribution_risk_flags or []),
            'moneyflow_3d_value': item.moneyflow_3d_value,
            'recent_large_order_net_inflow': item.recent_large_order_net_inflow,
            'recent_super_large_order_net_inflow': item.recent_super_large_order_net_inflow,
            'turnover_spike_ratio': item.turnover_spike_ratio,
            'recent_runup_5d': item.recent_runup_5d,
            'continuation_bias_score': item.continuation_bias_score,
            'continuation_positive_flags': list(item.continuation_positive_flags or []),
            'continuation_negative_flags': list(item.continuation_negative_flags or []),
            'top3_risk_penalty': item.top3_risk_penalty,
            'short_term_contradiction_penalty': item.short_term_contradiction_penalty,
            'final_display_recommendation_score': item.final_display_recommendation_score,
            'top3_status': item.top3_status,
            'top3_reason': item.top3_reason,
            'late_stage_momentum_flag': bool(item.late_stage_momentum_flag),
            'candidate_risk_blocked': bool(item.candidate_risk_blocked),
            'top3_extreme_risk_blocked': bool(getattr(item, 'top3_extreme_risk_blocked', False)),
            'top3_extreme_risk_reason': getattr(item, 'top3_extreme_risk_reason', None),
            'ai_confidence': item.ai_confidence,
            'display_confidence': item.display_confidence,
            'technical_signal': item.technical_signal,
            'summary': item.summary,
            'recommendation_text': item.recommendation_text,
            'entry_price': item.entry_price,
            'fundamental_bonus': item.fundamental_bonus,
            'fundamental_bonus_breakdown': dict(item.fundamental_bonus_breakdown or {}),
            'recommend_rank': item.recommend_rank,
            'frontlist_rank': item.frontlist_rank,
            'rerank_pool_rank': item.rerank_pool_rank,
            'rerank_model_score': item.rerank_model_score,
            'rerank_rule_score': item.rerank_rule_score,
            'rerank_blend_score': item.rerank_blend_score,
            'rerank_rule_weight': item.rerank_rule_weight,
            'rerank_model_target': item.rerank_model_target,
            'rerank_selected_for_llm': bool(item.rerank_selected_for_llm),
            'selection_stage': getattr(item, 'selection_stage', None),
            'selection_reason': getattr(item, 'selection_reason', None),
            'selection_reason_components': dict(getattr(item, 'selection_reason_components', {}) or {}),
            'structured_rank_score': getattr(item, 'structured_rank_score', None),
            'structured_rank_position': getattr(item, 'structured_rank_position', None),
            'previous_recommendation_score': item.previous_recommendation_score,
            'previous_overall_score': item.previous_overall_score,
            'previous_confidence': item.previous_confidence,
            'score_change': item.score_change,
            'today_present': bool(item.today_present),
            'absence_reason': item.absence_reason,
            'action_plan': dict(item.action_plan or {}),
            'review_status': item.review_status,
            'yesterday_conclusion': item.yesterday_conclusion,
            'today_verdict': item.today_verdict,
            'miss_reason_candidates': list(item.miss_reason_candidates or []),
            'missing_factor_candidates': list(item.missing_factor_candidates or []),
        }

    @staticmethod
    def _serialize_recommendation_item(item: RecommendationItem) -> Dict[str, Any]:
        performance = item.performance
        return {
            'id': item.id,
            'run_id': item.recommendation_run.run_id if item.recommendation_run else None,
            'ts_code': item.ts_code,
            'name': item.name,
            'recommend_rank': item.recommend_rank,
            'recommend_score': item.recommend_score,
            'ai_confidence': item.ai_confidence,
            'source_tag': item.source_tag or '今日Top3',
            'is_repeat_pick': bool(item.is_repeat_pick),
            'strategy_count': item.strategy_count,
            'news_mentioned': bool(item.news_mentioned),
            'technical_signal': item.technical_signal,
            'recommendation_text': item.recommendation_text,
            'status': item.status,
            'tracking_days': item.tracking_days,
            'trade_date': item.trade_date.isoformat() if item.trade_date else None,
            'entry_price': item.entry_price,
            'score_mode': item.score_mode,
            'rerank_pool_rank': item.rerank_pool_rank,
            'rerank_blend_score': item.rerank_blend_score,
            'rerank_model_score': item.rerank_model_score,
            'rerank_rule_score': item.rerank_rule_score,
            'rerank_rule_weight': item.rerank_rule_weight,
            'rerank_model_target': item.rerank_model_target,
            'selection_stage': item.selection_stage,
            'selection_reason': item.selection_reason,
            'selection_reason_components': dict(item.selection_reason_components or {}),
            'structured_rank_score': item.structured_rank_score,
            'structured_rank_position': item.structured_rank_position,
            'latest_price': performance.latest_price if performance else None,
            'return_1d': performance.return_1d if performance else None,
            'return_3d': performance.return_3d if performance else None,
            'return_5d': performance.return_5d if performance else None,
            'return_10d': performance.return_10d if performance else None,
            'max_drawdown_10d': performance.max_drawdown_10d if performance else None,
            'benchmark_code': performance.benchmark_code if performance else '000300.SH',
            'benchmark_return_5d': performance.benchmark_return_5d if performance else None,
            'vs_benchmark_5d': performance.vs_benchmark_5d if performance else None,
            'hit_5d': performance.hit_5d if performance else None,
        }

    @staticmethod
    def _serialize_short_term_training_sample(item: ShortTermTrainingSampleRecord) -> Dict[str, Any]:
        return {
            'id': item.id,
            'feature_schema_version': item.feature_schema_version,
            'trade_date': item.trade_date.isoformat() if item.trade_date else None,
            'ts_code': item.ts_code,
            'name': item.name,
            'source_tag': item.source_tag,
            'in_frontlist': bool(item.in_frontlist),
            'recommend_rank': item.recommend_rank,
            'strategy_count': item.strategy_count,
            'is_repeat_pick': bool(item.is_repeat_pick),
            'news_mentioned': bool(item.news_mentioned),
            'technical_signal': item.technical_signal,
            'entry_price': item.entry_price,
            'close': item.close,
            'pct_change': item.pct_change,
            'volume_ratio': item.volume_ratio,
            'turnover_rate': item.turnover_rate,
            'recommendation_score': item.recommendation_score,
            'overall_score': item.overall_score,
            'technical_score': item.technical_score,
            'fundamental_score': item.fundamental_score,
            'sentiment_score': item.sentiment_score,
            'news_score': item.news_score,
            'base_score': item.base_score,
            'sentiment_adjustment': item.sentiment_adjustment,
            'news_adjustment': item.news_adjustment,
            'industry': item.industry,
            'industry_heat_score': item.industry_heat_score,
            'industry_flow_bias': item.industry_flow_bias,
            'distribution_risk_score': item.distribution_risk_score,
            'distribution_risk_flags': list(item.distribution_risk_flags or []),
            'moneyflow_3d_value': item.moneyflow_3d_value,
            'recent_large_order_net_inflow': item.recent_large_order_net_inflow,
            'recent_super_large_order_net_inflow': item.recent_super_large_order_net_inflow,
            'turnover_spike_ratio': item.turnover_spike_ratio,
            'recent_runup_5d': item.recent_runup_5d,
            'continuation_bias_score': item.continuation_bias_score,
            'continuation_positive_flags': list(item.continuation_positive_flags or []),
            'continuation_negative_flags': list(item.continuation_negative_flags or []),
            'top3_risk_penalty': item.top3_risk_penalty,
            'short_term_contradiction_penalty': item.short_term_contradiction_penalty,
            'late_stage_momentum_flag': bool(item.late_stage_momentum_flag),
            'candidate_risk_blocked': bool(item.candidate_risk_blocked),
            'previous_recommendation_score': item.previous_recommendation_score,
            'previous_overall_score': item.previous_overall_score,
            'score_change': item.score_change,
            'action_plan': dict(item.action_plan or {}),
            'return_1d': item.return_1d,
            'return_3d': item.return_3d,
            'return_5d': item.return_5d,
            'return_10d': item.return_10d,
            'max_drawdown_10d': item.max_drawdown_10d,
            'benchmark_return_5d': item.benchmark_return_5d,
            'vs_benchmark_5d': item.vs_benchmark_5d,
            'label_up_1d': item.label_up_1d,
        }

    @staticmethod
    def _normalize_recommendation_status(raw_status: Any, tracking_status: Any) -> str:
        status_text = str(raw_status or "").strip().lower()
        if status_text == "active":
            return "new"
        if status_text:
            return status_text
        return 'tracking' if str(tracking_status or "").strip().lower() == 'active' else 'new'

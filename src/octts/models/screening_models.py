"""Database models for screening system using SQLAlchemy."""

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, Text, JSON, Index, ForeignKey, Table, Date, func, inspect, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import StaticPool

from octts.schemas.screener import ScreenCriteria, ScreenResult, StockScreenItem, TrackedRecommendationState

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


class NewsCluster(Base):
    """新闻聚类"""
    __tablename__ = 'news_clusters'

    id = Column(Integer, primary_key=True)
    cluster_id = Column(String(50), unique=True, nullable=False)
    cluster_date = Column(DateTime, nullable=False)
    theme = Column(String(200), nullable=False)
    importance = Column(Float)
    summary = Column(Text)
    key_stocks = Column(JSON)
    news_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)

    news_items = relationship("NewsItem", back_populates="cluster")

    __table_args__ = (
        Index('idx_cluster_date', 'cluster_date'),
        Index('idx_importance', 'importance'),
    )


class NewsItem(Base):
    """新闻条目"""
    __tablename__ = 'news_items'

    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey('news_clusters.id'))
    source = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    url = Column(String(500))
    publish_time = Column(DateTime, nullable=False)
    importance = Column(Float, default=0.5)
    sentiment = Column(Float, default=0.0)
    related_stocks = Column(JSON)
    tags = Column(JSON)
    content_hash = Column(String(64), unique=True)
    created_at = Column(DateTime, default=datetime.now)

    cluster = relationship("NewsCluster", back_populates="news_items")

    __table_args__ = (
        Index('idx_publish_time', 'publish_time'),
        Index('idx_source', 'source'),
    )


class IntelligentReport(Base):
    """智能报告"""
    __tablename__ = 'intelligent_reports'

    id = Column(Integer, primary_key=True)
    report_id = Column(String(50), unique=True, nullable=False)
    report_type = Column(String(20), nullable=False)
    title = Column(String(200), nullable=False)
    generate_time = Column(DateTime, nullable=False)
    summary = Column(Text)
    key_points = Column(JSON)
    recommendations = Column(JSON)
    sections = Column(JSON)
    report_metadata = Column("metadata", JSON)
    pushed_wechat = Column(Boolean, default=False)
    pushed_email = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index('idx_report_type', 'report_type'),
        Index('idx_generate_time', 'generate_time'),
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
    close = Column(Float)
    pct_change = Column(Float)
    volume_ratio = Column(Float)
    turnover_rate = Column(Float)
    strategy_count = Column(Integer, default=0)
    news_mentioned = Column(Boolean, default=False)
    ai_confidence = Column(Float)
    technical_signal = Column(String(200))
    recommendation_text = Column(Text)
    entry_price = Column(Float)
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
                'close': "ALTER TABLE recommendation_pool_states ADD COLUMN close FLOAT",
                'pct_change': "ALTER TABLE recommendation_pool_states ADD COLUMN pct_change FLOAT",
                'volume_ratio': "ALTER TABLE recommendation_pool_states ADD COLUMN volume_ratio FLOAT",
                'turnover_rate': "ALTER TABLE recommendation_pool_states ADD COLUMN turnover_rate FLOAT",
                'strategy_count': "ALTER TABLE recommendation_pool_states ADD COLUMN strategy_count INTEGER DEFAULT 0",
                'news_mentioned': "ALTER TABLE recommendation_pool_states ADD COLUMN news_mentioned BOOLEAN DEFAULT 0",
                'ai_confidence': "ALTER TABLE recommendation_pool_states ADD COLUMN ai_confidence FLOAT",
                'technical_signal': "ALTER TABLE recommendation_pool_states ADD COLUMN technical_signal VARCHAR(200)",
                'recommendation_text': "ALTER TABLE recommendation_pool_states ADD COLUMN recommendation_text TEXT",
                'entry_price': "ALTER TABLE recommendation_pool_states ADD COLUMN entry_price FLOAT",
                'updated_at': "ALTER TABLE recommendation_pool_states ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP",
            },
            'recommendation_items': {
                'source_tag': "ALTER TABLE recommendation_items ADD COLUMN source_tag VARCHAR(50) DEFAULT '今日Top3'",
                'is_repeat_pick': "ALTER TABLE recommendation_items ADD COLUMN is_repeat_pick BOOLEAN DEFAULT 0",
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
                    pct_change=item.pct_change or 0.0,
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
                    status=item.get('status') or ('tracking' if tracking_status == 'active' else 'new'),
                    tracking_days=item.get('tracking_days', 0),
                    trade_date=item.get('trade_date', trade_date),
                    entry_price=item.get('entry_price'),
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
                record.close = state.close
                record.pct_change = state.pct_change
                record.volume_ratio = state.volume_ratio
                record.turnover_rate = state.turnover_rate
                record.strategy_count = state.strategy_count
                record.news_mentioned = bool(state.news_mentioned)
                record.ai_confidence = state.ai_confidence
                record.technical_signal = state.technical_signal
                record.recommendation_text = state.recommendation_text
                record.entry_price = state.entry_price
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
                RecommendationPoolState.recommend_rank.asc(),
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
                RecommendationPoolState.recommend_rank.asc(),
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
                RecommendationItem.status.in_(['new', 'tracking'])
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

    def list_pending_performance_updates(self, lookback_days: int = 15, limit: int = 100) -> List[Dict[str, Any]]:
        session = self.get_session()
        try:
            cutoff_date = datetime.now().date() - timedelta(days=lookback_days)
            query = session.query(RecommendationItem).outerjoin(RecommendationPerformance).filter(
                RecommendationItem.trade_date >= cutoff_date,
                RecommendationItem.status.in_(['new', 'tracking', 'validated'])
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

    @staticmethod
    def _serialize_recommendation_pool_state(item: RecommendationPoolState) -> Dict[str, Any]:
        return {
            'id': item.id,
            'ts_code': item.ts_code,
            'trade_date': item.trade_date.isoformat() if item.trade_date else None,
            'name': item.name,
            'recommendation_score': item.recommendation_score,
            'priority_score': item.priority_score,
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
            'close': item.close,
            'pct_change': item.pct_change,
            'volume_ratio': item.volume_ratio,
            'turnover_rate': item.turnover_rate,
            'strategy_count': item.strategy_count,
            'news_mentioned': bool(item.news_mentioned),
            'ai_confidence': item.ai_confidence,
            'technical_signal': item.technical_signal,
            'recommendation_text': item.recommendation_text,
            'entry_price': item.entry_price,
            'previous_recommendation_score': getattr(item, 'previous_recommendation_score', None),
            'score_change': getattr(item, 'score_change', None),
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

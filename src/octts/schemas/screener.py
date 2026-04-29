"""Stock screener data models."""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ScreenCriteria(BaseModel):
    """股票筛选条件"""
    # 价格相关
    price_min: Optional[float] = Field(None, description="最低价格")
    price_max: Optional[float] = Field(None, description="最高价格")
    pct_change_min: Optional[float] = Field(None, description="最小涨跌幅%")
    pct_change_max: Optional[float] = Field(None, description="最大涨跌幅%")

    # 成交量相关
    volume_ratio_min: Optional[float] = Field(None, description="最小量比")
    volume_ratio_max: Optional[float] = Field(None, description="最大量比")
    turnover_rate_min: Optional[float] = Field(None, description="最小换手率%")
    turnover_rate_max: Optional[float] = Field(None, description="最大换手率%")

    # 技术指标
    rsi_min: Optional[float] = Field(None, description="最小RSI值")
    rsi_max: Optional[float] = Field(None, description="最大RSI值")
    ma5_above_ma20: Optional[bool] = Field(None, description="5日均线高于20日均线")
    price_above_ma5: Optional[bool] = Field(None, description="价格高于5日均线")
    price_above_ma60: Optional[bool] = Field(None, description="价格高于60日均线")
    technical_score_min: Optional[float] = Field(None, description="最小技术评分")
    recommendation_score_min: Optional[float] = Field(None, description="最小推荐评分")
    setup_quality_score_min: Optional[float] = Field(None, description="最小形态质量评分")
    setup_types: Optional[List[str]] = Field(None, description="限定形态类型")
    risk_level_max: Optional[str] = Field(None, description="最大风险等级")
    recommendation_min: Optional[str] = Field(None, description="最低推荐等级")
    require_bullish_ma_alignment: Optional[bool] = Field(None, description="要求均线多头排列")
    require_macd_above_signal: Optional[bool] = Field(None, description="要求MACD在信号线之上")
    price_position_min: Optional[float] = Field(None, description="20日价格位置下限(0-1)")
    price_position_max: Optional[float] = Field(None, description="20日价格位置上限(0-1)")
    distance_to_ma60_pct_min: Optional[float] = Field(None, description="距离MA60百分比下限")
    distance_to_ma60_pct_max: Optional[float] = Field(None, description="距离MA60百分比上限")

    # 市值相关
    market_cap_min: Optional[float] = Field(None, description="最小市值(亿)")
    market_cap_max: Optional[float] = Field(None, description="最大市值(亿)")

    # 行业/板块
    industries: Optional[List[str]] = Field(None, description="行业列表")
    exclude_st: bool = Field(True, description="排除ST股票")
    exclude_bj: bool = Field(False, description="排除北交所股票")
    max_recent_loss_years: Optional[int] = Field(None, description="最近连续亏损年数上限")
    require_positive_3d_moneyflow: bool = Field(False, description="要求近3日资金净流入为正")
    max_late_stage_turnover_rate: Optional[float] = Field(None, description="末端风险过滤: 最大允许换手率%")
    max_late_stage_volume_ratio: Optional[float] = Field(None, description="末端风险过滤: 最大允许量比")
    max_late_stage_pct_change: Optional[float] = Field(None, description="末端风险过滤: 最大允许当日涨幅%")
    max_late_stage_price_position: Optional[float] = Field(None, description="末端风险过滤: 最大允许20日价格位置(0-1)")

    # 排序
    sort_by: str = Field("pct_change", description="排序字段")
    sort_desc: bool = Field(True, description="降序排序")

    # 分页
    limit: int = Field(50, description="返回数量限制")
    offset: int = Field(0, description="偏移量")


class StockScreenItem(BaseModel):
    """筛选结果中的单个股票"""
    ts_code: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    close: float = Field(..., description="最新价")
    pct_change: Optional[float] = Field(None, description="涨跌幅%")
    volume_ratio: float = Field(..., description="量比")
    turnover_rate: float = Field(..., description="换手率%")

    # 技术指标
    rsi: Optional[float] = Field(None, description="RSI值")
    ma5: Optional[float] = Field(None, description="5日均价")
    ma10: Optional[float] = Field(None, description="10日均价")
    ma20: Optional[float] = Field(None, description="20日均价")
    ma60: Optional[float] = Field(None, description="60日均价")
    macd: Optional[float] = Field(None, description="MACD值")
    macd_signal: Optional[float] = Field(None, description="MACD信号线")
    macd_histogram: Optional[float] = Field(None, description="MACD柱状图")
    price_position_20d: Optional[float] = Field(None, description="20日价格位置")
    trend_status: Optional[str] = Field(None, description="趋势状态")
    momentum_status: Optional[str] = Field(None, description="动量状态")
    technical_score: Optional[float] = Field(None, description="技术评分")
    trend_score: Optional[float] = Field(None, description="趋势评分")
    momentum_score: Optional[float] = Field(None, description="动量评分")
    volume_score: Optional[float] = Field(None, description="量能评分")
    breakout_score: Optional[float] = Field(None, description="突破/形态评分")
    risk_score: Optional[float] = Field(None, description="执行风险评分")
    setup_quality_score: Optional[float] = Field(None, description="形态质量评分")
    recommendation_score: Optional[float] = Field(None, description="推荐评分")
    recommendation: Optional[str] = Field(None, description="推荐标签")
    setup_type: Optional[str] = Field(None, description="机会形态类型")
    risk_level: Optional[str] = Field(None, description="风险等级")
    entry_style: Optional[str] = Field(None, description="入场风格")
    confidence: Optional[str] = Field(None, description="信号置信度")
    risk_flags: List[str] = Field(default_factory=list, description="风险标记")
    setup_notes: List[str] = Field(default_factory=list, description="形态说明")
    distance_to_ma20_pct: Optional[float] = Field(None, description="距离MA20百分比")
    distance_to_ma60_pct: Optional[float] = Field(None, description="距离MA60百分比")
    breakout_strength: Optional[float] = Field(None, description="突破强度")

    # 基本面
    market_cap: Optional[float] = Field(None, description="市值(亿)")
    pe_ratio: Optional[float] = Field(None, description="市盈率")
    industry: Optional[str] = Field(None, description="行业")

    # 筛选得分
    score: Optional[float] = Field(None, description="综合得分")
    match_reasons: List[str] = Field(default_factory=list, description="符合条件")


class ScreenResult(BaseModel):
    """筛选结果"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    screen_id: str = Field(..., description="筛选ID")
    criteria: ScreenCriteria = Field(..., description="筛选条件")
    stocks: List[StockScreenItem] = Field(..., description="符合条件的股票")
    total_count: int = Field(..., description="符合条件的总数")
    screen_time: datetime = Field(default_factory=datetime.now, description="筛选时间")
    execution_time: float = Field(..., description="执行时间(秒)")


class TrackedRecommendationState(BaseModel):
    """持续跟踪推荐池状态"""

    ts_code: str = Field(..., description="股票代码")
    trade_date: date = Field(..., description="交易日")
    recommendation_score: float = Field(0.0, description="当前推荐分")
    priority_score: Optional[float] = Field(None, description="兼容旧数据的优先级分")
    overall_score: Optional[float] = Field(None, description="多维综合分")
    hit_streak_days: int = Field(0, description="兼容旧数据的连续命中天数")
    miss_streak_days: int = Field(0, description="兼容旧数据的连续未命中天数")
    in_frontlist: bool = Field(False, description="是否在展示窗口中")
    llm_focus_level: str = Field("low", description="LLM 关注级别")
    tracking_status: str = Field("shadow", description="跟踪状态")
    source_tag: str = Field("今日Top3", description="展示来源标签")
    is_repeat_pick: bool = Field(False, description="是否连续入选")
    setup_type: Optional[str] = Field(None, description="形态类型")
    risk_level: Optional[str] = Field(None, description="风险等级")
    recommendation: Optional[str] = Field(None, description="推荐标签")
    position_status: Optional[str] = Field(None, description="持仓标签")
    last_frontlist_date: Optional[date] = Field(None, description="最近进入展示窗口日期")
    times_entered_frontlist: int = Field(0, description="进入展示窗口次数")
    name: Optional[str] = Field(None, description="股票名称")
    technical_score: Optional[float] = Field(None, description="技术评分")
    fundamental_score: Optional[float] = Field(None, description="基本面评分")
    sentiment_score: Optional[float] = Field(None, description="情绪评分")
    news_score: Optional[float] = Field(None, description="新闻评分")
    close: Optional[float] = Field(None, description="收盘价")
    pct_change: Optional[float] = Field(None, description="涨跌幅")
    volume_ratio: Optional[float] = Field(None, description="量比")
    turnover_rate: Optional[float] = Field(None, description="换手率")
    ma20: Optional[float] = Field(None, description="20日均价")
    strategy_count: int = Field(0, description="命中策略数")
    divergence_score: Optional[float] = Field(None, description="多策略分歧分")
    strategy_consistency_label: Optional[str] = Field(None, description="策略一致性标签")
    news_mentioned: bool = Field(False, description="是否命中新闻热点")
    ai_confidence: Optional[float] = Field(None, description="AI 置信度")
    display_confidence: Optional[float] = Field(None, description="展示用原始置信度")
    overall_confidence: Optional[float] = Field(None, description="统一聚合后的总体置信度")
    confidence: Optional[float] = Field(None, description="最终对外透传的置信度")
    technical_signal: Optional[str] = Field(None, description="技术信号")
    summary: Optional[str] = Field(None, description="分析摘要")
    overview_reason: Optional[str] = Field(None, description="总览短结论")
    recommendation_text: Optional[str] = Field(None, description="推荐文案")
    entry_price: Optional[float] = Field(None, description="入场参考价")
    recommend_rank: Optional[int] = Field(None, description="推荐排序位次")
    rerank_selected_for_llm: bool = Field(False, description="是否进入 LLM 重点分析集合")
    selection_stage: Optional[str] = Field(None, description="入选阶段标记")
    selection_reason: Optional[str] = Field(None, description="入选原因摘要")
    selection_reason_components: Dict[str, Any] = Field(default_factory=dict, description="入选原因结构化细项")
    structured_rank_score: Optional[float] = Field(None, description="结构化排序分")
    structured_rank_position: Optional[int] = Field(None, description="结构化排序位次")
    rerank_pool_rank: Optional[int] = Field(None, description="模型重排池排序位次")
    rerank_model_score: Optional[float] = Field(None, description="模型重排原始分")
    rerank_rule_score: Optional[float] = Field(None, description="规则排序原始分")
    rerank_blend_score: Optional[float] = Field(None, description="模型与规则混合分")
    rerank_rule_weight: Optional[float] = Field(None, description="规则分权重")
    rerank_model_target: Optional[str] = Field(None, description="模型重排目标")
    fundamental_bonus: Optional[float] = Field(None, description="轻量基本面修正分")
    fundamental_bonus_breakdown: Dict[str, Any] = Field(default_factory=dict, description="轻量基本面修正细项")
    previous_recommendation_score: Optional[float] = Field(None, description="昨日推荐分")
    previous_overall_score: Optional[float] = Field(None, description="昨日综合分")
    previous_confidence: Optional[float] = Field(None, description="昨日置信度")
    score_change: Optional[float] = Field(None, description="较昨日分数变化")
    today_present: bool = Field(True, description="今日是否仍在候选池中")
    absence_reason: Optional[str] = Field(None, description="今日缺席原因")
    action_plan: Dict[str, Any] = Field(default_factory=dict, description="短线执行建议")
    review_status: Optional[str] = Field(None, description="昨日 Top3 今日复评状态")
    yesterday_conclusion: Optional[str] = Field(None, description="昨日结论摘要")
    today_verdict: Optional[str] = Field(None, description="今日复评结论")
    miss_reason_candidates: List[str] = Field(default_factory=list, description="失误归因候选")
    missing_factor_candidates: List[str] = Field(default_factory=list, description="疑似缺失因子")
    base_score: Optional[float] = Field(None, description="基础评分")
    sentiment_adjustment: Optional[float] = Field(None, description="情绪修正")
    news_adjustment: Optional[float] = Field(None, description="新闻修正")
    score_model: Optional[str] = Field(None, description="评分模型")
    industry: Optional[str] = Field(None, description="行业")
    industry_heat_score: Optional[float] = Field(None, description="行业热度分")
    industry_flow_bias: Optional[str] = Field(None, description="行业资金倾向")
    distribution_risk_score: Optional[float] = Field(None, description="派发风险评分")
    distribution_risk_flags: List[str] = Field(default_factory=list, description="派发风险标签")
    moneyflow_3d_value: Optional[float] = Field(None, description="近3日主力资金净流入")
    recent_large_order_net_inflow: Optional[float] = Field(None, description="近3日大单净流入")
    recent_super_large_order_net_inflow: Optional[float] = Field(None, description="近3日超大单净流入")
    turnover_spike_ratio: Optional[float] = Field(None, description="换手放大倍数")
    recent_runup_5d: Optional[float] = Field(None, description="近5日累计涨幅")
    continuation_bias_score: Optional[float] = Field(None, description="次日延续偏置分")
    continuation_positive_flags: List[str] = Field(default_factory=list, description="次日延续正向标签")
    continuation_negative_flags: List[str] = Field(default_factory=list, description="次日延续负向标签")
    top3_risk_penalty: Optional[float] = Field(None, description="今日Top3额外风险扣分")
    short_term_contradiction_penalty: Optional[float] = Field(None, description="短线矛盾扣分")
    final_display_recommendation_score: Optional[float] = Field(None, description="最终展示推荐分")
    late_stage_momentum_flag: bool = Field(False, description="是否疑似末端分歧")
    candidate_risk_blocked: bool = Field(False, description="是否被极端风险阻断")
    top3_extreme_risk_blocked: bool = Field(False, description="是否因极端风险被排除出Top3")
    top3_extreme_risk_reason: Optional[str] = Field(None, description="Top3极端风险排除原因")


class ScreenPreset(BaseModel):
    """预设筛选策略"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "oversold_bounce",
                "name": "超跌反弹",
                "description": "寻找RSI<30的超跌股票",
                "criteria": {
                    "rsi_max": 30,
                    "volume_ratio_min": 1.5,
                    "exclude_st": True
                },
                "category": "technical"
            }
        }
    )

    id: str = Field(..., description="预设ID")
    name: str = Field(..., description="策略名称")
    description: str = Field(..., description="策略描述")
    criteria: ScreenCriteria = Field(..., description="筛选条件")
    category: str = Field(..., description="策略类别")
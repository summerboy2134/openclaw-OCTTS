from __future__ import annotations

import json
from datetime import datetime

from typing import Optional

from octts.schemas.report import AnalysisPhase, HistoricalAnalysisRecord, MemorySummary, PositionStatus, PriceSnapshot


def build_intelligent_screening_report_prompt(
    *,
    market_data: dict[str, object],
    news_clusters: list[dict[str, object]],
    screening_context: dict[str, object],
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严格遵守排序约束的 A 股智能选股复盘分析助手。"
        "你的任务是解释系统已经给出的排序与分数，不允许改写排序，不允许擅自调整推荐先后。"
        "你只能输出一个合法 JSON 对象，不要输出 Markdown，不要输出代码块，不要解释。"
        "如果证据不足，使用保守描述、null、空数组或观察区间，不要编造精准价位。"
    )

    payload = {
        "task": "基于智能选股上下文生成结构化报告",
        "market_data": market_data,
        "news_clusters": news_clusters,
        "screening_context": screening_context,
        "field_semantics": {
            "overall_score": "多维综合分",
            "recommendation_score": "最终推荐排序分，解释排序时优先参考该字段",
            "priority_score": "与综合分兼容的展示优先级字段",
            "overall_confidence": "分析置信度",
            "display_confidence": "前端展示置信度",
            "strategy_count": "命中策略数量",
            "news_mentioned": "是否命中新闻催化",
            "score_change": "与上一交易日推荐分变化",
            "source_tag": "来源标签，如今日Top3/昨日延续/今日候选",
        },
        "instructions": [
            "不能改写系统排序，你的任务是解释为什么排这样。",
            "推荐分高但综合分一般时，要优先从新闻催化、策略共振、短线交易性解释。",
            "综合分高但推荐分不高时，要解释质量好但交易性一般。",
            "短线建议必须基于输入中的 close、entry_price、技术描述、支撑阻力或总结信息；若证据不足，只能给观察区间。",
            "对昨日 Top3 今日复评，需要说明延续/转弱/失效，并尽量输出失误候选与可能缺失的因子。",
        ],
        "output_schema": {
            "focus_stocks": [
                {
                    "ts_code": "string",
                    "name": "string",
                    "core_highlights": ["string"],
                    "risk_warnings": ["string"],
                    "overall_assessment": "string",
                    "action_plan": {
                        "action_bias": "买入|观察|减仓|不参与",
                        "entry_zone": "string|null",
                        "take_profit": "string|null",
                        "stop_loss": "string|null",
                        "holding_horizon": "string|null",
                        "invalid_condition": "string|null",
                    },
                }
            ],
            "yesterday_reviews": [
                {
                    "ts_code": "string",
                    "name": "string",
                    "yesterday_conclusion": "string",
                    "today_verdict": "延续|转弱|失效|观察",
                    "status": "延续|转弱|失效|观察",
                    "analysis": "string",
                    "miss_reason_candidates": ["string"],
                    "missing_factor_candidates": ["string"],
                }
            ],
            "comparison": {
                "basic_rank": ["ts_code"],
                "technical_rank": ["ts_code"],
                "risk_rank": ["ts_code"],
                "trading_rank": ["ts_code"],
                "best_short_term": "ts_code",
                "most_robust": "ts_code",
                "highest_risk": "ts_code",
            },
            "overall_action": {
                "headline": "string",
                "market_view": "string",
                "risk_summary": "string",
                "action_items": ["string"],
            },
        },
    }

    return system_prompt, json.dumps(payload, ensure_ascii=False, indent=2)



def build_report_prompt(
    *,
    phase: AnalysisPhase,
    snapshot: PriceSnapshot,
    previous_memory: Optional[MemorySummary],
    previous_record: Optional[HistoricalAnalysisRecord] = None,
    market_context: Optional[dict[str, object]] = None,
    previous_trading_snapshot: Optional[dict[str, object]] = None,
    is_default_pool_symbol: bool = False,
    position_status: Optional[PositionStatus] = None,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严谨的 A 股量化复盘分析助手。"
        "你必须基于历史结论与当前数据做连续性判断，不允许忽略上次观点。"
        "你只能输出一个合法 JSON 对象。"
        "不要输出 Markdown，不要输出代码块，不要输出解释文字。"
        "所有 key 必须使用双引号。"
        "如果缺少信息，请使用 null、空数组或保守结论，不要省略必填字段。"
        "输出要清晰完整，在保证结构化的前提下适当展开关键信号、关键价位和主要依据。"
    )

    previous_payload = previous_memory.model_dump(mode="json") if previous_memory else {
        "status": "initial_analysis",
        "message": "No previous memory available.",
    }
    previous_record_payload = previous_record.model_dump(mode="json") if previous_record else None

    user_payload = {
        "task": "基于上一次判断和当前数据进行趋势延续/修正分析",
        "phase": phase,
        "snapshot": snapshot.model_dump(mode="json"),
        "previous_memory": previous_payload,
        "previous_record": previous_record_payload,
        "market_context": market_context,
        "previous_trading_snapshot": previous_trading_snapshot,
        "symbol_context": {
            "is_default_pool_symbol": is_default_pool_symbol,
            "position_status": position_status,
        },
        "field_unit_hints": {
            "snapshot.amount": (
                "成交额，TuShare 日线字段原始单位为千元；如需换算为亿元，请使用 amount / 100000。"
                "例如 amount=128386.959 时，只能写成 1.284 亿或约 1.28 亿，不能写成 12.84 亿。"
            ),
            "market_context.current_daily_bar.amount": (
                "成交额，单位同 snapshot.amount，为千元；换算为亿元请除以 100000。"
                "例如 amount=128386.959 时，应写成 1.284 亿。"
            ),
            "market_context.previous_daily_bar.amount": (
                "成交额，单位同 snapshot.amount，为千元；换算为亿元请除以 100000。"
                "禁止把千元误当成万元或亿元。"
            ),
            "market_context.recent_daily_bars[].amount": (
                "成交额，单位同 snapshot.amount，为千元；换算为亿元请除以 100000。"
                "若文案写'成交额 X 亿'，必须先完成换算再输出。"
            ),
        },
        "time_context": _build_time_context(
            phase=phase,
            snapshot=snapshot,
            previous_memory=previous_memory,
            previous_record=previous_record,
            market_context=market_context,
            previous_trading_snapshot=previous_trading_snapshot,
        ),
        "output_schema": {
            "ts_code": "string",
            "phase": "morning|afternoon|review",
            "trend_judgement": "string",
            "trend_breakdown": {
                "short_term": "bullish|neutral|bearish",
                "mid_term": "bullish|neutral|bearish",
                "long_term": "bullish|neutral|bearish",
                "short_term_reason": "string",
                "mid_term_reason": "string",
                "long_term_reason": "string",
            },
            "previous_view_status": "confirmed|weakened|reversed|initial",
            "operation_advice": "string",
            "risk_warning": ["string"],
            "observation_points": ["string"],
            "summary_markdown": "string",
            "decision": {
                "signal": "buy|hold|reduce|sell|avoid",
                "rationale": "string",
                "entry_zone": {
                    "low": "number|null",
                    "high": "number|null",
                },
                "stop_loss": "number|null",
                "take_profit": ["number"],
                "invalidation_condition": "string",
                "holding_horizon": "intraday|swing|position",
                "confidence_score": "number between 0 and 1",
                "risk_reward_ratio": "number|null",
                "evidence": ["string"],
            },
            "prediction_windows": [
                {
                    "window": "next_1d|next_3d|next_5d",
                    "bias": "bullish|neutral|bearish",
                    "confidence_score": "number between 0 and 1",
                    "rationale": "string",
                }
            ],
            "memory": {
                "ts_code": "string",
                "phase": "morning|afternoon|review",
                "trend_bias": "bullish|neutral|bearish",
                "short_term_bias": "bullish|neutral|bearish|null",
                "mid_term_bias": "bullish|neutral|bearish|null",
                "long_term_bias": "bullish|neutral|bearish|null",
                "support_levels": ["number"],
                "resistance_levels": ["number"],
                "capital_flow_view": "string",
                "key_risks": ["string"],
                "next_checkpoints": ["string"],
                "confidence_score": "number between 0 and 1",
                "summary": "string",
            },
        },
        "analysis_instructions": [
            "明确判断上一次观点是延续、减弱、反转还是首次分析。",
            "当前分析所对应的日期只能以 snapshot.trade_date 与 time_context.current_trade_date 为准，不要根据系统当前时间、生成时间或 minute_summary 自行推断'今天'。",
            "历史观点状态只用于承接上一次分析结论，默认参考 previous_record 与 previous_memory，不要把它和上一交易日行情混为一谈。",
            "把 market_context 视为主要分析输入，其中 current_daily_bar、previous_daily_bar、recent_daily_bars、current_weekly_bar、previous_weekly_bar、recent_weekly_bars 为程序整理后的可靠行情上下文。",
            "凡是描述'收涨/收跌'、'放量/缩量'、'站上/跌破'、'今日/本次'相对变化时，优先基于 market_context.current_daily_bar 与 market_context.previous_daily_bar 做比较；仅在缺失时，才回退为当前 snapshot 截面描述。",
            "凡是引用 snapshot.amount 或 market_context 中的 amount 描述成交额时，必须按'千元'理解；若输出为'亿'，请用 amount / 100000 换算，禁止把千元误写成万元或亿元。",
            "成交额是高频易错项：若 snapshot.amount=128386.959，则成交额只能写成 1.284 亿、约 1.28 亿或 12838.6959 万，绝不能写成 12.84 亿。",
            "如果你在 trend_judgement、summary_markdown、operation_advice、risk_warning、observation_points、decision.rationale、decision.evidence、prediction_windows.rationale 或 memory.* 中写到'成交额 X 亿'，输出前必须逐项核对 X 是否等于 amount / 100000。",
            "若无法确认换算结果，宁可直接写'成交额放大/缩量'，也不要输出具体的'X 亿'数值。",
            "涉及历史比较时，优先引用 time_context 中的 previous_trade_date、previous_analysis_generated_at 与 current_trade_date。",
            "如果 market_context.previous_daily_bar 存在，优先引用其中的 trade_date 作为上一交易日，不要直接拿上一次分析日期代替上一交易日。",
            "默认不要使用'昨日'或'今日'这类相对时间词；只有在 current_trade_date 与上一交易日明确相邻时，才允许写'上一交易日'，否则统一写成具体日期或'上次分析'。",
            "跨周末或节假日时，禁止把上次分析直接表述为'昨日'。",
            "必须同时给出短线、中线、长线三层趋势判断，并分别解释依据。",
            "短线优先结合 minute_summary、market_context.current_daily_bar、market_context.previous_daily_bar 与 market_context.recent_daily_bars。",
            "中线优先结合 market_context.recent_daily_bars 与关键价位变化。",
            "长线优先结合 market_context.current_weekly_bar、market_context.previous_weekly_bar 与 market_context.recent_weekly_bars。",
            "解释哪些价格、量能、资金流数据支持当前结论，描述尽量短句化。",
            "如果历史观点被推翻，指出被推翻的原因。",
            "必须给出 next_1d、next_3d、next_5d 三个预测窗口的方向判断与信心分数。",
            "输出结构化交易决策，包括信号、入场区、止损位、止盈位和观点失效条件。",
            "如果暂不适合买入或卖出，signal 应明确给出 hold 或 avoid。",
            "如果 signal 为 avoid，不强制提供 entry_zone、stop_loss、take_profit；当判断属于'等待入场'或'观察回踩/突破'时，可以给出一个参考 entry_zone 作为关注区间。",
            "对于 avoid，若当前结论只是继续观望且没有明确触发位，可以留空 entry_zone、stop_loss、take_profit。",
            "对于 buy、hold、reduce、sell，优先结合支撑/阻力/最近高低点给出可执行价位：entry_zone 尽量提供 low 和 high，stop_loss 尽量提供具体数值，take_profit 至少给出一个目标位。",
            "如果 symbol_context.is_default_pool_symbol 为 true，这只股票视为重点跟踪标的，优先给出可执行的仓位管理意见，不要轻易输出泛化观望。",
            "如果 symbol_context.position_status 为 holding，优先在 hold、reduce、sell 中做决策；只有明确适合继续加仓时才给 buy，并尽量提供 stop_loss、take_profit 与失效条件。",
            "如果 symbol_context.position_status 为 watching，优先在 buy 与 avoid 中做决策；若暂不入场但存在明确触发位，可使用 avoid 并给出 entry_zone 作为观察区间。",
            "如果 symbol_context.is_default_pool_symbol 为 true 且 position_status 为空，也要尽量给出可执行参考；只有在边界不清晰时才使用 avoid。",
            "给出可执行但克制的操作建议，不要承诺收益。",
            "trend_judgement 控制在 80 字内。",
            "trend_breakdown 中每个 reason 控制在 110 字内。",
            "operation_advice 控制在 90 字内。",
            "risk_warning 最多 4 条，每条控制在 60 字内。",
            "observation_points 最多 4 条，每条控制在 60 字内。",
            "decision.rationale 控制在 120 字内，decision.evidence 最多 4 条。",
            "prediction_windows 每条 rationale 控制在 90 字内。",
            "summary_markdown 控制在 320 字内，memory.summary 控制在 180 字内。",
        ],
    }

    return system_prompt, json.dumps(user_payload, ensure_ascii=False, indent=2)


def _build_time_context(
    *,
    phase: AnalysisPhase,
    snapshot: PriceSnapshot,
    previous_memory: Optional[MemorySummary],
    previous_record: Optional[HistoricalAnalysisRecord],
    market_context: Optional[dict[str, object]],
    previous_trading_snapshot: Optional[dict[str, object]],
) -> dict[str, object]:
    previous_trade_date = previous_record.snapshot.trade_date if previous_record else None
    previous_generated_at = previous_record.generated_at if previous_record else (
        previous_memory.generated_at if previous_memory else None
    )
    current_trade_date = snapshot.trade_date
    previous_market_trade_date = _extract_trade_date(
        (market_context or {}).get("previous_daily_bar") if isinstance(market_context, dict) else previous_trading_snapshot
    )

    return {
        "current_phase": phase,
        "current_trade_date": current_trade_date,
        "current_trade_date_label": _format_trade_date(current_trade_date),
        "previous_analysis_phase": (
            previous_record.report.phase if previous_record else (previous_memory.phase if previous_memory else None)
        ),
        "previous_trade_date": previous_trade_date,
        "previous_trade_date_label": _format_trade_date(previous_trade_date),
        "previous_market_trade_date": previous_market_trade_date,
        "previous_market_trade_date_label": _format_trade_date(previous_market_trade_date),
        "previous_analysis_generated_at": (
            previous_generated_at.isoformat() if isinstance(previous_generated_at, datetime) else None
        ),
        "calendar_day_gap": _calculate_calendar_day_gap(previous_trade_date, current_trade_date),
        "has_previous_analysis": bool(previous_memory or previous_record),
        "has_previous_trading_snapshot": bool(previous_market_trade_date),
    }


def _format_trade_date(value: Optional[str]) -> Optional[str]:
    if not value or len(value) != 8:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def _calculate_calendar_day_gap(previous_trade_date: Optional[str], current_trade_date: Optional[str]) -> Optional[int]:
    if not previous_trade_date or not current_trade_date:
        return None
    try:
        previous_day = datetime.strptime(previous_trade_date, "%Y%m%d")
        current_day = datetime.strptime(current_trade_date, "%Y%m%d")
    except ValueError:
        return None
    return (current_day - previous_day).days


def _extract_trade_date(previous_trading_snapshot: Optional[dict[str, object]]) -> Optional[str]:
    if not isinstance(previous_trading_snapshot, dict):
        return None
    value = previous_trading_snapshot.get("trade_date")
    if value is None:
        return None
    text = str(value).strip()
    return text or None

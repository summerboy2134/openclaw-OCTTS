from __future__ import annotations

import json
from datetime import datetime

from typing import Optional

from octts.schemas.report import AnalysisPhase, HistoricalAnalysisRecord, MemorySummary, PriceSnapshot


def build_report_prompt(
    *,
    phase: AnalysisPhase,
    snapshot: PriceSnapshot,
    previous_memory: Optional[MemorySummary],
    previous_record: Optional[HistoricalAnalysisRecord] = None,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严谨的 A 股量化复盘分析助手。"
        "你必须基于历史结论与当前数据做连续性判断，不允许忽略上次观点。"
        "你只能输出一个合法 JSON 对象。"
        "不要输出 Markdown，不要输出代码块，不要输出解释文字。"
        "所有 key 必须使用双引号。"
        "如果缺少信息，请使用 null、空数组或保守结论，不要省略必填字段。"
        "输出要精炼，优先保留关键信号、关键价位和主要依据。"
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
        "time_context": _build_time_context(
            phase=phase,
            snapshot=snapshot,
            previous_memory=previous_memory,
            previous_record=previous_record,
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
            "涉及历史比较时，优先引用 time_context 中的 previous_trade_date、previous_analysis_generated_at 与 current_trade_date。",
            "默认不要使用'昨日'或'今日'这类相对时间词，除非能根据 time_context 明确确认就是相邻交易日；否则统一写成具体日期或'上次分析/上一交易日'。",
            "跨周末或节假日时，禁止把上次分析直接表述为'昨日'。",
            "必须同时给出短线、中线、长线三层趋势判断，并分别解释依据。",
            "短线优先结合 minute_summary 与最近 5 个交易日；中线结合 daily_summary；长线结合 weekly_summary。",
            "解释哪些价格、量能、资金流数据支持当前结论，描述尽量短句化。",
            "如果历史观点被推翻，指出被推翻的原因。",
            "必须给出 next_1d、next_3d、next_5d 三个预测窗口的方向判断与信心分数。",
            "输出结构化交易决策，包括信号、入场区、止损位、止盈位和观点失效条件。",
            "如果暂不适合买入或卖出，signal 应明确给出 hold 或 avoid。",
            "给出可执行但克制的操作建议，不要承诺收益。",
            "trend_judgement 控制在 50 字内。",
            "trend_breakdown 中每个 reason 控制在 70 字内。",
            "operation_advice 控制在 60 字内。",
            "risk_warning 最多 3 条，每条控制在 45 字内。",
            "observation_points 最多 3 条，每条控制在 45 字内。",
            "decision.rationale 控制在 80 字内，decision.evidence 最多 3 条。",
            "prediction_windows 每条 rationale 控制在 60 字内。",
            "summary_markdown 控制在 200 字内，memory.summary 控制在 120 字内。",
        ],
    }

    return system_prompt, json.dumps(user_payload, ensure_ascii=False, indent=2)


def _build_time_context(
    *,
    phase: AnalysisPhase,
    snapshot: PriceSnapshot,
    previous_memory: Optional[MemorySummary],
    previous_record: Optional[HistoricalAnalysisRecord],
) -> dict[str, object]:
    previous_trade_date = previous_record.snapshot.trade_date if previous_record else None
    previous_generated_at = previous_record.generated_at if previous_record else (
        previous_memory.generated_at if previous_memory else None
    )
    current_trade_date = snapshot.trade_date

    return {
        "current_phase": phase,
        "current_trade_date": current_trade_date,
        "current_trade_date_label": _format_trade_date(current_trade_date),
        "previous_analysis_phase": (
            previous_record.report.phase if previous_record else (previous_memory.phase if previous_memory else None)
        ),
        "previous_trade_date": previous_trade_date,
        "previous_trade_date_label": _format_trade_date(previous_trade_date),
        "previous_analysis_generated_at": (
            previous_generated_at.isoformat() if isinstance(previous_generated_at, datetime) else None
        ),
        "calendar_day_gap": _calculate_calendar_day_gap(previous_trade_date, current_trade_date),
        "has_previous_analysis": bool(previous_memory or previous_record),
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

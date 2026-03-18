import json

from octts.prompts.report_prompt import build_report_prompt
from octts.schemas.report import MemorySummary, PriceSnapshot


def test_build_report_prompt_marks_initial_analysis_without_memory() -> None:
    _, user_prompt = build_report_prompt(
        phase="morning",
        snapshot=PriceSnapshot(ts_code="600000.SH", close=10.5, pct_chg=1.2),
        previous_memory=None,
    )

    payload = json.loads(user_prompt)
    assert payload["previous_memory"]["status"] == "initial_analysis"


def test_build_report_prompt_embeds_previous_memory() -> None:
    memory = MemorySummary(
        ts_code="600000.SH",
        phase="review",
        trend_bias="bullish",
        short_term_bias="bullish",
        mid_term_bias="bullish",
        long_term_bias="neutral",
        capital_flow_view="main inflow improving",
        confidence_score=0.75,
        summary="trend remains constructive",
    )

    _, user_prompt = build_report_prompt(
        phase="afternoon",
        snapshot=PriceSnapshot(ts_code="600000.SH", close=10.5, pct_chg=1.2),
        previous_memory=memory,
    )

    payload = json.loads(user_prompt)
    assert payload["previous_memory"]["trend_bias"] == "bullish"
    assert "snapshot.amount" in payload["field_unit_hints"]
    assert "千元" in payload["field_unit_hints"]["snapshot.amount"]
    assert payload["output_schema"]["decision"]["signal"] == "buy|hold|reduce|sell|avoid"
    assert payload["output_schema"]["trend_breakdown"]["short_term"] == "bullish|neutral|bearish"
    assert payload["output_schema"]["prediction_windows"][0]["window"] == "next_1d|next_3d|next_5d"
    assert any("entry_zone" in item for item in payload["analysis_instructions"])
    assert any("千元" in item and "amount" in item for item in payload["analysis_instructions"])
    assert any("signal 为 avoid" in item and "不强制提供" in item for item in payload["analysis_instructions"])
    assert any("等待入场" in item and "entry_zone" in item for item in payload["analysis_instructions"])

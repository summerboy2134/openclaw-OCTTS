import json

from octts.clients.llm_client import _coerce_structured_payload, _extract_json, _extract_snapshot_amount_yi


def test_extract_json_supports_plain_json() -> None:
    payload = _extract_json('{"ts_code":"600000.SH","phase":"review"}')
    assert payload["ts_code"] == "600000.SH"


def test_extract_json_supports_code_fence_wrapped_json() -> None:
    payload = _extract_json('```json\n{"ts_code":"600000.SH","phase":"review"}\n```')
    assert payload["phase"] == "review"


def test_extract_json_repairs_trailing_commas() -> None:
    payload = _extract_json('{"items":[1,2,],"meta":{"ok":true,}}')
    assert payload["items"] == [1, 2]
    assert payload["meta"]["ok"] is True


def test_extract_json_repairs_missing_comma() -> None:
    payload = _extract_json('{"ts_code":"600000.SH","prediction_windows":[{"window":"next_1d","bias":"bullish""confidence_score":0.7}]}')
    assert payload["ts_code"] == "600000.SH"
    assert payload["prediction_windows"][0]["confidence_score"] == 0.7


def test_coerce_structured_payload_builds_missing_memory() -> None:
    payload = _coerce_structured_payload(
        {
            "ts_code": "600309.SH",
            "phase": "afternoon",
            "trend_judgement": "空头趋势延续。",
            "trend_breakdown": {
                "short_term": "bearish",
                "mid_term": "bearish",
                "long_term": "neutral",
                "short_term_reason": "短线走弱",
                "mid_term_reason": "中线承压",
                "long_term_reason": "长线震荡",
            },
            "previous_view_status": "confirmed",
            "operation_advice": "继续观望。",
            "risk_warning": ["若跌破 83.0 元，风险加大。"],
            "observation_points": ["关注 83.0 元支撑", "关注 86.5 元阻力"],
            "summary_markdown": "**空头趋势延续**，继续观察支撑。",
            "decision": {
                "signal": "avoid",
                "rationale": "趋势偏弱。",
                "entry_zone": {"low": None, "high": None},
                "stop_loss": None,
                "take_profit": [],
                "invalidation_condition": "重新站上 86.5 元。",
                "holding_horizon": "swing",
                "confidence_score": 0.82,
                "risk_reward_ratio": None,
                "evidence": ["主力资金持续流出。"],
            },
            "prediction_windows": [],
        }
    )

    assert payload["memory"]["ts_code"] == "600309.SH"
    assert payload["memory"]["trend_bias"] == "bearish"
    assert payload["memory"]["confidence_score"] == 0.82


def test_extract_snapshot_amount_yi_from_prompt_payload() -> None:
    amount_yi = _extract_snapshot_amount_yi('{"snapshot":{"amount":128386.959}}')
    assert round(amount_yi or 0, 3) == 1.284


def test_coerce_structured_payload_corrects_amount_scale_errors() -> None:
    payload = _coerce_structured_payload(
        {
            "ts_code": "600000.SH",
            "phase": "review",
            "trend_judgement": "放量反弹，成交额12.84亿元，短线修复延续。",
            "trend_breakdown": {
                "short_term": "bullish",
                "mid_term": "neutral",
                "long_term": "neutral",
                "short_term_reason": "成交额12.84亿，量能回暖。",
                "mid_term_reason": "结构仍待确认。",
                "long_term_reason": "周线仍在整理区间。",
            },
            "previous_view_status": "confirmed",
            "operation_advice": "若成交额维持在12.84亿元附近，可继续观察。",
            "risk_warning": ["若成交额回落至12.84亿以下，修复力度或减弱。"],
            "observation_points": ["关注成交额12.84亿元能否延续。"],
            "summary_markdown": "成交额12.84亿元，价格反弹。",
            "decision": {
                "signal": "hold",
                "rationale": "成交额12.84亿元带动反弹。",
                "entry_zone": {"low": 10.0, "high": 10.3},
                "stop_loss": 9.8,
                "take_profit": [10.6],
                "invalidation_condition": "成交额12.84亿无法延续。",
                "holding_horizon": "swing",
                "confidence_score": 0.7,
                "risk_reward_ratio": 1.8,
                "evidence": ["成交额12.84亿明显放大。"],
            },
            "prediction_windows": [
                {
                    "window": "next_1d",
                    "bias": "neutral",
                    "confidence_score": 0.55,
                    "rationale": "成交额12.84亿元若延续，次日仍可观察。",
                }
            ],
            "memory": {
                "ts_code": "600000.SH",
                "phase": "review",
                "trend_bias": "neutral",
                "short_term_bias": "bullish",
                "mid_term_bias": "neutral",
                "long_term_bias": "neutral",
                "support_levels": [10.0],
                "resistance_levels": [10.6],
                "capital_flow_view": "成交额12.84亿元，资金有所回流。",
                "key_risks": ["成交额12.84亿不可持续。"],
                "next_checkpoints": ["复核成交额12.84亿元是否继续放大。"],
                "confidence_score": 0.7,
                "summary": "成交额12.84亿元，等待确认。",
            },
        },
        expected_snapshot_amount_yi=1.284,
    )

    assert "12.84亿" not in json.dumps(payload, ensure_ascii=False)
    assert "12.84亿元" not in json.dumps(payload, ensure_ascii=False)
    assert payload["trend_judgement"] == "放量反弹，成交额1.284亿元，短线修复延续。"
    assert payload["memory"]["summary"] == "成交额1.284亿元，等待确认。"

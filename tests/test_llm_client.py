from octts.clients.llm_client import _coerce_structured_payload, _extract_json


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

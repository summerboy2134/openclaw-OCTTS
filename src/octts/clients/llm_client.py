from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from json_repair import repair_json

from octts.config import Settings
from octts.schemas.report import StructuredAnalysis


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed.") from exc

        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self._settings = settings

    def analyze(self, *, system_prompt: str, user_prompt: str) -> StructuredAnalysis:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Optional[Exception] = None
        use_json_mode = self._settings.llm_json_mode
        total_attempts = 1 + max(0, self._settings.llm_retry_attempts)
        max_tokens = self._settings.llm_max_tokens

        for attempt in range(total_attempts):
            try:
                content, finish_reason = self._request_content(
                    messages=messages,
                    use_json_mode=use_json_mode,
                    max_tokens=max_tokens,
                )
                payload = _extract_json(content)
                payload = _coerce_structured_payload(payload)
                analysis = StructuredAnalysis.model_validate(payload)
                if finish_reason == "length" and attempt < total_attempts - 1:
                    raise ValueError("Model output was truncated due to token limit.")
                return analysis
            except Exception as exc:
                last_error = exc
                _write_debug_response(content=locals().get("content", ""), error=str(exc))
                if attempt == total_attempts - 1:
                    break
                messages = _build_repair_messages(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    invalid_content=locals().get("content", ""),
                    error=str(exc),
                )
                use_json_mode = False
                max_tokens = _next_retry_max_tokens(max_tokens)

        raise ValueError(f"LLM structured output parsing failed: {last_error}") from last_error

    def _request_content(
        self,
        *,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        max_tokens: int,
    ) -> tuple[str, Optional[str]]:
        request_kwargs: dict[str, Any] = {
            "model": self._settings.llm_model,
            "temperature": self._settings.llm_temperature,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
        if use_json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if use_json_mode and _looks_like_json_mode_not_supported(exc):
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("response_format", None)
                response = self._client.chat.completions.create(**fallback_kwargs)
            else:
                raise

        choice = response.choices[0]
        return choice.message.content or "{}", getattr(choice, "finish_reason", None)


def _extract_json(content: str) -> dict[str, Any]:
    normalized = _normalize_json_candidate(content)
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response did not contain valid JSON.") from None
        candidate = normalized[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _cleanup_common_json_issues(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                repaired_object = repair_json(repaired, return_objects=True)
                if not isinstance(repaired_object, dict):
                    raise ValueError("LLM response could not be repaired into a JSON object.") from None
                return repaired_object


def _normalize_json_candidate(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    return stripped


def _cleanup_common_json_issues(content: str) -> str:
    repaired = content.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    repaired = repaired.replace(",]", "]").replace(",}", "}")
    return repaired


def _build_repair_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    invalid_content: str,
    error: str,
) -> list[dict[str, str]]:
    repair_prompt = (
        "你上一次返回的内容存在格式或字段问题。"
        "请基于原始任务重新输出一个且仅一个合法 JSON 对象。"
        "不要解释，不要 Markdown，不要代码块。"
        "必须包含这些顶层字段："
        "ts_code, phase, trend_judgement, trend_breakdown, previous_view_status, "
        "operation_advice, risk_warning, observation_points, summary_markdown, "
        "decision, prediction_windows, memory。"
        "其中 memory 为必填对象，不允许缺失。"
        f"解析错误：{error}\n"
        f"原始任务：\n{user_prompt}\n"
        f"上一次返回：\n{invalid_content}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_prompt},
    ]


def _looks_like_json_mode_not_supported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message or "json schema" in message


def _coerce_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(payload)
    memory_payload = coerced.get("memory")
    if not isinstance(memory_payload, dict):
        coerced["memory"] = _build_memory_fallback(coerced)
    else:
        coerced["memory"] = _merge_memory_defaults(coerced, memory_payload)
    return coerced


def _build_memory_fallback(payload: dict[str, Any]) -> dict[str, Any]:
    trend_breakdown = payload.get("trend_breakdown") if isinstance(payload.get("trend_breakdown"), dict) else {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    signal = decision.get("signal")
    trend_bias = _bias_from_signal(signal) or trend_breakdown.get("mid_term") or "neutral"
    support_levels = _extract_numeric_levels(payload.get("observation_points"), prefer_keywords=("支撑",))
    resistance_levels = _extract_numeric_levels(payload.get("observation_points"), prefer_keywords=("阻力", "压力"))
    if not resistance_levels:
        resistance_levels = _safe_float_list(decision.get("take_profit"))

    return {
        "ts_code": payload.get("ts_code"),
        "phase": payload.get("phase"),
        "trend_bias": trend_bias,
        "short_term_bias": trend_breakdown.get("short_term"),
        "mid_term_bias": trend_breakdown.get("mid_term"),
        "long_term_bias": trend_breakdown.get("long_term"),
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "capital_flow_view": _build_capital_flow_view(payload),
        "key_risks": payload.get("risk_warning") or [],
        "next_checkpoints": payload.get("observation_points") or [],
        "confidence_score": decision.get("confidence_score") if isinstance(decision.get("confidence_score"), (int, float)) else 0.5,
        "summary": _build_memory_summary(payload),
    }


def _merge_memory_defaults(payload: dict[str, Any], memory_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(memory_payload)
    fallback = _build_memory_fallback(payload)
    for key, value in fallback.items():
        current = merged.get(key)
        if current in (None, "", []):
            merged[key] = value
    return merged


def _build_capital_flow_view(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    evidence = decision.get("evidence") if isinstance(decision.get("evidence"), list) else []
    for item in evidence:
        text = str(item)
        if "资金" in text or "主力" in text:
            return text
    return str(payload.get("trend_judgement") or "资金流向信息不足，维持谨慎判断。")


def _build_memory_summary(payload: dict[str, Any]) -> str:
    summary_markdown = str(payload.get("summary_markdown") or "")
    plain_text = re.sub(r"[*#>`_]", "", summary_markdown).strip()
    if plain_text:
        return plain_text[:180]
    return str(payload.get("trend_judgement") or "暂无摘要")


def _extract_numeric_levels(value: Any, *, prefer_keywords: tuple[str, ...]) -> list[float]:
    items = value if isinstance(value, list) else []
    results: list[float] = []
    for item in items:
        text = str(item)
        if prefer_keywords and not any(keyword in text for keyword in prefer_keywords):
            continue
        for match in re.findall(r"\d+(?:\.\d+)?", text):
            try:
                results.append(float(match))
            except ValueError:
                continue
    return results[:5]


def _safe_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _bias_from_signal(signal: Any) -> Optional[str]:
    mapping = {
        "buy": "bullish",
        "hold": "neutral",
        "reduce": "bearish",
        "sell": "bearish",
        "avoid": "bearish",
    }
    return mapping.get(str(signal))


def _next_retry_max_tokens(current: int) -> int:
    return min(int(current * 1.5), 4000)


def _write_debug_response(*, content: str, error: str) -> None:
    debug_dir = Path("memory/llm_debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "last_invalid_response.txt"
    debug_path.write_text(
        f"error: {error}\n\n{content}",
        encoding="utf-8",
    )

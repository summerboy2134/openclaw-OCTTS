from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc
from typing import Optional
from uuid import uuid4

from octts.clients.llm_client import LLMClient
from octts.clients.tushare_client import TushareClient
from octts.clients.wecom_client import WeComClient
from octts.config import Settings
from octts.prompts.report_prompt import build_report_prompt
from octts.schemas.report import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisPhase,
    HistoricalAnalysisRecord,
    MemorySummary,
    PriceSnapshot,
    SymbolAnalysisError,
    StructuredAnalysis,
    ValidationUpdate,
)
from octts.services.history_store import FileHistoryStore, build_initial_validation
from octts.services.memory_store import MemoryStore

TREND_BIAS_LABELS = {
    "bullish": "看多",
    "neutral": "中性",
    "bearish": "看空",
}

SIGNAL_LABELS = {
    "buy": "买入",
    "hold": "持有",
    "reduce": "减仓",
    "sell": "卖出",
    "avoid": "观望",
}

PREDICTION_WINDOW_LABELS = {
    "next_1d": "未来1个交易日",
    "next_3d": "未来3个交易日",
    "next_5d": "未来5个交易日",
}

PREVIOUS_VIEW_STATUS_LABELS = {
    "confirmed": "延续确认",
    "weakened": "有所减弱",
    "reversed": "观点反转",
    "initial": "首次分析",
}


class AnalysisPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        tushare_client: TushareClient,
        llm_client: LLMClient,
        memory_store: MemoryStore,
        history_store: FileHistoryStore,
        wecom_client: Optional[WeComClient] = None,
    ) -> None:
        self._settings = settings
        self._tushare_client = tushare_client
        self._llm_client = llm_client
        self._memory_store = memory_store
        self._history_store = history_store
        self._wecom_client = wecom_client

    def run(self, request: AnalysisRequest) -> AnalysisResult:
        stock_pool = request.stock_pool or self._settings.stock_pool
        if not stock_pool:
            raise ValueError("No stock pool provided. Set OCTTS_STOCK_POOL or pass stock_pool in request.")

        reports: list[StructuredAnalysis] = []
        raw_payloads: dict[str, object] = {}
        validation_updates: list[ValidationUpdate] = []
        errors: list[SymbolAnalysisError] = []
        request_id = str(uuid4())
        now = datetime.now(UTC)
        allow_partial_success = len(stock_pool) > 1

        for ts_code in stock_pool:
            previous_memory = None
            previous_record = None
            market_context = None
            previous_trading_snapshot = None
            snapshot = None
            system_prompt = None
            user_prompt = None
            try:
                snapshot = self._tushare_client.fetch_snapshot(
                    ts_code=ts_code,
                    phase=request.phase,
                    trade_date=request.trade_date,
                )
                validation_updates.extend(
                    self._history_store.refresh_validations(
                        ts_code=ts_code,
                        snapshot=snapshot,
                    )
                )
                previous_record = self._history_store.get_latest_record(
                    ts_code,
                    phase="review",
                    before_trade_date=snapshot.trade_date,
                )
                previous_memory = previous_record.report.memory if previous_record else None
                market_context = _build_market_context(snapshot)
                previous_trading_snapshot = market_context.get("previous_daily_bar")
                system_prompt, user_prompt, report = self.generate_report_from_snapshot(
                    phase=request.phase,
                    snapshot=snapshot,
                    previous_memory=previous_memory,
                    previous_record=previous_record,
                    market_context=market_context,
                    previous_trading_snapshot=previous_trading_snapshot,
                )
                self._memory_store.set(report.memory)
                reports.append(report)
                self._history_store.append(
                    HistoricalAnalysisRecord(
                        record_id=str(uuid4()),
                        request_id=request_id,
                        generated_at=now,
                        snapshot=snapshot,
                        report=report,
                        validation=build_initial_validation(
                            decision=report.decision,
                            snapshot=snapshot,
                        ),
                    )
                )
                raw_payloads[ts_code] = {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "previous_memory": previous_memory.model_dump(mode="json") if previous_memory else None,
                    "previous_record": previous_record.model_dump(mode="json") if previous_record else None,
                    "market_context": market_context,
                    "previous_trading_snapshot": previous_trading_snapshot,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            except Exception as exc:
                raw_payloads[ts_code] = {
                    "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
                    "previous_memory": previous_memory.model_dump(mode="json") if previous_memory else None,
                    "previous_record": previous_record.model_dump(mode="json") if previous_record else None,
                    "market_context": market_context,
                    "previous_trading_snapshot": previous_trading_snapshot,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "error": str(exc),
                }
                if not allow_partial_success:
                    raise
                errors.append(SymbolAnalysisError(ts_code=ts_code, phase=request.phase, error=str(exc)))
                continue

        notification_sent = False
        if request.notify and self._wecom_client and reports:
            self._wecom_client.send_markdown(format_reports_as_markdown(reports))
            notification_sent = True

        return AnalysisResult(
            request_id=request_id,
            phase=request.phase,
            reports=reports,
            notification_sent=notification_sent,
            validation_updates=validation_updates,
            errors=errors,
            raw_payloads=raw_payloads,
        )

    def generate_report_from_snapshot(
        self,
        *,
        phase: AnalysisPhase,
        snapshot: PriceSnapshot,
        previous_memory: Optional[MemorySummary],
        previous_record: Optional[HistoricalAnalysisRecord] = None,
        market_context: Optional[dict[str, object]] = None,
        previous_trading_snapshot: Optional[dict[str, object]] = None,
    ) -> tuple[str, str, StructuredAnalysis]:
        market_context = market_context or _build_market_context(snapshot)
        previous_trading_snapshot = previous_trading_snapshot or market_context.get("previous_daily_bar")
        system_prompt, user_prompt = build_report_prompt(
            phase=phase,
            snapshot=snapshot,
            previous_memory=previous_memory,
            previous_record=previous_record,
            market_context=market_context,
            previous_trading_snapshot=previous_trading_snapshot,
        )
        report = self._llm_client.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return system_prompt, user_prompt, report


def format_reports_as_markdown(reports: list[StructuredAnalysis]) -> str:
    lines = ["# OCTTS 自动分析报告", ""]
    for report in reports:
        trend_breakdown = (
            f"短线 {_translate_trend_bias(report.trend_breakdown.short_term)} / "
            f"中线 {_translate_trend_bias(report.trend_breakdown.mid_term)} / "
            f"长线 {_translate_trend_bias(report.trend_breakdown.long_term)}"
        )
        prediction_windows = "；".join(
            f"{_translate_prediction_window(item.window)}："
            f"{_translate_trend_bias(item.bias)} ({item.confidence_score:.0%})"
            for item in report.prediction_windows
        ) or "无"
        lines.extend(
            [
                f"## {report.ts_code}",
                f"> 趋势判断：{report.trend_judgement}",
                f"> 三层趋势：{trend_breakdown}",
                f"> 历史观点状态：{_translate_previous_view_status(report.previous_view_status)}",
                f"> 操作建议：{report.operation_advice}",
                f"> 交易信号：{_translate_signal(report.decision.signal)}",
                f"> 入场区间：{_format_entry_zone(report)}",
                f"> 止损位：{_format_stop_loss(report)}",
                f"> 目标位：{_format_take_profit(report)}",
                f"> 预测窗口：{prediction_windows}",
                "",
                report.summary_markdown,
                "",
                f"风险预警：{'；'.join(report.risk_warning) if report.risk_warning else '无'}",
                f"观察要点：{'；'.join(report.observation_points) if report.observation_points else '无'}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _format_entry_zone(report: StructuredAnalysis) -> str:
    zone = report.decision.entry_zone
    if zone is None:
        return "观望" if report.decision.signal == "avoid" else "未设置"
    low = zone.low if zone.low is not None else "-"
    high = zone.high if zone.high is not None else "-"
    if report.decision.signal == "avoid" and low == "-" and high == "-":
        return "观望"
    return f"{low} - {high}"


def _format_take_profit(report: StructuredAnalysis) -> str:
    if not report.decision.take_profit:
        return "未设置"
    return " / ".join(str(item) for item in report.decision.take_profit)


def _format_stop_loss(report: StructuredAnalysis) -> str:
    return str(report.decision.stop_loss) if report.decision.stop_loss is not None else "未设置"


def _translate_trend_bias(value: str) -> str:
    return TREND_BIAS_LABELS.get(value, value)


def _translate_signal(value: str) -> str:
    return SIGNAL_LABELS.get(value, value)


def _translate_prediction_window(value: str) -> str:
    return PREDICTION_WINDOW_LABELS.get(value, value)


def _translate_previous_view_status(value: str) -> str:
    return PREVIOUS_VIEW_STATUS_LABELS.get(value, value)


def _build_market_context(snapshot: PriceSnapshot) -> dict[str, object]:
    daily_bars = _normalize_bar_series(snapshot.daily_summary)
    current_daily_bar = _snapshot_to_daily_bar(snapshot)
    if current_daily_bar:
        daily_bars = _merge_bar_series(daily_bars, [current_daily_bar])
    previous_daily_bar = _pick_previous_bar(daily_bars, snapshot.trade_date)

    weekly_bars = _normalize_bar_series(snapshot.weekly_summary)
    current_weekly_bar = weekly_bars[-1] if weekly_bars else None
    previous_weekly_bar = weekly_bars[-2] if len(weekly_bars) >= 2 else None

    return {
        "current_daily_bar": current_daily_bar,
        "previous_daily_bar": previous_daily_bar,
        "recent_daily_bars": daily_bars[-5:],
        "current_weekly_bar": current_weekly_bar,
        "previous_weekly_bar": previous_weekly_bar,
        "recent_weekly_bars": weekly_bars[-8:],
    }


def _normalize_bar_series(raw_bars: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for item in raw_bars:
        if not isinstance(item, dict):
            continue
        bar = _serialize_bar(item)
        trade_date = bar.get("trade_date")
        if trade_date:
            normalized[str(trade_date)] = bar
    return sorted(normalized.values(), key=lambda item: str(item.get("trade_date") or ""))


def _merge_bar_series(
    existing_bars: list[dict[str, object]], incoming_bars: list[dict[str, object]]
) -> list[dict[str, object]]:
    merged = list(existing_bars)
    by_trade_date = {str(item.get("trade_date")): index for index, item in enumerate(merged) if item.get("trade_date")}
    for bar in incoming_bars:
        trade_date = bar.get("trade_date")
        if not trade_date:
            continue
        key = str(trade_date)
        if key in by_trade_date:
            merged[by_trade_date[key]] = bar
        else:
            by_trade_date[key] = len(merged)
            merged.append(bar)
    return sorted(merged, key=lambda item: str(item.get("trade_date") or ""))


def _snapshot_to_daily_bar(snapshot: PriceSnapshot) -> Optional[dict[str, object]]:
    trade_date = _normalize_trade_date(snapshot.trade_date)
    if not trade_date:
        return None
    return {
        "trade_date": trade_date,
        "open": _safe_float(snapshot.open),
        "high": _safe_float(snapshot.high),
        "low": _safe_float(snapshot.low),
        "close": _safe_float(snapshot.close),
        "pct_chg": _safe_float(snapshot.pct_chg),
        "amount": _safe_float(snapshot.amount),
        "vol": None,
        "turnover_rate": _safe_float(snapshot.turnover_rate),
        "vol_ratio": _safe_float(snapshot.vol_ratio),
    }


def _serialize_bar(raw_bar: dict[str, object]) -> dict[str, object]:
    return {
        "trade_date": _normalize_trade_date(raw_bar.get("trade_date")),
        "open": _safe_float(raw_bar.get("open")),
        "high": _safe_float(raw_bar.get("high")),
        "low": _safe_float(raw_bar.get("low")),
        "close": _safe_float(raw_bar.get("close")),
        "pct_chg": _safe_float(raw_bar.get("pct_chg")),
        "vol": _safe_float(raw_bar.get("vol")),
        "amount": _safe_float(raw_bar.get("amount")),
        "turnover_rate": _safe_float(raw_bar.get("turnover_rate")),
        "vol_ratio": _safe_float(raw_bar.get("vol_ratio")),
    }


def _pick_previous_bar(
    bars: list[dict[str, object]], current_trade_date: Optional[str]
) -> Optional[dict[str, object]]:
    normalized_current = _normalize_trade_date(current_trade_date)
    candidates = [
        item for item in bars if item.get("trade_date") and (not normalized_current or str(item["trade_date"]) < normalized_current)
    ]
    if not candidates:
        return None
    return candidates[-1]


def _extract_previous_trading_snapshot(snapshot: PriceSnapshot) -> Optional[dict[str, object]]:
    market_context = _build_market_context(snapshot)
    previous_daily_bar = market_context.get("previous_daily_bar")
    return previous_daily_bar if isinstance(previous_daily_bar, dict) else None


def _normalize_trade_date(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if len(text) == 8 and text.isdigit() else None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

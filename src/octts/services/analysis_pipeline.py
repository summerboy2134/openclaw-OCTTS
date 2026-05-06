from __future__ import annotations

import logging
from datetime import datetime, timezone

UTC = timezone.utc
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from octts.clients.llm_client import LLMClient
from octts.clients.tushare_client import TushareClient
from octts.clients.wecom_client import WeComClient
from octts.config import Settings
from octts.indicators.technical import build_technical_snapshot
from octts.prompts.report_prompt import build_report_prompt
from octts.schemas.report import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisPhase,
    HistoricalAnalysisRecord,
    MemorySummary,
    PositionStatus,
    PriceSnapshot,
    SymbolAnalysisError,
    StructuredAnalysis,
    ValidationUpdate,
)
from octts.services.history_store import FileHistoryStore, build_initial_validation
from octts.services.memory_store import MemoryStore
from octts.services.position_store import FilePositionStore
from octts.services.daily_analysis_context import DailyAnalysisScreeningContextProvider

logger = logging.getLogger(__name__)

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
        position_store: FilePositionStore,
        wecom_client: Optional[WeComClient] = None,
        screening_context_provider: Optional[DailyAnalysisScreeningContextProvider] = None,
    ) -> None:
        self._settings = settings
        self._tushare_client = tushare_client
        self._llm_client = llm_client
        self._memory_store = memory_store
        self._history_store = history_store
        self._position_store = position_store
        self._wecom_client = wecom_client
        self._screening_context_provider = screening_context_provider or (
            DailyAnalysisScreeningContextProvider(settings)
            if settings.screening_enabled
            else None
        )

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
        default_stock_pool = {item.strip().upper() for item in self._settings.stock_pool if item.strip()}

        for ts_code in stock_pool:
            previous_memory = None
            previous_record = None
            market_context = None
            screening_context = None
            previous_trading_snapshot = None
            snapshot = None
            system_prompt = None
            user_prompt = None
            position_status = self._position_store.get_status(ts_code)
            is_default_pool_symbol = ts_code.strip().upper() in default_stock_pool
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
                screening_context = self._build_screening_context(ts_code, snapshot=snapshot)
                system_prompt, user_prompt, report = self.generate_report_from_snapshot(
                    phase=request.phase,
                    snapshot=snapshot,
                    previous_memory=previous_memory,
                    previous_record=previous_record,
                    market_context=market_context,
                    previous_trading_snapshot=previous_trading_snapshot,
                    screening_context=screening_context,
                    is_default_pool_symbol=is_default_pool_symbol,
                    position_status=position_status,
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
                    "screening_context": screening_context,
                    "previous_trading_snapshot": previous_trading_snapshot,
                    "position_status": position_status,
                    "is_default_pool_symbol": is_default_pool_symbol,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            except Exception as exc:
                raw_payloads[ts_code] = {
                    "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
                    "previous_memory": previous_memory.model_dump(mode="json") if previous_memory else None,
                    "previous_record": previous_record.model_dump(mode="json") if previous_record else None,
                    "market_context": market_context,
                    "screening_context": screening_context,
                    "previous_trading_snapshot": previous_trading_snapshot,
                    "position_status": position_status,
                    "is_default_pool_symbol": is_default_pool_symbol,
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
        screening_context: Optional[dict[str, object]] = None,
        is_default_pool_symbol: bool = False,
        position_status: Optional[PositionStatus] = None,
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
            screening_context=screening_context,
            is_default_pool_symbol=is_default_pool_symbol,
            position_status=position_status,
        )
        report = self._llm_client.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return system_prompt, user_prompt, report

    def _build_screening_context(
        self,
        ts_code: str,
        *,
        snapshot: Optional[PriceSnapshot] = None,
    ) -> Optional[dict[str, object]]:
        context: Optional[dict[str, object]] = None
        if self._screening_context_provider is not None:
            try:
                context = self._screening_context_provider.build_for_symbol(ts_code)
            except Exception:
                context = {
                    "data_available": False,
                    "message": "智能选股上下文读取失败；本次仅基于个股行情、财务和历史观点分析。",
                }
        if snapshot is None:
            return context or None
        return self._augment_symbol_context(ts_code=ts_code, snapshot=snapshot, context=context)

    def _augment_symbol_context(
        self,
        *,
        ts_code: str,
        snapshot: PriceSnapshot,
        context: Optional[dict[str, object]],
    ) -> Optional[dict[str, object]]:
        enriched_context: Dict[str, Any] = dict(context or {})
        normalized_code = ts_code.strip().upper()
        existing_stock_context = enriched_context.get("stock_context")
        symbol_in_latest_pool = isinstance(existing_stock_context, dict) and bool(existing_stock_context)
        standalone_context = self._build_standalone_stock_context(ts_code=normalized_code, snapshot=snapshot)
        if not enriched_context and not standalone_context:
            return None
        enriched_context.setdefault("ts_code", normalized_code)
        enriched_context["symbol_in_latest_pool"] = symbol_in_latest_pool
        if standalone_context:
            enriched_context["standalone_stock_context"] = standalone_context
            if not symbol_in_latest_pool:
                standalone_stock_context = dict(standalone_context.get("stock_context") or {})
                if standalone_stock_context:
                    enriched_context["stock_context"] = standalone_stock_context
                    enriched_context["latest_pool_state"] = standalone_stock_context
                    enriched_context["message"] = (
                        "该股未出现在最新智能选股池中，已补充单股实时增强分析上下文。"
                    )
            enriched_context["data_available"] = True
        return enriched_context or None

    def _build_standalone_stock_context(
        self,
        *,
        ts_code: str,
        snapshot: PriceSnapshot,
    ) -> Optional[Dict[str, Any]]:
        technical_snapshot = _build_snapshot_technical_context(snapshot)
        moneyflow_context = _build_moneyflow_context(
            rows=self._tushare_client.fetch_moneyflow(ts_code, trade_date=snapshot.trade_date),
            fallback_summary=snapshot.moneyflow_summary,
        )
        top_list_rows = self._tushare_client.fetch_top_list(ts_code, trade_date=snapshot.trade_date)
        limit_list_row = self._tushare_client.fetch_limit_list(ts_code, trade_date=snapshot.trade_date)
        company_profile = self._tushare_client.fetch_company_profile(ts_code)
        earnings_forecast_rows = self._tushare_client.fetch_earnings_forecast(ts_code)
        stock_info = self._tushare_client.fetch_stock_info(ts_code)

        business_summary = _build_business_summary(company_profile=company_profile, stock_info=stock_info)
        top_list_summary = _summarize_top_list_rows(top_list_rows)
        limit_status = _summarize_limit_list_row(limit_list_row)
        earnings_forecast_summary = _summarize_earnings_forecast(earnings_forecast_rows)

        stock_context: Dict[str, Any] = {
            "ts_code": ts_code,
            "name": snapshot.name or stock_info.get("name"),
            "trade_date": snapshot.trade_date,
            "source_tag": "单股增强分析",
            "tracking_status": "standalone",
            "today_present": False,
            "absence_reason": "not_in_latest_screening_pool",
            "close": _safe_float(snapshot.close),
            "pct_change": _safe_float(snapshot.pct_chg),
            "turnover_rate": _safe_float(snapshot.turnover_rate),
            "volume_ratio": _safe_float(snapshot.vol_ratio),
            "industry": stock_info.get("industry"),
            "recommendation_score": technical_snapshot.get("recommendation_score"),
            "overall_score": technical_snapshot.get("setup_quality_score"),
            "priority_score": technical_snapshot.get("setup_quality_score"),
            "risk_score": technical_snapshot.get("risk_score"),
            "risk_level": technical_snapshot.get("risk_level"),
            "risk_flags": list(technical_snapshot.get("risk_flags") or []),
            "ma20": technical_snapshot.get("ma20"),
            "moneyflow_3d_value": moneyflow_context.get("recent_3d_net_inflow"),
            "recent_large_order_net_inflow": moneyflow_context.get("recent_large_order_net_inflow"),
            "recent_super_large_order_net_inflow": moneyflow_context.get("recent_super_large_order_net_inflow"),
            "top_list_summary": top_list_summary,
            "limit_status": limit_status,
            "business_summary": business_summary,
            "earnings_forecast_summary": earnings_forecast_summary,
            "standalone_score_context": True,
            "standalone_context_source": "tushare_single_stock",
        }
        stock_context = _drop_empty_values(stock_context)

        standalone_context: Dict[str, Any] = {
            "context_source": "tushare_single_stock",
            "symbol_in_latest_pool": False,
            "stock_context": stock_context,
            "technical_snapshot": technical_snapshot,
            "moneyflow_context": moneyflow_context,
            "market_event_context": _drop_empty_values(
                {
                    "top_list": top_list_rows,
                    "top_list_summary": top_list_summary,
                    "limit_list": limit_list_row,
                    "limit_status": limit_status,
                }
            ),
            "company_profile": _drop_empty_values(
                {
                    "industry": stock_info.get("industry"),
                    "market": stock_info.get("market"),
                    "area": stock_info.get("area"),
                    "list_date": stock_info.get("list_date"),
                    "business_summary": business_summary,
                    "employees": company_profile.get("employees"),
                    "website": company_profile.get("website"),
                }
            ),
            "earnings_context": _drop_empty_values(
                {
                    "earnings_forecast_summary": earnings_forecast_summary,
                    "earnings_forecast": earnings_forecast_rows[:2],
                }
            ),
        }
        return _drop_empty_values(standalone_context) or None



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


def _build_snapshot_technical_context(snapshot: PriceSnapshot) -> Dict[str, Any]:
    bars = _normalize_bar_series(snapshot.daily_summary)
    current_bar = _snapshot_to_daily_bar(snapshot)
    if current_bar:
        bars = _merge_bar_series(bars, [current_bar])
    if not bars:
        return {}
    closes = pd.Series([float(item.get("close") or 0.0) for item in bars], dtype="float64")
    highs = pd.Series(
        [float(item.get("high") or item.get("close") or 0.0) for item in bars],
        dtype="float64",
    )
    lows = pd.Series(
        [float(item.get("low") or item.get("close") or 0.0) for item in bars],
        dtype="float64",
    )
    volumes = pd.Series([float(item.get("vol") or 0.0) for item in bars], dtype="float64")
    technical = build_technical_snapshot(closes, highs, lows, volumes)
    return _drop_empty_values(
        {
            "close": technical.close,
            "ma5": technical.ma5,
            "ma10": technical.ma10,
            "ma20": technical.ma20,
            "ma60": technical.ma60,
            "rsi": technical.rsi,
            "macd": technical.macd,
            "macd_signal": technical.macd_signal,
            "macd_histogram": technical.macd_histogram,
            "volume_ratio": technical.volume_ratio,
            "price_position_20d": technical.price_position_20d,
            "breakout": technical.breakout,
            "trend_status": technical.trend_status,
            "momentum_status": technical.momentum_status,
            "technical_score": technical.technical_score,
            "trend_score": technical.trend_score,
            "momentum_score": technical.momentum_score,
            "volume_score": technical.volume_score,
            "breakout_score": technical.breakout_score,
            "risk_score": technical.risk_score,
            "setup_quality_score": technical.setup_quality_score,
            "recommendation_score": technical.recommendation_score,
            "recommendation": technical.recommendation,
            "setup_type": technical.setup_type,
            "risk_level": technical.risk_level,
            "entry_style": technical.entry_style,
            "confidence": technical.confidence,
            "risk_flags": list(technical.risk_flags or []),
            "setup_notes": list(technical.setup_notes or []),
            "distance_to_ma20_pct": technical.distance_to_ma20_pct,
            "distance_to_ma60_pct": technical.distance_to_ma60_pct,
            "breakout_strength": technical.breakout_strength,
        }
    )


def _build_moneyflow_context(
    *,
    rows: List[Dict[str, Any]],
    fallback_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sorted_rows = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda item: str(item.get("trade_date") or ""),
    )
    latest_rows = sorted_rows[-3:]
    recent_3d_net_inflow = 0.0
    recent_large_order_net_inflow = 0.0
    recent_super_large_order_net_inflow = 0.0
    for row in latest_rows:
        recent_3d_net_inflow += _safe_float(row.get("net_mf_amount")) or 0.0
        recent_large_order_net_inflow += (_safe_float(row.get("buy_lg_amount")) or 0.0) - (
            _safe_float(row.get("sell_lg_amount")) or 0.0
        )
        recent_super_large_order_net_inflow += (_safe_float(row.get("buy_elg_amount")) or 0.0) - (
            _safe_float(row.get("sell_elg_amount")) or 0.0
        )
    result = {
        "rows": len(sorted_rows),
        "recent_3d_net_inflow": round(recent_3d_net_inflow, 2) if latest_rows else None,
        "recent_large_order_net_inflow": round(recent_large_order_net_inflow, 2) if latest_rows else None,
        "recent_super_large_order_net_inflow": round(recent_super_large_order_net_inflow, 2) if latest_rows else None,
        "latest_net_mf_amount": _safe_float((sorted_rows[-1] if sorted_rows else {}).get("net_mf_amount")),
        "fallback_net_mf_amount": _safe_float((fallback_summary or {}).get("net_mf_amount")),
    }
    return _drop_empty_values(result)


def _build_business_summary(*, company_profile: Dict[str, Any], stock_info: Dict[str, Any]) -> Optional[str]:
    parts: List[str] = []
    main_business = str(company_profile.get("main_business") or "").strip()
    if main_business:
        parts.append(main_business)
    industry = str(stock_info.get("industry") or "").strip()
    if industry and industry not in "".join(parts):
        parts.append(f"所属行业{industry}")
    text = "；".join(part for part in parts if part)
    return text[:220] if text else None


def _summarize_earnings_forecast(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None
    latest = rows[0]
    summary = str(latest.get("summary") or "").strip()
    change_reason = str(latest.get("change_reason") or "").strip()
    forecast_type = str(latest.get("type") or "").strip()
    parts = [part for part in [forecast_type, summary, change_reason] if part]
    if not parts:
        return None
    text = "；".join(parts)
    return text[:220]


def _summarize_top_list_rows(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return None
    parts: List[str] = []
    for row in rows[:3]:
        reason = str(row.get("reason") or "龙虎榜").strip()
        net_amount = _safe_float(row.get("net_amount"))
        net_rate = _safe_float(row.get("net_rate"))
        text = reason
        if net_amount is not None:
            text += f"，净买入{net_amount:.1f}万"
        if net_rate is not None:
            text += f"，净买率{net_rate:.1f}%"
        parts.append(text)
    return "；".join(parts) if parts else None


def _summarize_limit_list_row(row: Optional[Dict[str, Any]]) -> Optional[str]:
    if not row:
        return None
    limit_status = str(row.get("limit") or "").strip()
    open_times = row.get("open_times")
    last_time = str(row.get("last_time") or "").strip()
    parts = [limit_status or "涨跌停异动"]
    if open_times not in (None, ""):
        try:
            parts.append(f"开板{int(float(open_times or 0))}次")
        except (TypeError, ValueError):
            logger.debug("Invalid limit list open_times: %s", open_times)
    if last_time:
        parts.append(f"最后封板/触及时间{last_time}")
    return "，".join(parts)


def _drop_empty_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }

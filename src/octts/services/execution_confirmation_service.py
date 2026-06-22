from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from typing import Any, Dict, List, Optional

from octts.clients.email_client import EmailClient
from octts.clients.tushare_client import TushareClient
from octts.config import Settings
from octts.services.history_store import FileHistoryStore
from octts.services.report_email_service import ReportEmailService
from octts.services.report_exporter import ReportExporter
from octts.services.screening_store import ScreeningStore
from octts.services.position_store import create_position_store
from octts.services.stock_screener import StockScreener

logger = logging.getLogger(__name__)


@dataclass
class ConfirmationDecision:
    ts_code: str
    name: str
    status: str
    reasons: List[str]
    event_rows: List[Dict[str, Any]]
    auction_rows: List[Dict[str, Any]]
    auction_label: Optional[str] = None
    auction_assessment: Dict[str, Any] = None
    original_rank: Optional[int] = None
    recommendation_score: Optional[float] = None


class ExecutionConfirmationService:
    """Pre-open execution confirmation for the previous candidate Top3."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: Optional[ScreeningStore] = None,
        email_service: Optional[ReportEmailService] = None,
    ) -> None:
        self.settings = settings
        self.store = store or ScreeningStore(settings)
        self.screener = StockScreener(settings)
        self.tushare_client = TushareClient(settings)
        self.email_service = email_service
        if self.email_service is None and settings.email_enabled:
            history_store = FileHistoryStore(settings.history_dir_path)
            self.email_service = ReportEmailService(
                settings=settings,
                history_store=history_store,
                report_exporter=ReportExporter(
                    settings=settings,
                    history_store=history_store,
                    position_store=create_position_store(settings),
                ),
                email_client=EmailClient(settings),
            )

    async def run_pre_open_confirmation(
        self,
        *,
        source_trade_date: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not self.settings.screening_execution_confirmation_enabled:
            return {"success": False, "skipped": True, "reason": "execution_confirmation_disabled"}

        window_status = self._check_run_window()
        if not force and not self.settings.screening_execution_confirmation_allow_non_window and not window_status["in_window"]:
            return {
                "success": False,
                "skipped": True,
                "reason": "outside_confirmation_window",
                "window": window_status,
            }

        target_trade_date = self._resolve_source_trade_date(source_trade_date)
        candidates = self._load_candidate_top3(target_trade_date)
        decisions = [self._confirm_candidate(item, target_trade_date=target_trade_date) for item in candidates]
        confirmed = [item for item in decisions if item.status == "confirmed_buy"]
        watch_only = [item for item in decisions if item.status == "watch_only"]
        vetoed = [item for item in decisions if item.status == "vetoed"]
        data_availability = self._build_data_availability_summary(candidates, decisions)

        payload = {
            "success": True,
            "workflow_mode": "pre_open_confirmation",
            "source_trade_date": target_trade_date.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "window": window_status,
            "candidate_count": len(candidates),
            "confirmed_count": len(confirmed),
            "watch_only_count": len(watch_only),
            "vetoed_count": len(vetoed),
            "data_availability": data_availability,
            "decisions": [self._serialize_decision(item) for item in decisions],
        }

        if self.settings.screening_execution_confirmation_notify:
            await self._send_confirmation_email(payload)
        return payload

    def _resolve_source_trade_date(self, source_trade_date: Optional[str]) -> date:
        if source_trade_date:
            return datetime.strptime(str(source_trade_date).replace("-", ""), "%Y%m%d").date()
        latest_trade_date = self.screener._get_latest_trade_date()
        latest_date = datetime.strptime(latest_trade_date, "%Y%m%d").date()
        previous_date = self.store.get_previous_recommendation_pool_trade_date(latest_date)
        return previous_date or latest_date

    def _load_candidate_top3(self, source_trade_date: date) -> List[Dict[str, Any]]:
        rows = self.store.list_recommendation_pool(trade_date=source_trade_date, front_only=True, limit=10)
        today_top = [row for row in rows if str(row.get("source_tag") or "") == "今日Top3"]
        candidates = today_top or rows[:3]
        return sorted(candidates, key=lambda item: int(item.get("frontlist_rank") or item.get("recommend_rank") or 9999))[:3]

    def _confirm_candidate(self, item: Dict[str, Any], *, target_trade_date: date) -> ConfirmationDecision:
        ts_code = str(item.get("ts_code") or "").strip().upper()
        name = str(item.get("name") or ts_code).strip()
        reasons: List[str] = []
        status = "confirmed_buy"
        event_rows = self._fetch_event_rows(ts_code=ts_code, target_trade_date=target_trade_date)
        auction_rows = self._fetch_auction_rows(ts_code=ts_code, target_trade_date=target_trade_date)

        event_risk = self._has_event_risk(event_rows)
        if event_risk:
            status = "watch_only"
            reasons.append("检测到公告/异动类事件信息，执行前需人工复核")

        auction_assessment = self._assess_auction(auction_rows)
        if auction_assessment.get("available"):
            reasons.extend(auction_assessment.get("reasons") or [])
            status = self._merge_status(status, auction_assessment.get("status") or "confirmed_buy")
        else:
            reasons.append("未获取到开盘前集合竞价数据，需人工结合盘口确认")
        auction_label = auction_assessment.get("label")

        risk_flags = [str(flag) for flag in item.get("distribution_risk_flags") or [] if flag]
        recent_runup = self._safe_float(item.get("recent_runup_5d"))
        distribution_risk_score = self._safe_float(item.get("distribution_risk_score"))
        if recent_runup is not None and recent_runup >= 12:
            status = self._merge_status(status, "watch_only")
            reasons.append(f"近5日涨幅偏高({recent_runup:.1f}%)")
        if distribution_risk_score is not None and distribution_risk_score >= 35:
            status = self._merge_status(status, "watch_only")
            reasons.append(f"派发/末端风险分较高({distribution_risk_score:.1f})")
        if risk_flags:
            reasons.append("风险标签：" + "、".join(risk_flags[:4]))

        if event_rows:
            event_summary = self._summarize_events(event_rows)
            if event_summary:
                reasons.append(event_summary)

        severe_keywords = ("监管", "关注函", "澄清", "异常波动", "严重异常", "减持", "立案")
        if any(any(keyword in self._row_text(row) for keyword in severe_keywords) for row in event_rows):
            status = "vetoed"
            reasons.append("公告/事件文本命中高风险关键词")

        if not reasons:
            reasons.append("未触发公告/异动与既有风险降级条件")

        return ConfirmationDecision(
            ts_code=ts_code,
            name=name,
            status=status,
            reasons=reasons,
            event_rows=event_rows,
            auction_rows=auction_rows,
            auction_label=auction_label,
            auction_assessment=auction_assessment if auction_assessment.get("available") else {},
            original_rank=self._safe_int(item.get("frontlist_rank") or item.get("recommend_rank")),
            recommendation_score=self._safe_float(item.get("recommendation_score")),
        )

    def _fetch_event_rows(self, *, ts_code: str, target_trade_date: date) -> List[Dict[str, Any]]:
        start_date = target_trade_date.strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")
        rows: List[Dict[str, Any]] = []
        try:
            rows.extend(self.screener.client.fetch_disclosure_announcements(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                limit=20,
            ))
        except Exception:
            logger.exception("Failed to fetch disclosure announcements: ts_code=%s", ts_code)
        try:
            rows.extend(self.screener.client.fetch_special_treatment_rows(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                limit=20,
            ))
        except Exception:
            logger.exception("Failed to fetch special treatment rows: ts_code=%s", ts_code)
        return rows[:20]

    def _fetch_auction_rows(self, *, ts_code: str, target_trade_date: date) -> List[Dict[str, Any]]:
        candidates = [target_trade_date.strftime("%Y%m%d"), datetime.now().strftime("%Y%m%d")]
        try:
            rows = []
            for trade_date in candidates:
                payload = self.tushare_client.fetch_close_auction_batch(ts_codes=[ts_code], trade_date=trade_date)
                row = payload.get(ts_code)
                if row:
                    rows.append(row)
            return rows
        except Exception:
            logger.exception("Failed to fetch auction rows: ts_code=%s", ts_code)
            return []

    async def _send_confirmation_email(self, payload: Dict[str, Any]) -> None:
        if not self.email_service or not self.settings.email_enabled:
            return
        subject = f"开盘前执行确认 - {datetime.now().strftime('%Y-%m-%d')}"
        html_content = self._format_confirmation_html(payload)
        body = "OCTTS 开盘前执行确认已完成，请查看邮件正文。"
        await self.email_service.send_email(subject=subject, html_content=html_content, body=body)

    def _format_confirmation_html(self, payload: Dict[str, Any]) -> str:
        rows = []
        status_labels = {
            "confirmed_buy": "确认通过",
            "watch_only": "降级观察",
            "vetoed": "剔除",
        }
        data_availability = payload.get("data_availability") or {}
        availability_text = self._format_availability(data_availability)
        for item in payload.get("decisions") or []:
            reasons = "<br>".join(html.escape(str(reason)) for reason in item.get("reasons") or [])
            auction = item.get("auction") or {}
            auction_assessment = item.get("auction_assessment") or {}
            auction_text = html.escape(self._format_auction_summary(auction_assessment or auction)) if auction else ""
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('original_rank') or '--'))}</td>"
                f"<td>{html.escape(str(item.get('ts_code') or ''))}</td>"
                f"<td>{html.escape(str(item.get('name') or ''))}</td>"
                f"<td>{html.escape(status_labels.get(str(item.get('status')), str(item.get('status'))))}</td>"
                f"<td>{html.escape(item.get('auction_label') or self._format_auction_label(auction))}</td>"
                f"<td>{html.escape(self._format_score(item.get('recommendation_score')))}</td>"
                f"<td>{reasons}</td>"
                f"<td>{auction_text}</td>"
                "</tr>"
            )
        table_rows = "".join(rows) or "<tr><td colspan='8'>未找到候选 Top3。</td></tr>"
        return f"""
<html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
<h2>OCTTS 开盘前执行确认</h2>
<p>候选来源交易日：{html.escape(str(payload.get('source_trade_date') or ''))}</p>
<p>生成时间：{html.escape(str(payload.get('generated_at') or ''))}</p>
<p>确认结果：通过 {payload.get('confirmed_count', 0)}，观察 {payload.get('watch_only_count', 0)}，剔除 {payload.get('vetoed_count', 0)}</p>
<p>{html.escape(availability_text)}</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; width: 100%;">
<thead><tr><th>原排名</th><th>代码</th><th>名称</th><th>确认状态</th><th>竞价类型</th><th>推荐分</th><th>原因</th><th>竞价摘要</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</body></html>
"""

    def _check_run_window(self) -> Dict[str, Any]:
        now = datetime.now().time()
        start = self._parse_time(self.settings.screening_execution_confirmation_window_start)
        end = self._parse_time(self.settings.screening_execution_confirmation_window_end)
        return {
            "now": datetime.now().strftime("%H:%M:%S"),
            "start": self.settings.screening_execution_confirmation_window_start,
            "end": self.settings.screening_execution_confirmation_window_end,
            "in_window": start <= now <= end,
        }

    @staticmethod
    def _parse_time(value: str) -> dt_time:
        return datetime.strptime(value, "%H:%M").time()

    def _assess_auction(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"available": False}
        row = rows[0]
        open_price = self._safe_float(row.get("open"))
        close_price = self._safe_float(row.get("close"))
        high_price = self._safe_float(row.get("high"))
        low_price = self._safe_float(row.get("low"))
        amount = self._safe_float(row.get("amount"))
        vwap = self._safe_float(row.get("vwap"))
        vol = self._safe_float(row.get("vol"))
        reference_price = self._safe_float(row.get("pre_close")) or self._safe_float(row.get("yclose")) or close_price or open_price
        open_change_pct = self._pct_change(open_price, reference_price)
        close_change_pct = self._pct_change(close_price, reference_price)
        vwap_gap_pct = self._pct_change(vwap, open_price)
        amplitude_pct = self._pct_range(high_price, low_price, reference_price)
        reasons: List[str] = []
        status = "confirmed_buy"
        if open_change_pct is not None and open_change_pct <= -3:
            status = "vetoed"
            reasons.append(f"竞价低开过深({open_change_pct:.2f}%)")
        elif open_change_pct is not None and open_change_pct <= -1.5:
            status = "watch_only"
            reasons.append(f"竞价低开({open_change_pct:.2f}%)")
        elif open_change_pct is not None and open_change_pct >= 3:
            reasons.append(f"竞价高开({open_change_pct:.2f}%)")
        if amount is not None and amount <= 0:
            status = "watch_only" if status != "vetoed" else status
            reasons.append("竞价成交额为空或异常")
        if vol is not None and vol <= 0:
            status = "watch_only" if status != "vetoed" else status
            reasons.append("竞价成交量为空或异常")
        if vwap_gap_pct is not None and abs(vwap_gap_pct) >= 1.2:
            status = self._merge_status(status, "watch_only")
            reasons.append(f"竞价均价与开盘价偏离较大({vwap_gap_pct:+.2f}%)")
        if amplitude_pct is not None and amplitude_pct >= 4:
            status = self._merge_status(status, "watch_only")
            reasons.append(f"竞价振幅偏大({amplitude_pct:.2f}%)")
        if close_change_pct is not None and open_change_pct is not None:
            if open_change_pct > 0 and close_change_pct < 0:
                status = self._merge_status(status, "watch_only")
                reasons.append("竞价表现与收盘竞价方向不一致")
        label = self._classify_auction(
            open_change_pct=open_change_pct,
            vwap_gap_pct=vwap_gap_pct,
            amplitude_pct=amplitude_pct,
            amount=amount,
            vol=vol,
        )
        return {
            "available": True,
            "status": status,
            "label": label,
            "open_price": open_price,
            "close_price": close_price,
            "high_price": high_price,
            "low_price": low_price,
            "amount": amount,
            "vol": vol,
            "vwap": vwap,
            "open_change_pct": open_change_pct,
            "close_change_pct": close_change_pct,
            "vwap_gap_pct": vwap_gap_pct,
            "amplitude_pct": amplitude_pct,
            "reasons": reasons,
            "raw": row,
        }

    @staticmethod
    def _classify_auction(
        *,
        open_change_pct: Optional[float],
        vwap_gap_pct: Optional[float],
        amplitude_pct: Optional[float],
        amount: Optional[float],
        vol: Optional[float],
    ) -> str:
        weak_liquidity = (amount is not None and amount <= 0) or (vol is not None and vol <= 0)
        unstable = (amplitude_pct is not None and amplitude_pct >= 4) or (vwap_gap_pct is not None and abs(vwap_gap_pct) >= 1.2)
        if open_change_pct is None:
            return "平开分歧" if ExecutionConfirmationService._pct_range(
                ExecutionConfirmationService._safe_float(auction.get("high")),
                ExecutionConfirmationService._safe_float(auction.get("low")),
                reference_price,
            ) not in (None, 0) else "竞价未分类"
        if open_change_pct >= 1.5:
            return "高开弱承接" if weak_liquidity or unstable else "高开强承接"
        if open_change_pct <= -1.5:
            return "低开承压"
        return "平开分歧" if unstable else "平开中性"

    def _summarize_events(self, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        row = rows[0]
        headline = self._first_non_empty(
            row.get("title"),
            row.get("name"),
            row.get("ann_title"),
            row.get("content"),
            row.get("summary"),
            row.get("change_reason"),
        )
        if not headline:
            headline = self._row_text(row)[:60]
        if len(headline) > 70:
            headline = headline[:67] + "..."
        return f"事件摘要：{headline}"

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    @staticmethod
    def _serialize_decision(item: ConfirmationDecision) -> Dict[str, Any]:
        return {
            "ts_code": item.ts_code,
            "name": item.name,
            "status": item.status,
            "reasons": list(item.reasons),
            "event_rows": item.event_rows,
            "auction_rows": item.auction_rows,
            "auction": item.auction_rows[0] if item.auction_rows else {},
            "auction_label": item.auction_label,
            "auction_assessment": dict(item.auction_assessment or {}),
            "original_rank": item.original_rank,
            "recommendation_score": item.recommendation_score,
        }

    @staticmethod
    def _has_event_risk(rows: List[Dict[str, Any]]) -> bool:
        if not rows:
            return False
        keywords = ("异动", "异常", "波动", "澄清", "监管", "关注", "减持", "解禁", "风险")
        return any(any(keyword in ExecutionConfirmationService._row_text(row) for keyword in keywords) for row in rows)

    @staticmethod
    def _row_text(row: Dict[str, Any]) -> str:
        return " ".join(str(value or "") for value in row.values())

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_data_availability_summary(candidates: List[Dict[str, Any]], decisions: List[ConfirmationDecision]) -> Dict[str, Any]:
        auction_available = sum(1 for item in decisions if item.auction_rows)
        return {
            "candidate_count": len(candidates),
            "auction_available_count": auction_available,
            "auction_required": bool(candidates),
        }

    @staticmethod
    def _merge_status(current: str, incoming: str) -> str:
        order = {"confirmed_buy": 0, "watch_only": 1, "vetoed": 2}
        return incoming if order.get(incoming, 0) > order.get(current, 0) else current

    @staticmethod
    def _pct_change(current: Optional[float], base: Optional[float]) -> Optional[float]:
        if current is None or base in (None, 0):
            return None
        return (current - base) / base * 100.0

    @staticmethod
    def _pct_range(high: Optional[float], low: Optional[float], base: Optional[float]) -> Optional[float]:
        if high is None or low is None or base in (None, 0):
            return None
        return (high - low) / base * 100.0

    @staticmethod
    def _format_availability(data: Dict[str, Any]) -> str:
        if not data:
            return "数据源状态：未评估"
        return (
            f"数据源状态：竞价可用 {data.get('auction_available_count', 0)}/{data.get('candidate_count', 0)}。"
        )

    @staticmethod
    def _format_auction_label(auction: Dict[str, Any]) -> str:
        if not auction:
            return "--"
        reference_price = (
            ExecutionConfirmationService._safe_float(auction.get("pre_close"))
            or ExecutionConfirmationService._safe_float(auction.get("yclose"))
            or ExecutionConfirmationService._safe_float(auction.get("close"))
            or ExecutionConfirmationService._safe_float(auction.get("open"))
        )
        open_change_pct = ExecutionConfirmationService._pct_change(
            ExecutionConfirmationService._safe_float(auction.get("open")),
            reference_price,
        )
        if open_change_pct is None:
            return "竞价未分类"
        vwap_gap_pct = ExecutionConfirmationService._pct_change(
            ExecutionConfirmationService._safe_float(auction.get("vwap")),
            ExecutionConfirmationService._safe_float(auction.get("open")),
        )
        amplitude_pct = ExecutionConfirmationService._pct_range(
            ExecutionConfirmationService._safe_float(auction.get("high")),
            ExecutionConfirmationService._safe_float(auction.get("low")),
            reference_price,
        )
        return ExecutionConfirmationService._classify_auction(
            open_change_pct=open_change_pct,
            vwap_gap_pct=vwap_gap_pct,
            amplitude_pct=amplitude_pct,
            amount=ExecutionConfirmationService._safe_float(auction.get("amount")),
            vol=ExecutionConfirmationService._safe_float(auction.get("vol")),
        )

    @staticmethod
    def _format_auction_summary(auction: Dict[str, Any]) -> str:
        if not auction:
            return ""
        parts = []
        for key, label in (("open_change_pct", "竞价开盘"), ("vwap_gap_pct", "均价偏离"), ("amplitude_pct", "振幅")):
            value = auction.get(key)
            if value is not None:
                parts.append(f"{label}:{float(value):+.2f}%")
        if auction.get("amount") is not None:
            parts.append(f"金额:{float(auction.get('amount')):.0f}")
        if auction.get("vol") is not None:
            parts.append(f"量:{float(auction.get('vol')):.0f}")
        return "；".join(parts)

    @staticmethod
    def _format_score(value: Any) -> str:
        try:
            return f"{float(value):.1f}"
        except (TypeError, ValueError):
            return "--"

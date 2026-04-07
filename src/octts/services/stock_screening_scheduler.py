"""Stock screening scheduler service."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from octts.clients.email_client import EmailClient
from octts.config import Settings
from octts.schemas.screener import ScreenPreset
from octts.services.history_store import FileHistoryStore
from octts.services.position_store import create_position_store
from octts.services.report_exporter import ReportExporter
from octts.services.stock_screener import StockScreener
from octts.services.screening_store import ScreeningStore
from octts.services.screening_report import ScreeningReportService
from octts.clients.wecom_client import WeComClient
from octts.services.report_email_service import ReportEmailService

logger = logging.getLogger(__name__)


class StockScreeningScheduler:
    """定时执行选股策略并推送结果"""

    def __init__(
        self,
        settings: Settings,
        screener: Optional[StockScreener] = None,
        store: Optional[ScreeningStore] = None,
        report_service: Optional[ScreeningReportService] = None,
        wecom_client: Optional[WeComClient] = None,
        email_service: Optional[ReportEmailService] = None,
    ):
        self.settings = settings
        self.screener = screener or StockScreener(settings)
        self.store = store or ScreeningStore(settings)
        self.report_service = report_service or ScreeningReportService(settings)
        self.wecom_client = wecom_client
        self.email_service = email_service

    async def run_daily_screening(self) -> Dict[str, Any]:
        """
        执行每日选股任务

        Returns:
            执行结果统计
        """
        logger.info("Starting daily stock screening")
        start_time = datetime.now()

        # 获取要运行的策略
        strategies = self._get_active_strategies()
        results = {}
        total_stocks = 0
        trade_date = self.screener._get_latest_trade_date()
        logger.info("Daily stock screening resolve trade_date: %s", trade_date)
        market_snapshot = self.screener.client.get_or_build_screening_snapshot(trade_date)
        logger.info(
            "Daily stock screening snapshot ready: stocks=%s, basic=%s, cached_daily=%s",
            len(market_snapshot.get("stocks", [])),
            len(market_snapshot.get("daily_basic", {})),
            sum(1 for items in market_snapshot.get("daily", {}).values() if items),
        )

        logger.info(
            "Daily stock screening: %s strategies queued, shared trade_date=%s",
            len(strategies),
            trade_date,
        )

        for strategy in strategies:
            try:
                # 执行筛选
                result = self.screener.screen(
                    strategy.criteria,
                    trade_date=trade_date,
                    market_snapshot=market_snapshot,
                )
                results[strategy.id] = result
                total_stocks += len(result.stocks)

                # 保存结果
                await self.store.save_screening_result(strategy.id, result)

                logger.info(
                    f"Strategy '{strategy.name}' found {len(result.stocks)} stocks"
                )

            except Exception as e:
                logger.error(f"Strategy '{strategy.name}' failed: {e}")
                results[strategy.id] = None

        # 生成综合报告
        report = await self.report_service.generate_daily_report(results)
        self._save_intelligent_snapshot_from_daily_results(
            screening_results=results,
            report=report,
            trade_date=trade_date,
        )

        # 推送通知
        if self.settings.screening_notify:
            await self._send_notifications(report)

        duration = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "strategies_run": len(strategies),
            "total_stocks": total_stocks,
            "duration_seconds": duration,
            "report_id": report.get("report_id"),
        }

    def _get_active_strategies(self) -> List[ScreenPreset]:
        """获取配置的活跃策略"""
        # 从配置中读取要运行的策略
        strategy_ids = self.settings.screening_strategies

        if not strategy_ids:
            # 默认运行所有预设策略
            return StockScreener.get_presets()

        # 获取指定的策略
        all_presets = StockScreener.get_presets()
        active_strategies = [
            preset for preset in all_presets
            if preset.id in strategy_ids
        ]

        return active_strategies

    def _save_intelligent_snapshot_from_daily_results(
        self,
        *,
        screening_results: Dict[str, Any],
        report: Dict[str, Any],
        trade_date: str,
    ) -> None:
        stock_payloads: list[dict[str, Any]] = []
        total_stocks = 0

        for strategy_result in report.get("strategy_results", []):
            for stock in strategy_result.get("top_stocks", []):
                stock_payloads.append(
                    {
                        "ts_code": stock.get("ts_code"),
                        "name": stock.get("name") or "",
                        "score": stock.get("pct_change") or 0.0,
                        "overall_score": stock.get("pct_change") or 0.0,
                        "overall_confidence": None,
                        "recommendation": "规则选股命中",
                        "summary": f"命中策略：{strategy_result.get('strategy_name', '')}",
                        "technical_signal": strategy_result.get("strategy_name") or "规则命中",
                        "technical_score": None,
                        "strategy_count": 1,
                        "news_mentioned": False,
                    }
                )
                total_stocks += 1

        ai_analyses = {
            item["ts_code"]: item
            for item in stock_payloads
            if item.get("ts_code")
        }
        frontlist = [
            {
                "ts_code": item["ts_code"],
                "name": item.get("name") or "",
                "priority_score": item.get("score", 0.0),
                "recommendation_score": item.get("score", 0.0),
                "in_frontlist": True,
                "tracking_status": "active",
                "llm_focus_level": "medium",
                "hit_streak_days": 0,
                "miss_streak_days": 0,
                "technical_signal": item.get("technical_signal") or "规则命中",
                "recommendation_text": item.get("summary") or "规则选股命中",
                "ai_confidence": item.get("overall_confidence"),
            }
            for item in list(ai_analyses.values())[:20]
        ]

        snapshot = {
            "generated_at": report.get("report_time") or datetime.now().isoformat(),
            "snapshot_type": "daily_screening_compat",
            "screening_results": {
                "strategy_count": len([result for result in screening_results.values() if result is not None]),
                "total_stocks": total_stocks,
                "final_recommendations": len(frontlist),
                "frontlist_count": len(frontlist),
                "shadow_count": 0,
                "candidate_count": len(ai_analyses),
                "trade_date": trade_date,
                "source": "daily_screening_compat",
            },
            "recommendation_pool": {
                "frontlist": frontlist,
                "shadow": [],
                "shadow_symbols": [],
            },
            "ai_analyses": ai_analyses,
            "news_clusters": [],
            "intelligent_report": {
                "report_id": report.get("report_id", ""),
                "title": "定时选股结果（兼容快照）",
                "summary": f"本次自动任务产出的是普通定时选股结果，保存在兼容快照中。共命中 {total_stocks} 只股票。",
                "sections": [],
                "recommendations": [],
                "key_points": [],
            },
        }

        snapshot_dir = Path(self.settings.history_dir_path) / "intelligent_screening"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("daily_screening_compat_latest.json", "latest.json"):
            snapshot_path = snapshot_dir / file_name
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

    async def _send_notifications(self, report: Dict[str, Any]) -> None:
        """发送选股结果通知"""
        # 企业微信通知
        if self.wecom_client and self.settings.wecom_webhook_url:
            try:
                message = self._format_wecom_message(report)
                self.wecom_client.send_markdown(message)
                logger.info("Sent screening report to WeChat")
            except Exception as e:
                logger.error(f"Failed to send WeChat notification: {e}")

        # 邮件通知
        if self.email_service and self.settings.email_enabled:
            try:
                await self.email_service.send_screening_report(report)
                logger.info("Sent screening report via email")
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")

    def _format_wecom_message(self, report: Dict[str, Any]) -> str:
        """格式化企业微信消息"""
        lines = [
            f"## 📊 每日选股报告",
            f"**时间**: {report['report_time']}",
            f"**策略数**: {report['strategy_count']}",
            f"**选出股票**: {report['total_stocks']}只",
            "",
        ]

        # 添加每个策略的结果
        for strategy_result in report.get("strategy_results", []):
            lines.extend([
                f"### {strategy_result['strategy_name']}",
                f"选出 **{strategy_result['stock_count']}** 只股票",
            ])

            # 显示前5只股票
            for stock in strategy_result['top_stocks'][:5]:
                lines.append(
                    f"- {stock['name']}({stock['ts_code']}) "
                    f"涨幅:{stock['pct_change']:.2f}% "
                    f"量比:{stock['volume_ratio']:.2f}"
                )

            if strategy_result['stock_count'] > 5:
                lines.append(f"- ... 还有 {strategy_result['stock_count'] - 5} 只")

            lines.append("")

        return "\n".join(lines)


def create_screening_scheduler(settings: Settings) -> StockScreeningScheduler:
    """创建选股调度器实例"""
    wecom_client = None
    if settings.wecom_webhook_url:
        wecom_client = WeComClient(settings)

    email_service = None
    if settings.email_enabled:
        history_store = FileHistoryStore(settings.history_dir_path)
        email_service = ReportEmailService(
            settings=settings,
            history_store=history_store,
            report_exporter=ReportExporter(
                settings=settings,
                history_store=history_store,
                position_store=create_position_store(settings),
            ),
            email_client=EmailClient(settings),
        )

    return StockScreeningScheduler(
        settings=settings,
        wecom_client=wecom_client,
        email_service=email_service,
    )
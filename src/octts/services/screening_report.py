"""Stock screening report generation service."""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from octts.config import Settings
from octts.schemas.screener import ScreenResult, ScreenPreset
from octts.services.screening_store import ScreeningStore

logger = logging.getLogger(__name__)


class ScreeningReportService:
    """选股报告生成服务"""

    def __init__(
        self,
        settings: Settings,
        store: Optional[ScreeningStore] = None
    ):
        self.settings = settings
        self.store = store or ScreeningStore(settings)

    async def generate_daily_report(
        self,
        screening_results: Dict[str, Optional[ScreenResult]]
    ) -> Dict[str, Any]:
        """
        生成每日选股报告

        Args:
            screening_results: {strategy_id: result}

        Returns:
            报告数据
        """
        report_time = datetime.now()
        strategy_results = []
        total_stocks = 0
        unique_stocks = set()

        # 处理每个策略的结果
        for strategy_id, result in screening_results.items():
            if result is None:
                continue

            # 获取策略信息
            preset = self._get_preset_by_id(strategy_id)
            if not preset:
                continue

            # 统计
            stock_count = len(result.stocks)
            total_stocks += stock_count

            # 收集唯一股票
            for stock in result.stocks:
                unique_stocks.add(stock.ts_code)

            # 准备策略结果
            strategy_result = {
                "strategy_id": strategy_id,
                "strategy_name": preset.name,
                "strategy_description": preset.description,
                "stock_count": stock_count,
                "execution_time": result.execution_time,
                "top_stocks": [
                    {
                        "ts_code": stock.ts_code,
                        "name": stock.name,
                        "close": stock.close,
                        "pct_change": stock.pct_change,
                        "volume_ratio": stock.volume_ratio,
                        "match_reasons": stock.match_reasons,
                    }
                    for stock in result.stocks[:10]  # 前10只
                ],
            }

            strategy_results.append(strategy_result)

        # 生成报告
        report = {
            "report_id": f"screening_{report_time.strftime('%Y%m%d_%H%M%S')}",
            "report_time": report_time.isoformat(),
            "report_date": report_time.date().isoformat(),
            "strategy_count": len(strategy_results),
            "total_stocks": total_stocks,
            "unique_stocks": len(unique_stocks),
            "strategy_results": strategy_results,
            "market_overview": await self._get_market_overview(),
        }

        # 保存报告
        await self._save_report(report)

        return report

    def _get_preset_by_id(self, strategy_id: str) -> Optional[ScreenPreset]:
        """根据ID获取策略预设"""
        from octts.services.stock_screener import StockScreener

        presets = StockScreener.get_presets()
        for preset in presets:
            if preset.id == strategy_id:
                return preset
        return None

    async def _get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览（可选）"""
        # TODO: 可以添加大盘指数、涨跌统计等信息
        return {
            "index_status": "待实现",
            "market_sentiment": "待实现",
        }

    async def _save_report(self, report: Dict[str, Any]) -> None:
        """保存报告到文件"""
        import json
        from pathlib import Path

        report_path = Path(self.settings.history_dir_path) / "reports" / "screening"
        report_path.mkdir(parents=True, exist_ok=True)

        file_name = f"{report['report_id']}.json"
        file_path = report_path / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved screening report to {file_path}")

    def format_html_report(self, report: Dict[str, Any]) -> str:
        """
        格式化HTML报告

        Args:
            report: 报告数据

        Returns:
            HTML字符串
        """
        html_parts = [
            f"<h2>每日选股报告</h2>",
            f"<p><strong>时间:</strong> {report['report_time']}</p>",
            f"<p><strong>运行策略:</strong> {report['strategy_count']}个</p>",
            f"<p><strong>选出股票:</strong> {report['total_stocks']}只（去重后{report['unique_stocks']}只）</p>",
            "<hr>",
        ]

        # 添加每个策略的结果
        for strategy in report.get("strategy_results", []):
            html_parts.extend([
                f"<h3>{strategy['strategy_name']}</h3>",
                f"<p>{strategy['strategy_description']}</p>",
                f"<p><strong>选出股票:</strong> {strategy['stock_count']}只</p>",
            ])

            if strategy['top_stocks']:
                html_parts.append("<table border='1' cellpadding='5'>")
                html_parts.append(
                    "<tr><th>代码</th><th>名称</th><th>现价</th>"
                    "<th>涨幅%</th><th>量比</th><th>匹配原因</th></tr>"
                )

                for stock in strategy['top_stocks']:
                    reasons = "<br>".join(stock.get('match_reasons', []))
                    html_parts.append(
                        f"<tr>"
                        f"<td>{stock['ts_code']}</td>"
                        f"<td>{stock['name']}</td>"
                        f"<td>{stock['close']:.2f}</td>"
                        f"<td>{stock['pct_change']:.2f}%</td>"
                        f"<td>{stock['volume_ratio']:.2f}</td>"
                        f"<td>{reasons}</td>"
                        f"</tr>"
                    )

                html_parts.append("</table>")

                if strategy['stock_count'] > len(strategy['top_stocks']):
                    html_parts.append(
                        f"<p><em>... 还有 {strategy['stock_count'] - len(strategy['top_stocks'])} 只股票</em></p>"
                    )

            html_parts.append("<hr>")

        return "\n".join(html_parts)

    def format_markdown_report(self, report: Dict[str, Any]) -> str:
        """
        格式化Markdown报告

        Args:
            report: 报告数据

        Returns:
            Markdown字符串
        """
        lines = [
            "# 每日选股报告",
            f"**时间:** {report['report_time']}",
            f"**运行策略:** {report['strategy_count']}个",
            f"**选出股票:** {report['total_stocks']}只（去重后{report['unique_stocks']}只）",
            "",
            "---",
            "",
        ]

        # 添加每个策略的结果
        for strategy in report.get("strategy_results", []):
            lines.extend([
                f"## {strategy['strategy_name']}",
                f"_{strategy['strategy_description']}_",
                f"**选出股票:** {strategy['stock_count']}只",
                "",
            ])

            if strategy['top_stocks']:
                lines.extend([
                    "| 代码 | 名称 | 现价 | 涨幅% | 量比 | 匹配原因 |",
                    "|------|------|------|-------|------|----------|",
                ])

                for stock in strategy['top_stocks']:
                    reasons = ", ".join(stock.get('match_reasons', []))
                    lines.append(
                        f"| {stock['ts_code']} | {stock['name']} | "
                        f"{stock['close']:.2f} | {stock['pct_change']:.2f}% | "
                        f"{stock['volume_ratio']:.2f} | {reasons} |"
                    )

                if strategy['stock_count'] > len(strategy['top_stocks']):
                    lines.append(
                        f"\n_... 还有 {strategy['stock_count'] - len(strategy['top_stocks'])} 只股票_"
                    )

            lines.extend(["", "---", ""])

        return "\n".join(lines)
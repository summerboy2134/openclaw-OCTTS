"""Lightweight backtesting for stock screening strategies."""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from octts.config import Settings
from octts.clients.tushare_client import TushareClient
from octts.services.stock_screener import StockScreener
from octts.schemas.screener import ScreenPreset

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_return: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    detail_records: List[Dict]  # 详细交易记录


class LightweightBacktester:
    """
    轻量级回测器

    特点：
    1. 简单易用，无需复杂配置
    2. 快速验证策略有效性
    3. 提供关键指标
    """

    def __init__(
        self,
        settings: Settings,
        tushare_client: Optional[TushareClient] = None,
        screener: Optional[StockScreener] = None
    ):
        self.settings = settings
        self.tushare_client = tushare_client or TushareClient(settings)
        self.screener = screener or StockScreener(settings)

    def backtest_strategy(
        self,
        strategy: ScreenPreset,
        start_date: str,
        end_date: str,
        holding_days: int = 5,  # 持有天数
        top_n: int = 10,  # 每次选几只
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0005,
        stock_pool: Optional[List[str]] = None,
    ) -> BacktestResult:
        """
        回测单个策略

        Args:
            strategy: 选股策略
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            holding_days: 持有天数（T+1限制）
            top_n: 每次选股数量

        Returns:
            回测结果
        """
        logger.info(f"Backtesting strategy: {strategy.name}")

        # 获取交易日列表
        trade_dates = self.tushare_client.fetch_trading_dates(
            start_date=start_date,
            end_date=end_date
        )

        # 每隔holding_days执行一次策略
        test_dates = trade_dates[::holding_days]

        trades = []
        normalized_stock_pool = {
            code.strip().upper() for code in (stock_pool or []) if isinstance(code, str) and code.strip()
        }

        for test_date in test_dates:
            selected_stocks = self._run_strategy_on_date(
                strategy,
                test_date,
                top_n
            )
            if normalized_stock_pool:
                selected_stocks = [
                    stock for stock in selected_stocks
                    if stock.strip().upper() in normalized_stock_pool
                ]

            if not selected_stocks:
                continue

            entry_date = self._get_next_trade_date(test_date, trade_dates)
            exit_date = self._get_exit_date(test_date, holding_days, trade_dates)
            if not entry_date or not exit_date:
                continue

            for stock in selected_stocks:
                entry_trade = self._get_trade_price(stock, entry_date, prefer_open=True)
                exit_trade = self._get_trade_price(stock, exit_date, prefer_open=False)

                if not entry_trade or not exit_trade:
                    continue

                entry_price = entry_trade["price"] * (1 + slippage_rate)
                exit_price = exit_trade["price"] * (1 - slippage_rate)
                gross_return = (exit_price - entry_price) / entry_price
                net_return = gross_return - (2 * commission_rate)

                trades.append({
                    'ts_code': stock,
                    'signal_date': test_date,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(exit_price, 4),
                    'entry_price_field': entry_trade["field"],
                    'exit_price_field': exit_trade["field"],
                    'gross_return_pct': round(gross_return * 100, 2),
                    'return_pct': round(net_return * 100, 2),
                    'holding_days': holding_days,
                    'commission_rate': commission_rate,
                    'slippage_rate': slippage_rate,
                })

        # 计算统计指标
        return self._calculate_metrics(
            strategy.name,
            start_date,
            end_date,
            trades
        )

    def _run_strategy_on_date(
        self,
        strategy: ScreenPreset,
        trade_date: str,
        top_n: int
    ) -> List[str]:
        """在指定日期运行策略"""
        try:
            # 模拟历史数据环境
            result = self.screener.screen(strategy.criteria, trade_date=trade_date)

            # 返回前N只股票
            return [s.ts_code for s in result.stocks[:top_n]]

        except Exception as e:
            logger.error(f"Failed to run strategy on {trade_date}: {e}")
            return []

    def _get_trade_price(
        self,
        ts_code: str,
        trade_date: str,
        prefer_open: bool
    ) -> Optional[Dict[str, Any]]:
        """获取股票在指定日期的成交价格，优先开盘价，缺失时退回收盘价。"""
        try:
            data = self.tushare_client.fetch_daily_bars(
                ts_code=ts_code,
                start_date=trade_date,
                end_date=trade_date
            )

            if data and len(data) > 0:
                bar = data[0]
                if isinstance(bar, dict):
                    open_price = bar.get("open")
                    close_price = bar.get("close")
                else:
                    open_price = getattr(bar, "open", None)
                    close_price = getattr(bar, "close", None)
                if prefer_open and open_price not in (None, 0):
                    return {"price": float(open_price), "field": "open"}
                if close_price not in (None, 0):
                    return {"price": float(close_price), "field": "close"}
                if open_price not in (None, 0):
                    return {"price": float(open_price), "field": "open"}

        except Exception as e:
            logger.error(f"Failed to get price for {ts_code} on {trade_date}: {e}")

        return None

    def _get_next_trade_date(self, trade_date: str, trade_dates: List[str]) -> Optional[str]:
        try:
            trade_idx = trade_dates.index(trade_date)
            next_idx = trade_idx + 1
            if next_idx < len(trade_dates):
                return trade_dates[next_idx]
        except ValueError:
            pass
        return None

    def _get_exit_date(
        self,
        entry_date: str,
        holding_days: int,
        trade_dates: List[str]
    ) -> Optional[str]:
        """获取退出日期"""
        try:
            entry_idx = trade_dates.index(entry_date)
            exit_idx = entry_idx + holding_days

            if exit_idx < len(trade_dates):
                return trade_dates[exit_idx]

        except ValueError:
            pass

        return None

    def _calculate_metrics(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
        trades: List[Dict[str, Any]]
    ) -> BacktestResult:
        """计算回测指标"""
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_return=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                detail_records=[]
            )

        # 转换为DataFrame便于计算
        df = pd.DataFrame(trades)

        # 基础统计
        total_trades = len(trades)
        winning_trades = len(df[df['return_pct'] > 0])
        losing_trades = len(df[df['return_pct'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_return = df['return_pct'].mean()

        period_returns = df.groupby('entry_date')['return_pct'].mean().sort_index()
        equity_curve = (1 + period_returns / 100).cumprod()
        total_return = (equity_curve.iloc[-1] - 1) * 100

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # 夏普比率（简化版）
        holding_period = int(df['holding_days'].iloc[0]) if 'holding_days' in df.columns and not df.empty else 5
        if period_returns.std() > 0:
            sharpe_ratio = period_returns.mean() / period_returns.std() * (252 / max(holding_period, 1)) ** 0.5
        else:
            sharpe_ratio = 0

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_return=round(avg_return, 2),
            total_return=round(total_return, 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            detail_records=trades
        )

    def compare_strategies(
        self,
        strategies: List[ScreenPreset],
        start_date: str,
        end_date: str,
        holding_days: int = 5,
        top_n: int = 10,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0005,
        stock_pool: Optional[List[str]] = None,
    ) -> Dict[str, BacktestResult]:
        """
        对比多个策略

        Returns:
            {策略名: 回测结果}
        """
        results = {}

        for strategy in strategies:
            result = self.backtest_strategy(
                strategy,
                start_date,
                end_date,
                holding_days,
                top_n=top_n,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                stock_pool=stock_pool,
            )
            results[strategy.name] = result

        return results

    def generate_report(
        self,
        results: Dict[str, BacktestResult]
    ) -> str:
        """生成回测报告"""
        lines = [
            "# 策略回测报告",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 策略对比",
            "",
            "| 策略 | 胜率 | 平均收益 | 总收益 | 最大回撤 | 夏普比率 | 交易次数 |",
            "|------|------|----------|--------|----------|----------|----------|"
        ]

        # 按总收益排序
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].total_return,
            reverse=True
        )

        for name, result in sorted_results:
            lines.append(
                f"| {name} | {result.win_rate:.1%} | {result.avg_return:.2f}% | "
                f"{result.total_return:.2f}% | {result.max_drawdown:.2f}% | "
                f"{result.sharpe_ratio:.2f} | {result.total_trades} |"
            )

        # 添加详细分析
        lines.extend([
            "",
            "## 详细分析",
            ""
        ])

        for name, result in sorted_results[:3]:  # 只显示前3名
            lines.extend([
                f"### {name}",
                f"- **总交易次数**：{result.total_trades}",
                f"- **盈利交易**：{result.winning_trades} ({result.win_rate:.1%})",
                f"- **亏损交易**：{result.losing_trades}",
                f"- **平均每笔收益**：{result.avg_return:.2f}%",
                f"- **最大回撤**：{result.max_drawdown:.2f}%",
                ""
            ])

        return "\n".join(lines)
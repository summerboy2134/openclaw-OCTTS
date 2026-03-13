from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from octts.schemas.backtest import (
    BacktestDailyPosition,
    BacktestMetrics,
    BacktestRequest,
    BacktestResult,
    BacktestTrade,
    DailyBar,
)
from octts.schemas.report import MemorySummary, StructuredAnalysis
from octts.services.analysis_pipeline import AnalysisPipeline


class BacktestMarketDataClient(Protocol):
    def fetch_trading_dates(self, *, start_date: str, end_date: str) -> list[str]:
        ...

    def fetch_daily_bars(self, *, ts_code: str, start_date: str, end_date: str) -> list[DailyBar]:
        ...

    def fetch_historical_snapshot(self, *, ts_code: str, phase: str, trade_date: str):
        ...


@dataclass
class PendingOrder:
    ts_code: str
    signal_date: str
    entry_date: str
    report: StructuredAnalysis


@dataclass
class OpenPosition:
    ts_code: str
    signal_date: str
    entry_date: str
    entry_price: float
    quantity: float
    gross_value: float
    commission_paid: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    max_holding_days: int


class BacktestEngine:
    def __init__(
        self,
        *,
        pipeline: AnalysisPipeline,
        market_data_client: BacktestMarketDataClient,
    ) -> None:
        self._pipeline = pipeline
        self._market_data_client = market_data_client

    def run(self, request: BacktestRequest) -> BacktestResult:
        stock_pool = request.stock_pool or self._pipeline._settings.stock_pool
        if not stock_pool:
            raise ValueError("No stock pool provided. Set OCTTS_STOCK_POOL or pass stock_pool in request.")
        if request.phase != "review":
            raise ValueError("Backtest currently only supports the review phase.")

        trading_dates = self._market_data_client.fetch_trading_dates(
            start_date=request.start_date,
            end_date=request.end_date,
        )
        if len(trading_dates) < 2:
            raise ValueError("Backtest requires at least two open trading dates in the selected range.")

        bars_by_symbol = {
            ts_code: self._index_bars(
                self._market_data_client.fetch_daily_bars(
                    ts_code=ts_code,
                    start_date=request.start_date,
                    end_date=request.end_date,
                )
            )
            for ts_code in stock_pool
        }

        memory_by_symbol: dict[str, MemorySummary] = {}
        pending_orders: dict[str, PendingOrder] = {}
        open_positions: dict[str, OpenPosition] = {}
        trades: list[BacktestTrade] = []
        daily_positions: list[BacktestDailyPosition] = []
        cash = request.initial_cash

        for date_index, trade_date in enumerate(trading_dates):
            cash = self._process_pending_entries(
                trade_date=trade_date,
                pending_orders=pending_orders,
                open_positions=open_positions,
                bars_by_symbol=bars_by_symbol,
                request=request,
                cash=cash,
            )
            cash = self._process_open_positions(
                trade_date=trade_date,
                open_positions=open_positions,
                bars_by_symbol=bars_by_symbol,
                request=request,
                trades=trades,
                cash=cash,
                date_index=date_index,
                trading_dates=trading_dates,
            )

            for ts_code in stock_pool:
                bar = bars_by_symbol.get(ts_code, {}).get(trade_date)
                if bar is None:
                    continue
                snapshot = self._market_data_client.fetch_historical_snapshot(
                    ts_code=ts_code,
                    phase=request.phase,
                    trade_date=trade_date,
                )
                _, _, report = self._pipeline.generate_report_from_snapshot(
                    phase=request.phase,
                    snapshot=snapshot,
                    previous_memory=memory_by_symbol.get(ts_code),
                )
                memory_by_symbol[ts_code] = report.memory

                if report.decision.signal != "buy":
                    continue
                if ts_code in open_positions or ts_code in pending_orders:
                    continue
                if date_index + 1 >= len(trading_dates):
                    continue

                pending_orders[ts_code] = PendingOrder(
                    ts_code=ts_code,
                    signal_date=trade_date,
                    entry_date=trading_dates[date_index + 1],
                    report=report,
                )

            market_value = self._mark_to_market(open_positions=open_positions, bars_by_symbol=bars_by_symbol, trade_date=trade_date)
            daily_positions.append(
                BacktestDailyPosition(
                    trade_date=trade_date,
                    cash=round(cash, 4),
                    market_value=round(market_value, 4),
                    equity=round(cash + market_value, 4),
                    open_positions=len(open_positions),
                )
            )

        ending_cash = cash + self._mark_to_market(
            open_positions=open_positions,
            bars_by_symbol=bars_by_symbol,
            trade_date=trading_dates[-1],
        )
        metrics = self._build_metrics(
            initial_cash=request.initial_cash,
            ending_cash=ending_cash,
            trades=trades,
            daily_positions=daily_positions,
            trading_days=len(trading_dates),
        )
        return BacktestResult(
            phase=request.phase,
            stock_pool=stock_pool,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_cash=request.initial_cash,
            ending_cash=round(ending_cash, 4),
            trades=trades,
            daily_positions=daily_positions,
            metrics=metrics,
        )

    def _process_pending_entries(
        self,
        *,
        trade_date: str,
        pending_orders: dict[str, PendingOrder],
        open_positions: dict[str, OpenPosition],
        bars_by_symbol: dict[str, dict[str, DailyBar]],
        request: BacktestRequest,
        cash: float,
    ) -> float:
        updated_cash = cash
        for ts_code, order in list(pending_orders.items()):
            if order.entry_date != trade_date:
                continue
            bar = bars_by_symbol.get(ts_code, {}).get(trade_date)
            if bar is None or bar.open is None:
                pending_orders.pop(ts_code, None)
                continue
            if not self._entry_price_allowed(order.report, bar.open):
                pending_orders.pop(ts_code, None)
                continue

            slippage_multiplier = 1 + request.slippage_rate
            entry_price = bar.open * slippage_multiplier
            deployable_cash = updated_cash * request.position_size_pct
            commission = deployable_cash * request.commission_rate
            if deployable_cash <= commission or entry_price <= 0:
                pending_orders.pop(ts_code, None)
                continue

            quantity = max((deployable_cash - commission) / entry_price, 0.0)
            if quantity <= 0:
                pending_orders.pop(ts_code, None)
                continue

            updated_cash -= deployable_cash
            open_positions[ts_code] = OpenPosition(
                ts_code=ts_code,
                signal_date=order.signal_date,
                entry_date=trade_date,
                entry_price=entry_price,
                quantity=quantity,
                gross_value=deployable_cash,
                commission_paid=commission,
                stop_loss=order.report.decision.stop_loss,
                take_profit=order.report.decision.take_profit[0] if order.report.decision.take_profit else None,
                max_holding_days=_max_holding_days(order.report.decision.holding_horizon),
            )
            pending_orders.pop(ts_code, None)
        return updated_cash

    def _process_open_positions(
        self,
        *,
        trade_date: str,
        open_positions: dict[str, OpenPosition],
        bars_by_symbol: dict[str, dict[str, DailyBar]],
        request: BacktestRequest,
        trades: list[BacktestTrade],
        cash: float,
        date_index: int,
        trading_dates: list[str],
    ) -> float:
        updated_cash = cash
        for ts_code, position in list(open_positions.items()):
            bar = bars_by_symbol.get(ts_code, {}).get(trade_date)
            if bar is None or bar.close is None:
                continue

            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None
            low = bar.low if bar.low is not None else bar.close
            high = bar.high if bar.high is not None else bar.close

            if position.stop_loss is not None and low is not None and low <= position.stop_loss:
                exit_price = position.stop_loss * (1 - request.slippage_rate)
                exit_reason = "stop_loss"
            elif position.take_profit is not None and high is not None and high >= position.take_profit:
                exit_price = position.take_profit * (1 - request.slippage_rate)
                exit_reason = "take_profit"
            elif self._holding_days(position.entry_date, trade_date, trading_dates) >= position.max_holding_days:
                exit_price = bar.close * (1 - request.slippage_rate)
                exit_reason = "horizon_exit"

            if exit_price is None or exit_reason is None:
                continue

            gross_proceeds = position.quantity * exit_price
            exit_commission = gross_proceeds * request.commission_rate
            net_proceeds = gross_proceeds - exit_commission
            updated_cash += net_proceeds
            total_cost = position.gross_value + position.commission_paid
            pnl = net_proceeds - total_cost
            return_pct = pnl / total_cost if total_cost else 0.0
            trades.append(
                BacktestTrade(
                    ts_code=ts_code,
                    signal_date=position.signal_date,
                    entry_date=position.entry_date,
                    entry_price=round(position.entry_price, 4),
                    exit_date=trade_date,
                    exit_price=round(exit_price, 4),
                    exit_reason=exit_reason,
                    return_pct=round(return_pct, 6),
                    pnl=round(pnl, 4),
                    holding_days=self._holding_days(position.entry_date, trade_date, trading_dates),
                )
            )
            open_positions.pop(ts_code, None)

        return updated_cash

    def _index_bars(self, bars: list[DailyBar]) -> dict[str, DailyBar]:
        return {bar.trade_date: bar for bar in bars}

    def _mark_to_market(
        self,
        *,
        open_positions: dict[str, OpenPosition],
        bars_by_symbol: dict[str, dict[str, DailyBar]],
        trade_date: str,
    ) -> float:
        value = 0.0
        for ts_code, position in open_positions.items():
            bar = bars_by_symbol.get(ts_code, {}).get(trade_date)
            if bar is None or bar.close is None:
                continue
            value += position.quantity * bar.close
        return value

    def _entry_price_allowed(self, report: StructuredAnalysis, next_open: float) -> bool:
        zone = report.decision.entry_zone
        if zone is None:
            return True
        lower = zone.low if zone.low is not None else zone.high
        upper = zone.high if zone.high is not None else zone.low
        if lower is None and upper is None:
            return True
        if lower is None:
            lower = upper
        if upper is None:
            upper = lower
        assert lower is not None
        assert upper is not None
        return lower <= next_open <= upper

    def _holding_days(self, entry_date: str, exit_date: str, trading_dates: list[str]) -> int:
        date_to_index = {item: index for index, item in enumerate(trading_dates)}
        return max(date_to_index.get(exit_date, 0) - date_to_index.get(entry_date, 0) + 1, 1)

    def _build_metrics(
        self,
        *,
        initial_cash: float,
        ending_cash: float,
        trades: list[BacktestTrade],
        daily_positions: list[BacktestDailyPosition],
        trading_days: int,
    ) -> BacktestMetrics:
        total_return = (ending_cash - initial_cash) / initial_cash if initial_cash else 0.0
        annual_return = 0.0
        if trading_days > 0 and initial_cash > 0:
            annual_return = (ending_cash / initial_cash) ** (252 / trading_days) - 1 if ending_cash > 0 else -1.0

        max_drawdown = 0.0
        peak = 0.0
        for item in daily_positions:
            peak = max(peak, item.equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - item.equity) / peak)

        wins = [item for item in trades if item.pnl > 0]
        losses = [item for item in trades if item.pnl < 0]
        gross_profit = sum(item.pnl for item in wins)
        gross_loss = abs(sum(item.pnl for item in losses))
        avg_win = sum(item.return_pct for item in wins) / len(wins) if wins else 0.0
        avg_loss = sum(item.return_pct for item in losses) / len(losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

        return BacktestMetrics(
            total_return=round(total_return, 6),
            annual_return=round(annual_return, 6),
            max_drawdown=round(max_drawdown, 6),
            win_rate=round(len(wins) / len(trades), 6) if trades else 0.0,
            avg_win=round(avg_win, 6),
            avg_loss=round(avg_loss, 6),
            profit_factor=round(profit_factor, 6) if profit_factor is not None else None,
            trade_count=len(trades),
        )

def _max_holding_days(holding_horizon: str) -> int:
    if holding_horizon == "intraday":
        return 1
    if holding_horizon == "swing":
        return 5
    return 20

"""Recommendation performance tracking service."""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from octts.clients.tushare_client import TushareClient
from octts.config import Settings
from octts.schemas.backtest import DailyBar
from octts.services.screening_store import ScreeningStore


class RecommendationTracker:
    DEFAULT_BENCHMARK_CODE = "000300.SH"
    WINDOW_OFFSETS = [1, 3, 5, 10]

    def __init__(
        self,
        settings: Settings,
        store: Optional[ScreeningStore] = None,
        tushare_client: Optional[TushareClient] = None,
    ) -> None:
        self.settings = settings
        self.store = store or ScreeningStore(settings)
        self.tushare_client = tushare_client or TushareClient(settings)

    def update_recommendation_performance(self, lookback_days: int = 15) -> Dict[str, Any]:
        pending_items = self.store.list_pending_performance_updates(lookback_days=lookback_days)
        updated_items = []
        for item in pending_items:
            updated = self._update_single_item(item)
            if updated is not None:
                updated_items.append(updated)
        return {
            "lookback_days": lookback_days,
            "pending_count": len(pending_items),
            "updated_count": len(updated_items),
            "items": updated_items,
        }

    def _update_single_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trade_date = self._normalize_trade_date(item.get("trade_date"))
        ts_code = item.get("ts_code")
        if not trade_date or not ts_code:
            return None

        trading_dates = self._fetch_window_trading_dates(trade_date)
        if not trading_dates:
            return None

        target_dates = self._resolve_target_dates(trade_date, trading_dates)
        max_target_date = target_dates.get(10) or trading_dates[-1]
        stock_bars = self._fetch_bar_map(ts_code, trade_date, max_target_date)
        benchmark_bars = self._fetch_bar_map(self.DEFAULT_BENCHMARK_CODE, trade_date, max_target_date)
        entry_bar = stock_bars.get(trade_date)
        entry_price = self._close_price(entry_bar)
        if entry_price is None:
            return None

        latest_trade_date = max(stock_bars.keys()) if stock_bars else trade_date
        latest_price = self._close_price(stock_bars.get(latest_trade_date))
        tracking_days = self._count_tracking_days(trade_date, latest_trade_date, trading_dates)
        performance = {
            "entry_price": entry_price,
            "latest_price": latest_price,
            "tracking_days": tracking_days,
            "status": self._determine_status(tracking_days),
            "benchmark_code": self.DEFAULT_BENCHMARK_CODE,
        }

        for offset in self.WINDOW_OFFSETS:
            target_date = target_dates.get(offset)
            if not target_date:
                continue
            target_bar = stock_bars.get(target_date)
            target_price = self._close_price(target_bar)
            if target_price is None:
                continue
            performance[f"return_{offset}d"] = self._compute_return(entry_price, target_price)

        price_window = self._collect_window_prices(stock_bars, trading_dates, trade_date, 10)
        performance["max_drawdown_10d"] = self._compute_max_drawdown(entry_price, price_window)

        benchmark_target_date = target_dates.get(5)
        if benchmark_target_date:
            benchmark_entry = self._close_price(benchmark_bars.get(trade_date))
            benchmark_target = self._close_price(benchmark_bars.get(benchmark_target_date))
            benchmark_return = self._compute_return(benchmark_entry, benchmark_target)
            performance["benchmark_return_5d"] = benchmark_return
            stock_return = performance.get("return_5d")
            if stock_return is not None and benchmark_return is not None:
                performance["vs_benchmark_5d"] = stock_return - benchmark_return
                performance["hit_5d"] = stock_return > 0

        return self.store.upsert_recommendation_performance(item["id"], performance)

    def _fetch_window_trading_dates(self, trade_date: str) -> List[str]:
        end_date = (datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y%m%d")
        dates = self.tushare_client.fetch_trading_dates(
            start_date=trade_date.replace("-", ""),
            end_date=end_date,
        )
        return [self._normalize_trade_date(value) for value in dates if self._normalize_trade_date(value)]

    @classmethod
    def _resolve_target_dates(cls, trade_date: str, trading_dates: List[str]) -> Dict[int, str]:
        if trade_date not in trading_dates:
            return {}
        start_index = trading_dates.index(trade_date)
        target_dates: Dict[int, str] = {}
        for offset in cls.WINDOW_OFFSETS:
            index = start_index + offset
            if index < len(trading_dates):
                target_dates[offset] = trading_dates[index]
        return target_dates

    def _fetch_bar_map(self, ts_code: str, start_date: str, end_date: str) -> Dict[str, DailyBar]:
        bars = self.tushare_client.fetch_daily_bars(
            ts_code=ts_code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        )
        return {self._normalize_trade_date(bar.trade_date): bar for bar in bars if self._normalize_trade_date(bar.trade_date)}

    @staticmethod
    def _compute_return(entry_price: Optional[float], target_price: Optional[float]) -> Optional[float]:
        if entry_price in (None, 0) or target_price is None:
            return None
        return (target_price - entry_price) / entry_price

    @staticmethod
    def _compute_max_drawdown(entry_price: Optional[float], prices: List[float]) -> Optional[float]:
        if entry_price in (None, 0) or not prices:
            return None
        min_return = min((price - entry_price) / entry_price for price in prices)
        return min_return if min_return < 0 else 0.0

    @staticmethod
    def _close_price(bar: Optional[DailyBar]) -> Optional[float]:
        if bar is None:
            return None
        return bar.close

    @staticmethod
    def _normalize_trade_date(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if len(text) == 8 and text.isdigit():
            return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
        if len(text) == 10 and text[4] == '-' and text[7] == '-':
            return text
        return None

    @staticmethod
    def _count_tracking_days(trade_date: str, latest_trade_date: str, trading_dates: List[str]) -> int:
        if trade_date not in trading_dates or latest_trade_date not in trading_dates:
            return 0
        start_index = trading_dates.index(trade_date)
        latest_index = trading_dates.index(latest_trade_date)
        return max(0, latest_index - start_index)

    @staticmethod
    def _determine_status(tracking_days: int) -> str:
        if tracking_days >= 10:
            return "validated"
        if tracking_days >= 1:
            return "tracking"
        return "new"

    @staticmethod
    def _collect_window_prices(
        bar_map: Dict[str, DailyBar],
        trading_dates: List[str],
        trade_date: str,
        window: int,
    ) -> List[float]:
        if trade_date not in trading_dates:
            return []
        start_index = trading_dates.index(trade_date)
        prices: List[float] = []
        for target_date in trading_dates[start_index + 1:start_index + window + 1]:
            bar = bar_map.get(target_date)
            if bar is None or bar.close is None:
                continue
            prices.append(bar.close)
        return prices

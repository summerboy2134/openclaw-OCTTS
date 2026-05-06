from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Tuple

from octts.config import Settings

logger = logging.getLogger(__name__)
from octts.schemas.training import RAW_MARKET_FEATURE_SCHEMA_VERSION, RawMarketTrainingSample
from octts.services.market_raw_data_repository import MarketRawDataRepository


class RawMarketTrainingDatasetBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repo = MarketRawDataRepository(settings.database_url)

    def build_samples(
        self,
        *,
        start_date: date,
        end_date: date,
        min_history_days: int = 20,
        exclude_bj: bool = False,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[RawMarketTrainingSample]:
        """Build training samples.

        Args:
            start_date: Sample start date
            end_date: Sample end date
            min_history_days: Minimum history days required
            exclude_bj: Exclude Beijing stock exchange
            progress_callback: Optional callback(current, total, samples_count) for progress
        """
        trading_dates = self.repo.list_trading_dates(
            start_date=(start_date - timedelta(days=60)).strftime("%Y%m%d"),
            end_date=(end_date + timedelta(days=20)).strftime("%Y%m%d"),
        )
        sample_trade_dates = [
            value
            for value in trading_dates
            if start_date <= datetime.strptime(value, "%Y%m%d").date() <= end_date
        ]
        logger.info("加载日线数据...")
        daily_rows = self._load_daily_rows(trading_dates)
        logger.info("加载基本面数据...")
        daily_basic_rows = self._load_daily_basic_rows(trading_dates)
        logger.info("构建市场上下文...")
        market_context = self._build_market_context(sample_trade_dates, daily_rows)
        logger.info("构建排名上下文...")
        rank_context = self._build_rank_context(sample_trade_dates, daily_rows, daily_basic_rows)

        samples: List[RawMarketTrainingSample] = []
        all_codes = sorted(set(daily_rows.keys()) & set(daily_basic_rows.keys()))
        if exclude_bj:
            all_codes = [ts_code for ts_code in all_codes if not str(ts_code).strip().upper().endswith(".BJ")]

        total_codes = len(all_codes)
        logger.info("开始构建样本，共 %d 只股票...", total_codes)

        for idx, ts_code in enumerate(all_codes):
            samples.extend(
                self._build_samples_for_code(
                    ts_code=ts_code,
                    sample_trade_dates=sample_trade_dates,
                    all_trading_dates=trading_dates,
                    daily_map=daily_rows.get(ts_code, {}),
                    basic_map=daily_basic_rows.get(ts_code, {}),
                    market_context=market_context,
                    rank_context=rank_context,
                    min_history_days=min_history_days,
                )
            )
            # 每处理500只股票打印一次进度
            if progress_callback:
                progress_callback(idx + 1, total_codes, len(samples))
            elif (idx + 1) % 500 == 0 or (idx + 1) == total_codes:
                logger.info("进度: %d/%d (%.1f%%), 当前样本数: %d",
                           idx + 1, total_codes, (idx + 1) / total_codes * 100, len(samples))
        return samples

    def build_samples_for_codes(
        self,
        ts_codes: List[str],
        *,
        start_date: date,
        end_date: date,
        min_history_days: int = 20,
    ) -> List[RawMarketTrainingSample]:
        """Build samples only for specified stock codes (optimized for rerank).

        This is an optimized version that only loads data for the specified codes,
        rather than loading all ~5500 stocks from the database.
        """
        if not ts_codes:
            return []

        # Normalize codes
        code_set = {str(code).strip().upper() for code in ts_codes}

        trading_dates = self.repo.list_trading_dates(
            start_date=(start_date - timedelta(days=60)).strftime("%Y%m%d"),
            end_date=(end_date + timedelta(days=20)).strftime("%Y%m%d"),
        )
        sample_trade_dates = [
            value
            for value in trading_dates
            if start_date <= datetime.strptime(value, "%Y%m%d").date() <= end_date
        ]

        # Load only the specified codes' data
        daily_rows = self._load_daily_rows_for_codes(trading_dates, code_set)
        daily_basic_rows = self._load_daily_basic_rows_for_codes(trading_dates, code_set)

        # Build market context from loaded codes only (no full market scan)
        market_context = self._build_market_context(sample_trade_dates, daily_rows)
        rank_context = self._build_rank_context(sample_trade_dates, daily_rows, daily_basic_rows)

        samples: List[RawMarketTrainingSample] = []
        for ts_code in code_set:
            if ts_code not in daily_rows or ts_code not in daily_basic_rows:
                continue
            samples.extend(
                self._build_samples_for_code(
                    ts_code=ts_code,
                    sample_trade_dates=sample_trade_dates,
                    all_trading_dates=trading_dates,
                    daily_map=daily_rows.get(ts_code, {}),
                    basic_map=daily_basic_rows.get(ts_code, {}),
                    market_context=market_context,
                    rank_context=rank_context,
                    min_history_days=min_history_days,
                )
            )
        return samples

    def build_dataset_summary(
        self,
        *,
        start_date: date,
        end_date: date,
        min_history_days: int = 20,
        exclude_bj: bool = False,
    ) -> Dict[str, Any]:
        samples = self.build_samples(
            start_date=start_date,
            end_date=end_date,
            min_history_days=min_history_days,
            exclude_bj=exclude_bj,
        )
        labeled = [sample for sample in samples if sample.label_up_1d is not None]
        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "schema_version": RAW_MARKET_FEATURE_SCHEMA_VERSION,
            "sample_count": len(samples),
            "labeled_count": len(labeled),
            "ts_code_count": len({sample.ts_code for sample in samples}),
            "exclude_bj": exclude_bj,
        }, samples

    def _load_daily_rows(self, trading_dates: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        start_date = trading_dates[0]
        end_date = trading_dates[-1]
        session = self.repo._db.get_session()
        try:
            from octts.models.screening_models import MarketDaily

            rows = (
                session.query(MarketDaily)
                .filter(
                    MarketDaily.trade_date >= self.repo._parse_date(start_date),
                    MarketDaily.trade_date <= self.repo._parse_date(end_date),
                )
                .all()
            )
            result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                trade_key = row.trade_date.strftime("%Y%m%d")
                result[row.ts_code][trade_key] = self.repo._serialize_market_daily(row)
            return dict(result)
        finally:
            session.close()

    def _load_daily_rows_for_codes(
        self, trading_dates: List[str], ts_codes: set
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load daily data only for specified stock codes."""
        if not ts_codes:
            return {}
        start_date = trading_dates[0]
        end_date = trading_dates[-1]
        session = self.repo._db.get_session()
        try:
            from octts.models.screening_models import MarketDaily

            rows = (
                session.query(MarketDaily)
                .filter(
                    MarketDaily.trade_date >= self.repo._parse_date(start_date),
                    MarketDaily.trade_date <= self.repo._parse_date(end_date),
                    MarketDaily.ts_code.in_(ts_codes),
                )
                .all()
            )
            result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                trade_key = row.trade_date.strftime("%Y%m%d")
                result[row.ts_code][trade_key] = self.repo._serialize_market_daily(row)
            return dict(result)
        finally:
            session.close()

    def _load_daily_basic_rows(self, trading_dates: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        start_date = trading_dates[0]
        end_date = trading_dates[-1]
        session = self.repo._db.get_session()
        try:
            from octts.models.screening_models import MarketDailyBasic

            rows = (
                session.query(MarketDailyBasic)
                .filter(
                    MarketDailyBasic.trade_date >= self.repo._parse_date(start_date),
                    MarketDailyBasic.trade_date <= self.repo._parse_date(end_date),
                )
                .all()
            )
            result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                trade_key = row.trade_date.strftime("%Y%m%d")
                result[row.ts_code][trade_key] = self.repo._serialize_market_daily_basic(row)
            return dict(result)
        finally:
            session.close()

    def _load_daily_basic_rows_for_codes(
        self, trading_dates: List[str], ts_codes: set
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load daily basic data only for specified stock codes."""
        if not ts_codes:
            return {}
        start_date = trading_dates[0]
        end_date = trading_dates[-1]
        session = self.repo._db.get_session()
        try:
            from octts.models.screening_models import MarketDailyBasic

            rows = (
                session.query(MarketDailyBasic)
                .filter(
                    MarketDailyBasic.trade_date >= self.repo._parse_date(start_date),
                    MarketDailyBasic.trade_date <= self.repo._parse_date(end_date),
                    MarketDailyBasic.ts_code.in_(ts_codes),
                )
                .all()
            )
            result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
            for row in rows:
                trade_key = row.trade_date.strftime("%Y%m%d")
                result[row.ts_code][trade_key] = self.repo._serialize_market_daily_basic(row)
            return dict(result)
        finally:
            session.close()

    def _build_market_context(
        self,
        sample_trade_dates: List[str],
        daily_rows: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        day_returns: Dict[str, List[float]] = {trade_date: [] for trade_date in sample_trade_dates}
        day_up_counts: Dict[str, int] = {trade_date: 0 for trade_date in sample_trade_dates}
        day_counts: Dict[str, int] = {trade_date: 0 for trade_date in sample_trade_dates}
        for daily_map in daily_rows.values():
            for trade_date in sample_trade_dates:
                row = daily_map.get(trade_date)
                if not isinstance(row, dict):
                    continue
                pct_change = self._safe_float(row.get("pct_chg"))
                if pct_change is None:
                    continue
                day_returns[trade_date].append(pct_change / 100.0)
                day_counts[trade_date] += 1
                if pct_change > 0:
                    day_up_counts[trade_date] += 1

        market_return_1d: Dict[str, Optional[float]] = {}
        market_up_ratio_1d: Dict[str, Optional[float]] = {}
        for trade_date in sample_trade_dates:
            returns = day_returns.get(trade_date, [])
            market_return_1d[trade_date] = mean(returns) if returns else None
            count = day_counts.get(trade_date, 0)
            market_up_ratio_1d[trade_date] = (day_up_counts.get(trade_date, 0) / count) if count else None

        context: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, trade_date in enumerate(sample_trade_dates):
            context[trade_date] = {
                "market_return_1d": market_return_1d.get(trade_date),
                "market_return_3d": self._rolling_market_average(sample_trade_dates, market_return_1d, idx, 3),
                "market_return_5d": self._rolling_market_average(sample_trade_dates, market_return_1d, idx, 5),
                "market_up_ratio_1d": market_up_ratio_1d.get(trade_date),
                "market_up_ratio_3d_avg": self._rolling_market_average(sample_trade_dates, market_up_ratio_1d, idx, 3),
                "market_up_days_5d": self._rolling_market_up_days(sample_trade_dates, market_return_1d, idx, 5),
            }
        return context

    def _build_rank_context(
        self,
        sample_trade_dates: List[str],
        daily_rows: Dict[str, Dict[str, Dict[str, Any]]],
        daily_basic_rows: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        context: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
        for trade_date in sample_trade_dates:
            pct_change_items: List[Tuple[str, float]] = []
            turnover_items: List[Tuple[str, float]] = []
            volume_ratio_items: List[Tuple[str, float]] = []
            for ts_code, daily_map in daily_rows.items():
                daily_row = daily_map.get(trade_date)
                basic_row = daily_basic_rows.get(ts_code, {}).get(trade_date)
                if isinstance(daily_row, dict):
                    pct_change = self._safe_float(daily_row.get("pct_chg"))
                    if pct_change is not None:
                        pct_change_items.append((ts_code, pct_change))
                if isinstance(basic_row, dict):
                    turnover_rate = self._safe_float(basic_row.get("turnover_rate"))
                    if turnover_rate is not None:
                        turnover_items.append((ts_code, turnover_rate))
                    volume_ratio = self._safe_float(basic_row.get("volume_ratio"))
                    if volume_ratio is not None:
                        volume_ratio_items.append((ts_code, volume_ratio))
            context[trade_date] = {
                "pct_change_rank_pct": self._compute_rank_percentiles(pct_change_items),
                "turnover_rate_rank_pct": self._compute_rank_percentiles(turnover_items),
                "volume_ratio_rank_pct": self._compute_rank_percentiles(volume_ratio_items),
            }
        return context

    def _build_samples_for_code(
        self,
        *,
        ts_code: str,
        sample_trade_dates: List[str],
        all_trading_dates: List[str],
        daily_map: Dict[str, Dict[str, Any]],
        basic_map: Dict[str, Dict[str, Any]],
        market_context: Dict[str, Dict[str, Optional[float]]],
        rank_context: Dict[str, Dict[str, Dict[str, Optional[float]]]],
        min_history_days: int,
    ) -> List[RawMarketTrainingSample]:
        available_dates = [value for value in all_trading_dates if value in daily_map and value in basic_map]
        if not available_dates:
            return []
        index_map = {value: idx for idx, value in enumerate(available_dates)}
        closes = [self._safe_float(daily_map[value].get("close")) for value in available_dates]
        samples: List[RawMarketTrainingSample] = []

        for trade_date_text in sample_trade_dates:
            idx = index_map.get(trade_date_text)
            if idx is None or idx < min_history_days - 1:
                continue
            current_close = closes[idx]
            if current_close in (None, 0):
                continue
            daily_row = daily_map[trade_date_text]
            basic_row = basic_map[trade_date_text]
            history_closes = closes[:idx + 1]
            history_amounts = [self._safe_float(daily_map[value].get("amount")) for value in available_dates[:idx + 1]]
            market_row = market_context.get(trade_date_text, {})
            sample = RawMarketTrainingSample(
                trade_date=datetime.strptime(trade_date_text, "%Y%m%d").date(),
                ts_code=ts_code,
                entry_price=current_close,
                close=current_close,
                pct_change=self._safe_float(daily_row.get("pct_chg")),
                turnover_rate=self._safe_float(basic_row.get("turnover_rate")),
                volume_ratio=self._safe_float(basic_row.get("volume_ratio")),
                market_cap=self._safe_float(basic_row.get("total_mv")),
                pe_ttm=self._safe_float(basic_row.get("pe_ttm")),
                pb=self._safe_float(basic_row.get("pb")),
                amount=self._safe_float(daily_row.get("amount")),
                vol=self._safe_float(daily_row.get("vol")),
                return_3d_past=self._window_return(history_closes, 3),
                return_5d_past=self._window_return(history_closes, 5),
                return_10d_past=self._window_return(history_closes, 10),
                volatility_5d=self._window_volatility(history_closes, 5),
                volatility_10d=self._window_volatility(history_closes, 10),
                max_drawdown_10d_past=self._window_max_drawdown(history_closes, 10),
                close_to_ma5=self._close_to_ma(history_closes, 5),
                close_to_ma10=self._close_to_ma(history_closes, 10),
                close_to_ma20=self._close_to_ma(history_closes, 20),
                price_position_20d=self._price_position(history_closes, 20),
                price_position_10d=self._price_position(history_closes, 10),
                avg_turnover_rate_5d=self._rolling_basic_average(basic_map, available_dates, idx, "turnover_rate", 5),
                avg_volume_ratio_5d=self._rolling_basic_average(basic_map, available_dates, idx, "volume_ratio", 5),
                market_return_1d=market_row.get("market_return_1d"),
                market_return_3d=market_row.get("market_return_3d"),
                market_return_5d=market_row.get("market_return_5d"),
                market_up_ratio_1d=market_row.get("market_up_ratio_1d"),
                market_up_ratio_3d_avg=market_row.get("market_up_ratio_3d_avg"),
                market_up_days_5d=market_row.get("market_up_days_5d"),
                stock_vs_market_return_1d=self._subtract_optional(self._window_return(history_closes, 2), market_row.get("market_return_1d")),
                stock_vs_market_return_2d=self._subtract_optional(self._window_return(history_closes, 3), self._rolling_market_average(sample_trade_dates, {d: market_context.get(d, {}).get("market_return_1d") for d in sample_trade_dates}, sample_trade_dates.index(trade_date_text), 2)),
                stock_vs_market_return_3d=self._subtract_optional(self._window_return(history_closes, 3), market_row.get("market_return_3d")),
                stock_vs_market_return_5d=self._subtract_optional(self._window_return(history_closes, 5), market_row.get("market_return_5d")),
                stock_vs_market_return_10d=self._subtract_optional(self._window_return(history_closes, 10), self._rolling_market_average(sample_trade_dates, {d: market_context.get(d, {}).get("market_return_1d") for d in sample_trade_dates}, sample_trade_dates.index(trade_date_text), 10)),
                pct_change_rank_pct=rank_context.get(trade_date_text, {}).get("pct_change_rank_pct", {}).get(ts_code),
                turnover_rate_rank_pct=rank_context.get(trade_date_text, {}).get("turnover_rate_rank_pct", {}).get(ts_code),
                volume_ratio_rank_pct=rank_context.get(trade_date_text, {}).get("volume_ratio_rank_pct", {}).get(ts_code),
                up_days_3d=self._up_days(history_closes, 3),
                up_days_5d=self._up_days(history_closes, 5),
                new_high_gap_20d=self._distance_to_window_high(history_closes, 20),
                new_high_gap_10d=self._distance_to_window_high(history_closes, 10),
                new_low_gap_20d=self._distance_to_window_low(history_closes, 20),
                amount_ratio_1d_5d=self._ratio_of_window_means(history_amounts, 1, 5),
                amount_ratio_3d_10d=self._ratio_of_window_means(history_amounts, 3, 10),
                turnover_rate_change_1d=self._change_vs_window_average(basic_map, available_dates, idx, "turnover_rate", 1),
                turnover_rate_change_5d=self._change_vs_window_average(basic_map, available_dates, idx, "turnover_rate", 5),
                return_1d=self._future_return(closes, idx, 1),
                return_3d=self._future_return(closes, idx, 3),
                return_5d=self._future_return(closes, idx, 5),
                return_10d=self._future_return(closes, idx, 10),
            )
            sample.vs_market_1d = self._subtract_optional(sample.return_1d, market_row.get("market_return_1d"))
            sample.vs_market_3d = self._subtract_optional(sample.return_3d, market_row.get("market_return_3d"))
            sample.vs_market_5d = self._subtract_optional(sample.return_5d, market_row.get("market_return_5d"))
            sample.label_up_1d = None if sample.return_1d is None else sample.return_1d > 0
            sample.label_up_3d = None if sample.return_3d is None else sample.return_3d > 0
            sample.label_up_5d = None if sample.return_5d is None else sample.return_5d > 0
            sample.label_vs_market_1d = None if sample.vs_market_1d is None else sample.vs_market_1d > 0
            sample.label_vs_market_3d = None if sample.vs_market_3d is None else sample.vs_market_3d > 0
            sample.label_vs_market_5d = None if sample.vs_market_5d is None else sample.vs_market_5d > 0
            sample.label_strong_1d = None if sample.vs_market_1d is None else sample.vs_market_1d >= 0.02
            samples.append(sample)
        return samples

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _subtract_optional(left: Optional[float], right: Optional[float]) -> Optional[float]:
        if left is None or right is None:
            return None
        return left - right

    @staticmethod
    def _compute_rank_percentiles(items: List[Tuple[str, float]]) -> Dict[str, float]:
        if not items:
            return {}
        ordered = sorted(items, key=lambda item: item[1])
        total = len(ordered)
        if total == 1:
            return {ordered[0][0]: 1.0}
        result: Dict[str, float] = {}
        for idx, (ts_code, _) in enumerate(ordered):
            result[ts_code] = idx / (total - 1)
        return result

    @staticmethod
    def _rolling_market_average(
        sample_trade_dates: List[str],
        values: Dict[str, Optional[float]],
        idx: int,
        window: int,
    ) -> Optional[float]:
        if idx + 1 < window:
            return None
        collected = [
            values.get(trade_date)
            for trade_date in sample_trade_dates[idx - window + 1:idx + 1]
            if values.get(trade_date) is not None
        ]
        if not collected:
            return None
        return mean(collected)

    @staticmethod
    def _rolling_market_up_days(
        sample_trade_dates: List[str],
        values: Dict[str, Optional[float]],
        idx: int,
        window: int,
    ) -> Optional[int]:
        if idx + 1 < window:
            return None
        window_values = [values.get(trade_date) for trade_date in sample_trade_dates[idx - window + 1:idx + 1]]
        if any(value is None for value in window_values):
            return None
        return sum(1 for value in window_values if value and value > 0)

    @classmethod
    def _window_return(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window or closes[-1] in (None, 0) or closes[-window] in (None, 0):
            return None
        start = closes[-window]
        end = closes[-1]
        if start in (None, 0) or end is None:
            return None
        return (end - start) / start

    @classmethod
    def _future_return(cls, closes: List[Optional[float]], idx: int, offset: int) -> Optional[float]:
        if idx + offset >= len(closes):
            return None
        entry = closes[idx]
        future = closes[idx + offset]
        if entry in (None, 0) or future is None:
            return None
        return (future - entry) / entry

    @classmethod
    def _window_volatility(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window:
            return None
        values = [value for value in closes[-window:] if value not in (None, 0)]
        if len(values) < 2:
            return None
        returns = []
        for prev, curr in zip(values[:-1], values[1:]):
            if prev in (None, 0) or curr is None:
                continue
            returns.append((curr - prev) / prev)
        if len(returns) < 2:
            return None
        return float(pstdev(returns))

    @classmethod
    def _window_max_drawdown(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window:
            return None
        values = [value for value in closes[-window:] if value not in (None, 0)]
        if len(values) < 2:
            return None
        peak = values[0]
        worst_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            drawdown = (value - peak) / peak if peak else 0.0
            worst_drawdown = min(worst_drawdown, drawdown)
        return worst_drawdown

    @classmethod
    def _close_to_ma(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window:
            return None
        values = [value for value in closes[-window:] if value is not None]
        if len(values) < window:
            return None
        ma = mean(values)
        close = values[-1]
        if ma == 0:
            return None
        return (close - ma) / ma

    @classmethod
    def _price_position(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window:
            return None
        values = [value for value in closes[-window:] if value is not None]
        if len(values) < window:
            return None
        low = min(values)
        high = max(values)
        if high == low:
            return 0.5
        return (values[-1] - low) / (high - low)

    @classmethod
    def _up_days(cls, closes: List[Optional[float]], window: int) -> Optional[int]:
        if len(closes) < window:
            return None
        values = closes[-window:]
        if any(value is None for value in values):
            return None
        count = 0
        for prev, curr in zip(values[:-1], values[1:]):
            if prev in (None, 0) or curr is None:
                continue
            if curr > prev:
                count += 1
        return count

    @classmethod
    def _distance_to_window_high(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window or closes[-1] in (None, 0):
            return None
        values = [value for value in closes[-window:] if value not in (None, 0)]
        if len(values) < window:
            return None
        window_high = max(values)
        current = values[-1]
        return (current - window_high) / window_high if window_high else None

    @classmethod
    def _distance_to_window_low(cls, closes: List[Optional[float]], window: int) -> Optional[float]:
        if len(closes) < window or closes[-1] in (None, 0):
            return None
        values = [value for value in closes[-window:] if value not in (None, 0)]
        if len(values) < window:
            return None
        window_low = min(values)
        current = values[-1]
        return (current - window_low) / window_low if window_low else None

    @classmethod
    def _ratio_of_window_means(cls, values: List[Optional[float]], short_window: int, long_window: int) -> Optional[float]:
        if len(values) < long_window:
            return None
        short_values = [value for value in values[-short_window:] if value not in (None, 0)]
        long_values = [value for value in values[-long_window:] if value not in (None, 0)]
        if not short_values or not long_values:
            return None
        long_mean = mean(long_values)
        if long_mean == 0:
            return None
        return mean(short_values) / long_mean

    @classmethod
    def _change_vs_window_average(
        cls,
        basic_map: Dict[str, Dict[str, Any]],
        available_dates: List[str],
        idx: int,
        field: str,
        window: int,
    ) -> Optional[float]:
        if idx + 1 < window:
            return None
        current = cls._safe_float(basic_map.get(available_dates[idx], {}).get(field))
        average = cls._rolling_basic_average(basic_map, available_dates, idx, field, window)
        if current is None or average in (None, 0):
            return None
        return (current - average) / average

    @classmethod
    def _rolling_basic_average(
        cls,
        basic_map: Dict[str, Dict[str, Any]],
        available_dates: List[str],
        idx: int,
        field: str,
        window: int,
    ) -> Optional[float]:
        if idx + 1 < window:
            return None
        values: List[float] = []
        for trade_date_text in available_dates[idx - window + 1:idx + 1]:
            value = cls._safe_float(basic_map.get(trade_date_text, {}).get(field))
            if value is None:
                continue
            values.append(value)
        if not values:
            return None
        return mean(values)


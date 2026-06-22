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
        include_stock_moneyflow_in_limit_chase_risk: bool = True,
    ) -> List[RawMarketTrainingSample]:
        """Build training samples.

        Args:
            start_date: Sample start date
            end_date: Sample end date
            min_history_days: Minimum history days required
            exclude_bj: Exclude Beijing stock exchange
            progress_callback: Optional callback(current, total, samples_count) for progress
            include_stock_moneyflow_in_limit_chase_risk: Include per-stock moneyflow in
                limit_chase_failure_risk_score. Disable for pre-candidate universe ranking
                when prediction must match precomputed training feature semantics.
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
        logger.info("加载复权因子数据...")
        adj_factor_rows = self.repo.get_adj_factors_by_trade_dates(
            ts_codes=daily_rows.keys(),
            trading_dates=trading_dates,
        )
        logger.info("构建市场上下文...")
        market_context = self._build_market_context(sample_trade_dates, daily_rows)
        logger.info("构建排名上下文...")
        rank_context = self._build_rank_context(sample_trade_dates, daily_rows, daily_basic_rows)

        all_codes = sorted(set(daily_rows.keys()) & set(daily_basic_rows.keys()))
        if exclude_bj:
            all_codes = [ts_code for ts_code in all_codes if not str(ts_code).strip().upper().endswith(".BJ")]

        logger.info("加载涨停板数据...")
        limit_list_rows = self.repo.get_limit_list_by_trade_dates(
            ts_codes=all_codes,
            trading_dates=trading_dates,
        )
        logger.info("加载资金流数据...")
        moneyflow_rows = self.repo.get_moneyflow_by_trade_dates(
            ts_codes=all_codes,
            trading_dates=trading_dates,
        )

        samples: List[RawMarketTrainingSample] = []
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
                    adj_factor_map=adj_factor_rows.get(ts_code, {}),
                    limit_map=limit_list_rows.get(ts_code, {}),
                    moneyflow_map=moneyflow_rows.get(ts_code, {}),
                    market_context=market_context,
                    rank_context=rank_context,
                    min_history_days=min_history_days,
                    include_stock_moneyflow_in_limit_chase_risk=include_stock_moneyflow_in_limit_chase_risk,
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
        include_stock_moneyflow_in_limit_chase_risk: bool = True,
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
        adj_factor_rows = self.repo.get_adj_factors_by_trade_dates(
            ts_codes=code_set,
            trading_dates=trading_dates,
        )
        limit_list_rows = self.repo.get_limit_list_by_trade_dates(
            ts_codes=code_set,
            trading_dates=trading_dates,
        )
        moneyflow_rows = self.repo.get_moneyflow_by_trade_dates(
            ts_codes=code_set,
            trading_dates=trading_dates,
        )

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
                    adj_factor_map=adj_factor_rows.get(ts_code, {}),
                    limit_map=limit_list_rows.get(ts_code, {}),
                    moneyflow_map=moneyflow_rows.get(ts_code, {}),
                    market_context=market_context,
                    rank_context=rank_context,
                    min_history_days=min_history_days,
                    include_stock_moneyflow_in_limit_chase_risk=include_stock_moneyflow_in_limit_chase_risk,
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
        adj_factor_map: Dict[str, float],
        limit_map: Dict[str, Dict[str, Any]],
        moneyflow_map: Dict[str, Dict[str, Any]],
        market_context: Dict[str, Dict[str, Optional[float]]],
        rank_context: Dict[str, Dict[str, Dict[str, Optional[float]]]],
        min_history_days: int,
        include_stock_moneyflow_in_limit_chase_risk: bool = True,
    ) -> List[RawMarketTrainingSample]:
        available_dates = [value for value in all_trading_dates if value in daily_map and value in basic_map]
        if not available_dates:
            return []
        index_map = {value: idx for idx, value in enumerate(available_dates)}
        closes = [self._safe_float(daily_map[value].get("close")) for value in available_dates]
        adjusted_closes = self._build_forward_adjusted_closes(closes, available_dates, adj_factor_map)
        pct_changes = [self._safe_float(daily_map[value].get("pct_chg")) for value in available_dates]
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
            raw_open = self._safe_float(daily_row.get("open"))
            raw_high = self._safe_float(daily_row.get("high"))
            raw_low = self._safe_float(daily_row.get("low"))
            raw_pre_close = self._safe_float(daily_row.get("pre_close"))
            intraday_features = self._compute_intraday_features(
                open_price=raw_open,
                high_price=raw_high,
                low_price=raw_low,
                close_price=current_close,
                pre_close=raw_pre_close,
            )
            open_gap_pct = intraday_features.get("open_gap_pct")
            if open_gap_pct is not None and open_gap_pct >= 0.08:
                logger.debug(
                    "Large open gap detected and tolerated in sample construction: ts_code=%s trade_date=%s open_gap_pct=%.4f",
                    ts_code,
                    trade_date_text,
                    open_gap_pct,
                )
            history_closes = adjusted_closes[:idx + 1]
            history_amounts = [self._safe_float(daily_map[value].get("amount")) for value in available_dates[:idx + 1]]
            history_pct_changes = pct_changes[:idx + 1]
            close_continuity_flags = self._build_close_continuity_flags(daily_map, available_dates[:idx + 1])
            if self._has_adjusted_close_coverage(adj_factor_map, available_dates[:idx + 1]):
                close_continuity_flags = [True for _ in close_continuity_flags]
            missing_window_flag = bool(self._window_has_close_discontinuity(close_continuity_flags, 20))
            missing_feature_count = sum(
                1
                for value in (
                    self._window_volatility_guarded(history_closes, close_continuity_flags, 5),
                    self._window_volatility_guarded(history_closes, close_continuity_flags, 10),
                    self._window_max_drawdown_guarded(history_closes, close_continuity_flags, 10),
                    self._close_to_ma_guarded(history_closes, close_continuity_flags, 5),
                    self._close_to_ma_guarded(history_closes, close_continuity_flags, 10),
                    self._close_to_ma_guarded(history_closes, close_continuity_flags, 20),
                    self._price_position_guarded(history_closes, close_continuity_flags, 20),
                    self._price_position_guarded(history_closes, close_continuity_flags, 10),
                    self._distance_to_window_high_guarded(history_closes, close_continuity_flags, 20),
                    self._distance_to_window_high_guarded(history_closes, close_continuity_flags, 10),
                    self._distance_to_window_low_guarded(history_closes, close_continuity_flags, 20),
                )
                if value is None
            )
            market_row = market_context.get(trade_date_text, {})
            current_pct_change = self._safe_float(daily_row.get("pct_chg"))
            current_turnover_rate = self._safe_float(basic_row.get("turnover_rate"))
            current_volume_ratio = self._safe_float(basic_row.get("volume_ratio"))
            price_position_20d = self._price_position_guarded(history_closes, close_continuity_flags, 20)
            recent_runup_5d = self._recent_runup_pct(history_pct_changes, 5)
            turnover_spike_ratio = self._turnover_spike_ratio(basic_map, available_dates, idx)
            weak_market_flag = self._weak_market_flag(market_row)
            high_position_flag = None if price_position_20d is None else bool(price_position_20d >= 0.88)
            high_position_acceleration_flag = self._high_position_acceleration_flag(
                price_position=price_position_20d,
                pct_change=current_pct_change,
                volume_ratio=current_volume_ratio,
                turnover_rate=current_turnover_rate,
                recent_runup_5d=recent_runup_5d,
                turnover_spike_ratio=turnover_spike_ratio,
            )
            weak_market_high_position_flag = (
                None
                if weak_market_flag is None or high_position_flag is None
                else bool(weak_market_flag and high_position_flag)
            )
            prev_trade_date = available_dates[idx - 1] if idx > 0 else None
            prev_daily_row = daily_map.get(prev_trade_date, {}) if prev_trade_date else {}
            prev_limit_row = limit_map.get(prev_trade_date, {}) if prev_trade_date else {}
            prev_pct_change = self._safe_float(prev_daily_row.get("pct_chg"))
            prev_day_limit_up = self._is_limit_up_row(prev_limit_row, prev_pct_change)
            prev_day_limit_open_times = self._safe_int(prev_limit_row.get("open_times"))
            prev_day_limit_first_time = self._parse_limit_time_to_minutes(prev_limit_row.get("first_time"))
            prev_day_limit_last_time = self._parse_limit_time_to_minutes(prev_limit_row.get("last_time"))
            prev_day_up_stat_success, prev_day_up_stat_total, prev_day_up_stat_ratio = self._parse_up_stat(
                prev_limit_row.get("up_stat")
            )
            prev_day_limit_amount = self._safe_float(prev_limit_row.get("limit_amount"))
            prev_day_fd_amount = self._safe_float(prev_limit_row.get("fd_amount"))
            prev_day_limit_times = self._safe_float(prev_limit_row.get("limit_times"))
            prev_day_one_word_limit_flag = self._is_one_word_limit(
                limit_up=prev_day_limit_up,
                open_times=prev_day_limit_open_times,
                daily_row=prev_daily_row,
            )
            moneyflow_features = self._build_moneyflow_features(moneyflow_map, available_dates, idx)
            risk_moneyflow_features = moneyflow_features if include_stock_moneyflow_in_limit_chase_risk else {}
            limit_chase_failure_risk_score = self._limit_chase_failure_risk_score(
                pct_change=current_pct_change,
                prev_day_limit_up=prev_day_limit_up,
                moneyflow_net_3d=risk_moneyflow_features.get("moneyflow_net_3d"),
                moneyflow_large_net_3d=risk_moneyflow_features.get("moneyflow_large_net_3d"),
                moneyflow_elarge_net_3d=risk_moneyflow_features.get("moneyflow_elarge_net_3d"),
                price_position_20d=price_position_20d,
                turnover_spike_ratio=turnover_spike_ratio,
                weak_market_high_position_flag=weak_market_high_position_flag,
            )
            limit_like_moneyflow_divergence_flag = bool(
                current_pct_change is not None
                and current_pct_change >= 9.5
                and (
                    (
                        moneyflow_features.get("moneyflow_net_3d") is not None
                        and moneyflow_features.get("moneyflow_net_3d") <= 0
                    )
                    or (
                        moneyflow_features.get("moneyflow_large_net_3d") is not None
                        and moneyflow_features.get("moneyflow_large_net_3d") <= 0
                    )
                    or (
                        moneyflow_features.get("moneyflow_elarge_net_3d") is not None
                        and moneyflow_features.get("moneyflow_elarge_net_3d") <= 0
                    )
                )
            )
            future_return_1d = self._future_compound_return(pct_changes, idx, 1)
            future_return_3d = self._future_compound_return(pct_changes, idx, 3)
            future_return_5d = self._future_compound_return(pct_changes, idx, 5)
            future_return_10d = self._future_compound_return(pct_changes, idx, 10)
            future_next_pct_change = pct_changes[idx + 1] if idx + 1 < len(pct_changes) else None
            sample = RawMarketTrainingSample(
                trade_date=datetime.strptime(trade_date_text, "%Y%m%d").date(),
                ts_code=ts_code,
                entry_price=current_close,
                close=current_close,
                pct_change=current_pct_change,
                turnover_rate=current_turnover_rate,
                volume_ratio=current_volume_ratio,
                market_cap=self._safe_float(basic_row.get("total_mv")),
                pe_ttm=self._safe_float(basic_row.get("pe_ttm")),
                pb=self._safe_float(basic_row.get("pb")),
                amount=self._safe_float(daily_row.get("amount")),
                vol=self._safe_float(daily_row.get("vol")),
                return_3d_past=self._window_compound_return(history_pct_changes, 3),
                return_5d_past=self._window_compound_return(history_pct_changes, 5),
                return_10d_past=self._window_compound_return(history_pct_changes, 10),
                volatility_5d=self._window_volatility_guarded(history_closes, close_continuity_flags, 5),
                volatility_10d=self._window_volatility_guarded(history_closes, close_continuity_flags, 10),
                max_drawdown_10d_past=self._window_max_drawdown_guarded(history_closes, close_continuity_flags, 10),
                close_to_ma5=self._close_to_ma_guarded(history_closes, close_continuity_flags, 5),
                close_to_ma10=self._close_to_ma_guarded(history_closes, close_continuity_flags, 10),
                close_to_ma20=self._close_to_ma_guarded(history_closes, close_continuity_flags, 20),
                price_position_20d=price_position_20d,
                price_position_10d=self._price_position_guarded(history_closes, close_continuity_flags, 10),
                avg_turnover_rate_5d=self._rolling_basic_average(basic_map, available_dates, idx, "turnover_rate", 5),
                avg_volume_ratio_5d=self._rolling_basic_average(basic_map, available_dates, idx, "volume_ratio", 5),
                market_return_1d=market_row.get("market_return_1d"),
                market_return_3d=market_row.get("market_return_3d"),
                market_return_5d=market_row.get("market_return_5d"),
                market_up_ratio_1d=market_row.get("market_up_ratio_1d"),
                market_up_ratio_3d_avg=market_row.get("market_up_ratio_3d_avg"),
                market_up_days_5d=market_row.get("market_up_days_5d"),
                weak_market_flag=weak_market_flag,
                stock_vs_market_return_1d=self._subtract_optional(self._window_compound_return(history_pct_changes, 2), market_row.get("market_return_1d")),
                stock_vs_market_return_2d=self._subtract_optional(self._window_compound_return(history_pct_changes, 3), self._rolling_market_average(sample_trade_dates, {d: market_context.get(d, {}).get("market_return_1d") for d in sample_trade_dates}, sample_trade_dates.index(trade_date_text), 2)),
                stock_vs_market_return_3d=self._subtract_optional(self._window_compound_return(history_pct_changes, 3), market_row.get("market_return_3d")),
                stock_vs_market_return_5d=self._subtract_optional(self._window_compound_return(history_pct_changes, 5), market_row.get("market_return_5d")),
                stock_vs_market_return_10d=self._subtract_optional(self._window_compound_return(history_pct_changes, 10), self._rolling_market_average(sample_trade_dates, {d: market_context.get(d, {}).get("market_return_1d") for d in sample_trade_dates}, sample_trade_dates.index(trade_date_text), 10)),
                pct_change_rank_pct=rank_context.get(trade_date_text, {}).get("pct_change_rank_pct", {}).get(ts_code),
                turnover_rate_rank_pct=rank_context.get(trade_date_text, {}).get("turnover_rate_rank_pct", {}).get(ts_code),
                volume_ratio_rank_pct=rank_context.get(trade_date_text, {}).get("volume_ratio_rank_pct", {}).get(ts_code),
                up_days_3d=self._up_days_from_pct_changes(history_pct_changes, 3),
                up_days_5d=self._up_days_from_pct_changes(history_pct_changes, 5),
                new_high_gap_20d=self._distance_to_window_high_guarded(history_closes, close_continuity_flags, 20),
                new_high_gap_10d=self._distance_to_window_high_guarded(history_closes, close_continuity_flags, 10),
                new_low_gap_20d=self._distance_to_window_low_guarded(history_closes, close_continuity_flags, 20),
                amount_ratio_1d_5d=self._ratio_of_window_means(history_amounts, 1, 5),
                amount_ratio_3d_10d=self._ratio_of_window_means(history_amounts, 3, 10),
                turnover_rate_change_1d=self._change_vs_window_average(basic_map, available_dates, idx, "turnover_rate", 1),
                turnover_rate_change_5d=self._change_vs_window_average(basic_map, available_dates, idx, "turnover_rate", 5),
                recent_runup_5d=recent_runup_5d,
                turnover_spike_ratio=turnover_spike_ratio,
                high_position_flag=high_position_flag,
                high_position_acceleration_flag=high_position_acceleration_flag,
                weak_market_high_position_flag=weak_market_high_position_flag,
                open_gap_pct=open_gap_pct,
                open_gap_signed_pct=intraday_features.get("open_gap_signed_pct"),
                intraday_return=intraday_features.get("intraday_return"),
                amplitude=intraday_features.get("amplitude"),
                close_position_in_day=intraday_features.get("close_position_in_day"),
                upper_shadow_pct=intraday_features.get("upper_shadow_pct"),
                lower_shadow_pct=intraday_features.get("lower_shadow_pct"),
                close_to_high=intraday_features.get("close_to_high"),
                close_to_low=intraday_features.get("close_to_low"),
                missing_window_flag=missing_window_flag,
                missing_feature_count=missing_feature_count,
                prev_day_limit_up=prev_day_limit_up,
                prev_day_limit_open_times=prev_day_limit_open_times,
                prev_day_limit_first_time=prev_day_limit_first_time,
                prev_day_limit_last_time=prev_day_limit_last_time,
                prev_day_limit_amount=prev_day_limit_amount,
                prev_day_fd_amount=prev_day_fd_amount,
                prev_day_limit_times=prev_day_limit_times,
                prev_day_up_stat_success=prev_day_up_stat_success,
                prev_day_up_stat_total=prev_day_up_stat_total,
                prev_day_up_stat_ratio=prev_day_up_stat_ratio,
                prev_day_one_word_limit_flag=prev_day_one_word_limit_flag,
                moneyflow_net_1d=moneyflow_features.get("moneyflow_net_1d"),
                moneyflow_large_net_1d=moneyflow_features.get("moneyflow_large_net_1d"),
                moneyflow_elarge_net_1d=moneyflow_features.get("moneyflow_elarge_net_1d"),
                moneyflow_net_3d=moneyflow_features.get("moneyflow_net_3d"),
                moneyflow_large_net_3d=moneyflow_features.get("moneyflow_large_net_3d"),
                moneyflow_elarge_net_3d=moneyflow_features.get("moneyflow_elarge_net_3d"),
                moneyflow_positive_flag=moneyflow_features.get("moneyflow_positive_flag"),
                limit_like_moneyflow_divergence_flag=limit_like_moneyflow_divergence_flag,
                limit_chase_failure_risk_score=limit_chase_failure_risk_score,
                return_1d=future_return_1d,
                return_3d=future_return_3d,
                return_5d=future_return_5d,
                return_10d=future_return_10d,
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
            if prev_day_limit_up:
                sample.label_limit_relay_success_1d = None if sample.return_1d is None else sample.return_1d > 0
                sample.label_limit_relay_strong_1d = None if sample.vs_market_1d is None else sample.vs_market_1d >= 0.02
                sample.label_limit_relay_success_3d = None if sample.return_3d is None else sample.return_3d > 0
                sample.label_limit_relay_limit_up_1d = None if future_next_pct_change is None else future_next_pct_change >= 9.5
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
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_limit_time_to_minutes(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        text = str(value).strip().split(".")[0]
        if not text or not text.isdigit():
            return None
        text = text.zfill(6)
        try:
            hour = int(text[:2])
            minute = int(text[2:4])
            second = int(text[4:6]) if len(text) >= 6 else 0
        except ValueError:
            return None
        return hour * 60.0 + minute + second / 60.0

    @staticmethod
    def _parse_up_stat(value: Any) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        if value in (None, ""):
            return None, None, None
        text = str(value).strip()
        if "/" not in text:
            return None, None, None
        left, right = text.split("/", 1)
        try:
            success = int(left.strip())
            total = int(right.strip())
        except ValueError:
            return None, None, None
        ratio = (success / total) if total else None
        return success, total, ratio

    @classmethod
    def _is_limit_up_row(cls, limit_row: Dict[str, Any], pct_change: Optional[float]) -> Optional[bool]:
        limit_value = str(limit_row.get("limit") or "").strip().upper()
        if limit_value == "U":
            return True
        if limit_value == "D":
            return False
        limit_pct = cls._safe_float(limit_row.get("pct_chg"))
        if limit_pct is not None:
            return limit_pct >= 9.5
        if pct_change is None:
            return None
        return pct_change >= 9.5

    @classmethod
    def _is_one_word_limit(
        cls,
        *,
        limit_up: Optional[bool],
        open_times: Optional[int],
        daily_row: Dict[str, Any],
    ) -> Optional[bool]:
        if not limit_up:
            return False if limit_up is not None else None
        high = cls._safe_float(daily_row.get("high"))
        low = cls._safe_float(daily_row.get("low"))
        close = cls._safe_float(daily_row.get("close"))
        range_pct = None
        if high not in (None, 0) and low not in (None, 0) and close not in (None, 0):
            range_pct = (high - low) / close * 100.0
        return bool((open_times is not None and open_times <= 0) or (range_pct is not None and range_pct <= 0.35))

    @classmethod
    def _moneyflow_net(cls, row: Dict[str, Any], buy_key: str, sell_key: str) -> Optional[float]:
        if not isinstance(row, dict) or not row:
            return None
        buy_value = cls._safe_float(row.get(buy_key)) or 0.0
        sell_value = cls._safe_float(row.get(sell_key)) or 0.0
        return buy_value - sell_value

    @classmethod
    def _build_moneyflow_features(
        cls,
        moneyflow_map: Dict[str, Dict[str, Any]],
        available_dates: List[str],
        idx: int,
    ) -> Dict[str, Optional[float]]:
        trade_date = available_dates[idx]
        today = moneyflow_map.get(trade_date, {})
        net_1d = cls._safe_float(today.get("net_mf_amount")) if isinstance(today, dict) else None
        large_1d = cls._moneyflow_net(today, "buy_lg_amount", "sell_lg_amount")
        elarge_1d = cls._moneyflow_net(today, "buy_elg_amount", "sell_elg_amount")
        net_values: List[float] = []
        large_values: List[float] = []
        elarge_values: List[float] = []
        for offset in range(3):
            pos = idx - offset
            if pos < 0:
                continue
            row = moneyflow_map.get(available_dates[pos], {})
            if not isinstance(row, dict) or not row:
                continue
            net_value = cls._safe_float(row.get("net_mf_amount"))
            large_value = cls._moneyflow_net(row, "buy_lg_amount", "sell_lg_amount")
            elarge_value = cls._moneyflow_net(row, "buy_elg_amount", "sell_elg_amount")
            if net_value is not None:
                net_values.append(net_value)
            if large_value is not None:
                large_values.append(large_value)
            if elarge_value is not None:
                elarge_values.append(elarge_value)
        net_3d = sum(net_values) if net_values else None
        large_3d = sum(large_values) if large_values else None
        elarge_3d = sum(elarge_values) if elarge_values else None
        return {
            "moneyflow_net_1d": net_1d,
            "moneyflow_large_net_1d": large_1d,
            "moneyflow_elarge_net_1d": elarge_1d,
            "moneyflow_net_3d": net_3d,
            "moneyflow_large_net_3d": large_3d,
            "moneyflow_elarge_net_3d": elarge_3d,
            "moneyflow_positive_flag": None if net_3d is None else (1.0 if net_3d > 0 else 0.0),
        }

    @staticmethod
    def _limit_chase_failure_risk_score(
        *,
        pct_change: Optional[float],
        prev_day_limit_up: Optional[bool],
        moneyflow_net_3d: Optional[float],
        moneyflow_large_net_3d: Optional[float],
        moneyflow_elarge_net_3d: Optional[float],
        price_position_20d: Optional[float],
        turnover_spike_ratio: Optional[float],
        weak_market_high_position_flag: Optional[bool],
    ) -> float:
        score = 0.0
        if pct_change is not None and pct_change >= 9.5:
            score += 1.0
        if prev_day_limit_up:
            score += 1.0
        if moneyflow_net_3d is not None and moneyflow_net_3d <= 0:
            score += 1.0
        if (
            (moneyflow_large_net_3d is not None and moneyflow_large_net_3d <= 0)
            or (moneyflow_elarge_net_3d is not None and moneyflow_elarge_net_3d <= 0)
        ):
            score += 1.0
        if price_position_20d is not None and price_position_20d >= 0.88:
            score += 1.0
        if turnover_spike_ratio is not None and turnover_spike_ratio >= 1.6:
            score += 1.0
        if weak_market_high_position_flag:
            score += 1.0
        return score

    @staticmethod
    def _compute_intraday_features(
        *,
        open_price: Optional[float],
        high_price: Optional[float],
        low_price: Optional[float],
        close_price: Optional[float],
        pre_close: Optional[float],
    ) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {
            "open_gap_pct": None,
            "open_gap_signed_pct": None,
            "intraday_return": None,
            "amplitude": None,
            "close_position_in_day": None,
            "upper_shadow_pct": None,
            "lower_shadow_pct": None,
            "close_to_high": None,
            "close_to_low": None,
        }
        if close_price in (None, 0):
            return result
        if open_price not in (None, 0) and pre_close not in (None, 0):
            open_gap_signed_pct = (float(open_price) - float(pre_close)) / float(pre_close)
            result["open_gap_signed_pct"] = open_gap_signed_pct
            result["open_gap_pct"] = abs(open_gap_signed_pct)
        if open_price not in (None, 0):
            result["intraday_return"] = (float(close_price) - float(open_price)) / float(open_price)
        if high_price not in (None, 0):
            result["close_to_high"] = float(close_price) / float(high_price) - 1.0
        if low_price not in (None, 0):
            result["close_to_low"] = float(close_price) / float(low_price) - 1.0
        if high_price not in (None, 0) and low_price not in (None, 0):
            day_range = float(high_price) - float(low_price)
            result["amplitude"] = day_range / float(close_price)
            if day_range > 0:
                result["close_position_in_day"] = (float(close_price) - float(low_price)) / day_range
                if open_price not in (None, 0):
                    upper_shadow = float(high_price) - max(float(open_price), float(close_price))
                    lower_shadow = min(float(open_price), float(close_price)) - float(low_price)
                    result["upper_shadow_pct"] = max(0.0, upper_shadow) / float(close_price)
                    result["lower_shadow_pct"] = max(0.0, lower_shadow) / float(close_price)
        return result

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
    def _window_compound_return(cls, pct_changes: List[Optional[float]], window: int) -> Optional[float]:
        if len(pct_changes) < window:
            return None
        values = pct_changes[-window:]
        if any(value is None for value in values):
            return None
        compounded = 1.0
        for value in values:
            compounded *= 1.0 + (float(value) / 100.0)
        return compounded - 1.0

    @classmethod
    def _future_compound_return(cls, pct_changes: List[Optional[float]], idx: int, offset: int) -> Optional[float]:
        if idx + offset >= len(pct_changes):
            return None
        values = pct_changes[idx + 1:idx + offset + 1]
        if len(values) < offset or any(value is None for value in values):
            return None
        compounded = 1.0
        for value in values:
            compounded *= 1.0 + (float(value) / 100.0)
        return compounded - 1.0

    @classmethod
    def _build_close_continuity_flags(
        cls,
        daily_map: Dict[str, Dict[str, Any]],
        available_dates: List[str],
        *,
        threshold: float = 0.05,
    ) -> List[bool]:
        flags: List[bool] = [True]
        for prev_date, current_date in zip(available_dates[:-1], available_dates[1:]):
            prev_close = cls._safe_float(daily_map.get(prev_date, {}).get("close"))
            pre_close = cls._safe_float(daily_map.get(current_date, {}).get("pre_close"))
            if prev_close in (None, 0) or pre_close in (None, 0):
                flags.append(False)
                continue
            flags.append(abs(prev_close / pre_close - 1.0) <= threshold)
        return flags

    @classmethod
    def _has_adjusted_close_coverage(
        cls,
        adj_factor_map: Dict[str, float],
        available_dates: List[str],
    ) -> bool:
        factors = [cls._safe_float(adj_factor_map.get(value)) for value in available_dates]
        valid_factors = [factor for factor in factors if factor not in (None, 0)]
        return len(valid_factors) >= max(3, int(len(available_dates) * 0.6))

    @classmethod
    def _build_forward_adjusted_closes(
        cls,
        closes: List[Optional[float]],
        available_dates: List[str],
        adj_factor_map: Dict[str, float],
    ) -> List[Optional[float]]:
        """Build a stable historical price series for window features.

        Tushare daily OHLC is normally unadjusted. Instead of trying to infer whether
        the stored daily table is already adjusted, this method only applies an
        adjustment when valid adj_factor coverage is present. Features use adjusted
        closes for historical windows, while entry_price/close and return labels keep
        the raw tradable price and pct_chg fields.
        """
        factors = [cls._safe_float(adj_factor_map.get(value)) for value in available_dates]
        valid_factors = [factor for factor in factors if factor not in (None, 0)]
        if not cls._has_adjusted_close_coverage(adj_factor_map, available_dates):
            return closes
        latest_factor = valid_factors[-1]
        if latest_factor in (None, 0):
            return closes
        adjusted: List[Optional[float]] = []
        for close, factor in zip(closes, factors):
            if close is None or factor in (None, 0):
                adjusted.append(None)
                continue
            adjusted.append(float(close) * float(factor) / float(latest_factor))
        return adjusted

    @classmethod
    def _window_has_close_discontinuity(cls, continuity_flags: List[bool], window: int) -> bool:
        if len(continuity_flags) < window:
            return True
        return not all(continuity_flags[-window + 1:])

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
    def _window_volatility_guarded(
        cls,
        closes: List[Optional[float]],
        continuity_flags: List[bool],
        window: int,
    ) -> Optional[float]:
        if cls._window_has_close_discontinuity(continuity_flags, window):
            return None
        return cls._window_volatility(closes, window)

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
    def _window_max_drawdown_guarded(
        cls,
        closes: List[Optional[float]],
        continuity_flags: List[bool],
        window: int,
    ) -> Optional[float]:
        if cls._window_has_close_discontinuity(continuity_flags, window):
            return None
        return cls._window_max_drawdown(closes, window)

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
    def _close_to_ma_guarded(
        cls,
        closes: List[Optional[float]],
        continuity_flags: List[bool],
        window: int,
    ) -> Optional[float]:
        if cls._window_has_close_discontinuity(continuity_flags, window):
            return None
        return cls._close_to_ma(closes, window)

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
    def _price_position_guarded(
        cls,
        closes: List[Optional[float]],
        continuity_flags: List[bool],
        window: int,
    ) -> Optional[float]:
        if cls._window_has_close_discontinuity(continuity_flags, window):
            return None
        return cls._price_position(closes, window)

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
    def _up_days_from_pct_changes(cls, pct_changes: List[Optional[float]], window: int) -> Optional[int]:
        if len(pct_changes) < window:
            return None
        values = pct_changes[-window:]
        if any(value is None for value in values):
            return None
        return sum(1 for value in values if value is not None and float(value) > 0)

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
    def _distance_to_window_high_guarded(
        cls,
        closes: List[Optional[float]],
        continuity_flags: List[bool],
        window: int,
    ) -> Optional[float]:
        if cls._window_has_close_discontinuity(continuity_flags, window):
            return None
        return cls._distance_to_window_high(closes, window)

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
    def _distance_to_window_low_guarded(
        cls,
        closes: List[Optional[float]],
        continuity_flags: List[bool],
        window: int,
    ) -> Optional[float]:
        if cls._window_has_close_discontinuity(continuity_flags, window):
            return None
        return cls._distance_to_window_low(closes, window)

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

    @classmethod
    def _recent_runup_pct(cls, pct_changes: List[Optional[float]], window: int) -> Optional[float]:
        if len(pct_changes) < window:
            return None
        values = pct_changes[-window:]
        if any(value is None for value in values):
            return None
        return float(sum(values))

    @classmethod
    def _turnover_spike_ratio(
        cls,
        basic_map: Dict[str, Dict[str, Any]],
        available_dates: List[str],
        idx: int,
    ) -> Optional[float]:
        current = cls._safe_float(basic_map.get(available_dates[idx], {}).get("turnover_rate"))
        baseline = cls._rolling_basic_average(basic_map, available_dates, idx, "turnover_rate", 5)
        if current is None or baseline in (None, 0):
            return None
        return current / baseline

    @staticmethod
    def _weak_market_flag(market_row: Dict[str, Optional[float]]) -> Optional[bool]:
        if not isinstance(market_row, dict) or not market_row:
            return None
        market_return_1d = RawMarketTrainingDatasetBuilder._safe_float(market_row.get("market_return_1d"))
        market_return_3d = RawMarketTrainingDatasetBuilder._safe_float(market_row.get("market_return_3d"))
        market_up_ratio_1d = RawMarketTrainingDatasetBuilder._safe_float(market_row.get("market_up_ratio_1d"))
        market_up_ratio_3d_avg = RawMarketTrainingDatasetBuilder._safe_float(market_row.get("market_up_ratio_3d_avg"))
        market_up_days_5d = market_row.get("market_up_days_5d")
        has_any_signal = any(
            value is not None
            for value in [
                market_return_1d,
                market_return_3d,
                market_up_ratio_1d,
                market_up_ratio_3d_avg,
                market_up_days_5d,
            ]
        )
        if not has_any_signal:
            return None
        return bool(
            (market_return_1d is not None and market_return_1d < 0)
            or (market_return_3d is not None and market_return_3d < 0)
            or (market_up_ratio_1d is not None and market_up_ratio_1d < 0.45)
            or (market_up_ratio_3d_avg is not None and market_up_ratio_3d_avg < 0.45)
            or (market_up_days_5d is not None and int(market_up_days_5d) <= 2)
        )

    @staticmethod
    def _high_position_acceleration_flag(
        *,
        price_position: Optional[float],
        pct_change: Optional[float],
        volume_ratio: Optional[float],
        turnover_rate: Optional[float],
        recent_runup_5d: Optional[float],
        turnover_spike_ratio: Optional[float],
    ) -> Optional[bool]:
        if price_position is None:
            return None
        high_position = price_position >= 0.88
        same_day_acceleration = (
            pct_change is not None
            and volume_ratio is not None
            and turnover_rate is not None
            and pct_change >= 5.0
            and volume_ratio >= 2.5
            and turnover_rate >= 8.0
        )
        runup_acceleration = (
            recent_runup_5d is not None
            and volume_ratio is not None
            and recent_runup_5d >= 10.0
            and volume_ratio >= 2.5
        )
        turnover_acceleration = (
            recent_runup_5d is not None
            and turnover_spike_ratio is not None
            and recent_runup_5d >= 10.0
            and turnover_spike_ratio >= 1.6
        )
        return bool(high_position and (same_day_acceleration or runup_acceleration or turnover_acceleration))

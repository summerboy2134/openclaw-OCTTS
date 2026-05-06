from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import and_, func

from octts.models.screening_models import (
    DatabaseManager,
    MarketAdjFactor,
    MarketDaily,
    MarketDailyBasic,
    MarketIndustryMoneyflowDaily,
    MarketLimitListDaily,
    MarketMoneyflowDaily,
    MarketMoneyflowMarketDaily,
    MarketTopListDaily,
    MarketTradeCalendar,
)


class MarketRawDataRepository:
    def __init__(self, database_url: str) -> None:
        self._db = DatabaseManager(database_url)

    def list_trading_dates(self, *, start_date: str, end_date: str, exchange: str = "SSE") -> List[str]:
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketTradeCalendar)
                .filter(
                    MarketTradeCalendar.exchange == exchange,
                    MarketTradeCalendar.trade_date >= start,
                    MarketTradeCalendar.trade_date <= end,
                    MarketTradeCalendar.is_open.is_(True),
                )
                .order_by(MarketTradeCalendar.trade_date.asc())
                .all()
            )
            return [row.trade_date.strftime("%Y%m%d") for row in rows if row.trade_date is not None]
        finally:
            session.close()

    def get_daily(self, *, ts_code: str, trade_date: str) -> Optional[Dict[str, Any]]:
        target_date = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            row = (
                session.query(MarketDaily)
                .filter(MarketDaily.ts_code == ts_code, MarketDaily.trade_date == target_date)
                .first()
            )
            return self._serialize_market_daily(row) if row is not None else None
        finally:
            session.close()

    def get_daily_basic(self, *, ts_code: str, trade_date: str) -> Optional[Dict[str, Any]]:
        target_date = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            row = (
                session.query(MarketDailyBasic)
                .filter(MarketDailyBasic.ts_code == ts_code, MarketDailyBasic.trade_date == target_date)
                .first()
            )
            return self._serialize_market_daily_basic(row) if row is not None else None
        finally:
            session.close()

    def get_adj_factor(self, *, ts_code: str, trade_date: str) -> Optional[float]:
        target_date = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            row = (
                session.query(MarketAdjFactor)
                .filter(MarketAdjFactor.ts_code == ts_code, MarketAdjFactor.trade_date == target_date)
                .first()
            )
            return float(row.adj_factor) if row is not None and row.adj_factor is not None else None
        finally:
            session.close()

    def get_daily_range(self, *, ts_code: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketDaily)
                .filter(
                    MarketDaily.ts_code == ts_code,
                    MarketDaily.trade_date >= start,
                    MarketDaily.trade_date <= end,
                )
                .order_by(MarketDaily.trade_date.desc())
                .all()
            )
            return [self._serialize_market_daily(row) for row in rows]
        finally:
            session.close()

    def get_daily_by_trade_dates(
        self,
        *,
        ts_codes: Iterable[str],
        trading_dates: Iterable[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        ts_code_list = list(dict.fromkeys(ts_codes))
        trade_date_list = [self._parse_date(value) for value in dict.fromkeys(trading_dates)]
        if not ts_code_list or not trade_date_list:
            return {}
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketDaily)
                .filter(MarketDaily.ts_code.in_(ts_code_list), MarketDaily.trade_date.in_(trade_date_list))
                .all()
            )
            result: Dict[str, Dict[str, Dict[str, Any]]] = {ts_code: {} for ts_code in ts_code_list}
            for row in rows:
                result.setdefault(row.ts_code, {})[row.trade_date.strftime("%Y%m%d")] = self._serialize_market_daily(row)
            return result
        finally:
            session.close()

    def get_daily_basic_by_trade_dates(
        self,
        *,
        ts_codes: Iterable[str],
        trading_dates: Iterable[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        ts_code_list = list(dict.fromkeys(ts_codes))
        trade_date_list = [self._parse_date(value) for value in dict.fromkeys(trading_dates)]
        if not ts_code_list or not trade_date_list:
            return {}
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketDailyBasic)
                .filter(MarketDailyBasic.ts_code.in_(ts_code_list), MarketDailyBasic.trade_date.in_(trade_date_list))
                .all()
            )
            result: Dict[str, Dict[str, Dict[str, Any]]] = {ts_code: {} for ts_code in ts_code_list}
            for row in rows:
                result.setdefault(row.ts_code, {})[row.trade_date.strftime("%Y%m%d")] = self._serialize_market_daily_basic(row)
            return result
        finally:
            session.close()

    def get_adj_factors_by_trade_dates(
        self,
        *,
        ts_codes: Iterable[str],
        trading_dates: Iterable[str],
    ) -> Dict[str, Dict[str, float]]:
        ts_code_list = list(dict.fromkeys(ts_codes))
        trade_date_list = [self._parse_date(value) for value in dict.fromkeys(trading_dates)]
        if not ts_code_list or not trade_date_list:
            return {}
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketAdjFactor)
                .filter(MarketAdjFactor.ts_code.in_(ts_code_list), MarketAdjFactor.trade_date.in_(trade_date_list))
                .all()
            )
            result: Dict[str, Dict[str, float]] = {ts_code: {} for ts_code in ts_code_list}
            for row in rows:
                if row.adj_factor is None:
                    continue
                result.setdefault(row.ts_code, {})[row.trade_date.strftime("%Y%m%d")] = float(row.adj_factor)
            return result
        finally:
            session.close()

    def get_daily_batch_for_trade_date(self, *, ts_codes: List[str], trade_date: str) -> Dict[str, List[Dict[str, Any]]]:
        trade_date_value = self._parse_date(trade_date)
        if not ts_codes:
            return {}
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketDaily)
                .filter(MarketDaily.ts_code.in_(ts_codes), MarketDaily.trade_date == trade_date_value)
                .all()
            )
            result: Dict[str, List[Dict[str, Any]]] = {ts_code: [] for ts_code in ts_codes}
            for row in rows:
                result[row.ts_code] = [self._serialize_market_daily(row)]
            return result
        finally:
            session.close()

    def get_daily_basic_batch_for_trade_date(self, *, ts_codes: List[str], trade_date: str) -> Dict[str, Dict[str, Any]]:
        trade_date_value = self._parse_date(trade_date)
        if not ts_codes:
            return {}
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketDailyBasic)
                .filter(MarketDailyBasic.ts_code.in_(ts_codes), MarketDailyBasic.trade_date == trade_date_value)
                .all()
            )
            result: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                result[row.ts_code] = self._serialize_market_daily_basic(row)
            return result
        finally:
            session.close()

    def get_limit_list_by_trade_date(self, *, ts_codes: Iterable[str], trade_date: str) -> Dict[str, Dict[str, Any]]:
        ts_code_list = list(dict.fromkeys(ts_codes))
        if not ts_code_list:
            return {}
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketLimitListDaily)
                .filter(MarketLimitListDaily.ts_code.in_(ts_code_list), MarketLimitListDaily.trade_date == trade_date_value)
                .all()
            )
            return {row.ts_code: self._serialize_market_limit_list_daily(row) for row in rows}
        finally:
            session.close()

    def get_top_list_by_trade_date(self, *, ts_codes: Iterable[str], trade_date: str) -> Dict[str, List[Dict[str, Any]]]:
        ts_code_list = list(dict.fromkeys(ts_codes))
        if not ts_code_list:
            return {}
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketTopListDaily)
                .filter(MarketTopListDaily.ts_code.in_(ts_code_list), MarketTopListDaily.trade_date == trade_date_value)
                .all()
            )
            result: Dict[str, List[Dict[str, Any]]] = {ts_code: [] for ts_code in ts_code_list}
            for row in rows:
                result.setdefault(row.ts_code, []).append(self._serialize_market_top_list_daily(row))
            return result
        finally:
            session.close()

    def get_industry_moneyflow_by_trade_date(self, *, industries: Iterable[str], trade_date: str) -> Dict[str, Dict[str, Any]]:
        industry_list = [str(value).strip() for value in dict.fromkeys(industries) if str(value).strip()]
        if not industry_list:
            return {}
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketIndustryMoneyflowDaily)
                .filter(MarketIndustryMoneyflowDaily.industry.in_(industry_list), MarketIndustryMoneyflowDaily.trade_date == trade_date_value)
                .all()
            )
            return {str(row.industry): self._serialize_market_industry_moneyflow_daily(row) for row in rows if row.industry}
        finally:
            session.close()

    def get_market_moneyflow(self, *, trade_date: str) -> Optional[Dict[str, Any]]:
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            row = (
                session.query(MarketMoneyflowMarketDaily)
                .filter(MarketMoneyflowMarketDaily.trade_date == trade_date_value)
                .first()
            )
            return self._serialize_market_moneyflow_market_daily(row) if row is not None else None
        finally:
            session.close()

    def get_market_daily_summary(self, *, trade_date: str) -> Dict[str, Any]:
        """获取市场日线汇总统计：涨跌家数、平均涨跌幅、成交额等。"""
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            rows = session.query(MarketDaily).filter(MarketDaily.trade_date == trade_date_value).all()
            if not rows:
                return {"pct_count": 0, "rise_count": 0, "fall_count": 0, "flat_count": 0}
            rise_count = 0
            fall_count = 0
            flat_count = 0
            total_pct_chg = 0.0
            total_amount = 0.0
            limit_up_estimate = 0
            limit_down_estimate = 0
            for row in rows:
                pct_chg = float(row.pct_chg or 0.0)
                if pct_chg > 0.05:
                    rise_count += 1
                elif pct_chg < -0.05:
                    fall_count += 1
                else:
                    flat_count += 1
                total_pct_chg += pct_chg
                total_amount += float(row.amount or 0.0)
                if pct_chg >= 9.8:
                    limit_up_estimate += 1
                elif pct_chg <= -9.8:
                    limit_down_estimate += 1
            pct_count = len(rows)
            avg_pct_chg = total_pct_chg / pct_count if pct_count else 0.0
            return {
                "trade_date": trade_date,
                "pct_count": pct_count,
                "rise_count": rise_count,
                "fall_count": fall_count,
                "flat_count": flat_count,
                "avg_pct_chg": round(avg_pct_chg, 4),
                "total_amount": round(total_amount, 2),
                "limit_up_estimate": limit_up_estimate,
                "limit_down_estimate": limit_down_estimate,
            }
        finally:
            session.close()

    def get_market_limit_summary(self, *, trade_date: str) -> Dict[str, Any]:
        """获取涨跌停汇总统计。"""
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            limit_up_rows = (
                session.query(MarketLimitListDaily)
                .filter(MarketLimitListDaily.trade_date == trade_date_value, MarketLimitListDaily.limit == "U")
                .all()
            )
            limit_down_rows = (
                session.query(MarketLimitListDaily)
                .filter(MarketLimitListDaily.trade_date == trade_date_value, MarketLimitListDaily.limit == "D")
                .all()
            )
            return {
                "trade_date": trade_date,
                "limit_up": len(limit_up_rows),
                "limit_down": len(limit_down_rows),
                "total_count": len(limit_up_rows) + len(limit_down_rows),
            }
        finally:
            session.close()

    def count_rows_for_trade_date(self, *, model, trade_date: str) -> int:
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            return int(session.query(func.count()).select_from(model).filter(model.trade_date == trade_date_value).scalar() or 0)
        finally:
            session.close()

    def summarize_market_data_coverage(
        self,
        *,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
    ) -> Dict[str, Any]:
        trading_dates = self.list_trading_dates(start_date=start_date, end_date=end_date, exchange=exchange)
        datasets = {
            "daily": MarketDaily,
            "daily_basic": MarketDailyBasic,
            "adj_factor": MarketAdjFactor,
            "moneyflow": MarketMoneyflowDaily,
            "top_list": MarketTopListDaily,
            "limit_list": MarketLimitListDaily,
            "industry_moneyflow": MarketIndustryMoneyflowDaily,
            "market_moneyflow": MarketMoneyflowMarketDaily,
        }
        summary: Dict[str, Any] = {
            "start_date": self._parse_date(start_date).isoformat(),
            "end_date": self._parse_date(end_date).isoformat(),
            "trading_days": len(trading_dates),
            "trading_dates": trading_dates,
            "datasets": {},
        }
        session = self._db.get_session()
        try:
            parsed_dates = [self._parse_date(value) for value in trading_dates]
            for name, model in datasets.items():
                if not parsed_dates:
                    summary["datasets"][name] = {
                        "covered_trade_days": 0,
                        "missing_trade_days": 0,
                        "missing_dates": [],
                        "row_counts": {},
                        "min_rows": 0,
                        "max_rows": 0,
                    }
                    continue
                rows = (
                    session.query(model.trade_date, func.count().label("row_count"))
                    .filter(model.trade_date.in_(parsed_dates))
                    .group_by(model.trade_date)
                    .all()
                )
                row_counts = {
                    trade_date.strftime("%Y%m%d"): int(row_count or 0)
                    for trade_date, row_count in rows
                    if trade_date is not None
                }
                missing_dates = [trade_date for trade_date in trading_dates if row_counts.get(trade_date, 0) <= 0]
                counts = list(row_counts.values())
                summary["datasets"][name] = {
                    "covered_trade_days": len(trading_dates) - len(missing_dates),
                    "missing_trade_days": len(missing_dates),
                    "missing_dates": missing_dates,
                    "row_counts": row_counts,
                    "min_rows": min(counts) if counts else 0,
                    "max_rows": max(counts) if counts else 0,
                }
            return summary
        finally:
            session.close()

    def get_moneyflow_summaries_by_trade_date(
        self,
        *,
        ts_codes: Iterable[str],
        trade_date: str,
        lookback_days: int = 3,
    ) -> Dict[str, Dict[str, Any]]:
        ts_code_list = list(dict.fromkeys(ts_codes))
        if not ts_code_list:
            return {}
        trade_date_value = self._parse_date(trade_date)
        session = self._db.get_session()
        try:
            rows = (
                session.query(MarketMoneyflowDaily)
                .filter(MarketMoneyflowDaily.ts_code.in_(ts_code_list), MarketMoneyflowDaily.trade_date <= trade_date_value)
                .order_by(MarketMoneyflowDaily.ts_code.asc(), MarketMoneyflowDaily.trade_date.desc())
                .all()
            )
            grouped: Dict[str, List[MarketMoneyflowDaily]] = {ts_code: [] for ts_code in ts_code_list}
            for row in rows:
                bucket = grouped.setdefault(row.ts_code, [])
                if len(bucket) >= lookback_days:
                    continue
                bucket.append(row)

            summaries: Dict[str, Dict[str, Any]] = {}
            for ts_code, items in grouped.items():
                if not items:
                    continue
                recent_3d_net_inflow = sum(float(item.net_mf_amount or 0.0) for item in items)
                recent_large_order_net_inflow = sum(float((item.buy_lg_amount or 0.0) - (item.sell_lg_amount or 0.0)) for item in items)
                recent_super_large_order_net_inflow = sum(float((item.buy_elg_amount or 0.0) - (item.sell_elg_amount or 0.0)) for item in items)
                summaries[ts_code] = {
                    "recent_3d_net_inflow": round(recent_3d_net_inflow, 2),
                    "recent_large_order_net_inflow": round(recent_large_order_net_inflow, 2),
                    "recent_super_large_order_net_inflow": round(recent_super_large_order_net_inflow, 2),
                    "positive_flag": 1.0 if recent_3d_net_inflow > 0 else 0.0,
                    "rows": len(items),
                }
            return summaries
        finally:
            session.close()

    def save_trade_calendar(self, rows: List[Dict[str, Any]], *, exchange: str = "SSE") -> int:
        return self._db.upsert_market_trade_calendar(rows, exchange=exchange, force_refresh=False)

    def save_daily(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_daily(rows, force_refresh=False)

    def save_daily_basic(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_daily_basic(rows, force_refresh=False)

    def save_adj_factor(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_adj_factor(rows, force_refresh=False)

    def save_moneyflow(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_moneyflow_daily(rows, force_refresh=False)

    def save_top_list(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_top_list_daily(rows, force_refresh=False)

    def save_limit_list(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_limit_list_daily(rows, force_refresh=False)

    def save_industry_moneyflow(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_industry_moneyflow_daily(rows, force_refresh=False)

    def save_market_moneyflow(self, rows: List[Dict[str, Any]]) -> int:
        return self._db.upsert_market_moneyflow_market_daily(rows, force_refresh=False)

    def get_calendar_dates(self, *, start_date: str, end_date: str, exchange: str = "SSE") -> List[str]:
        return self.list_trading_dates(start_date=start_date, end_date=end_date, exchange=exchange)

    def has_full_trade_calendar_range(self, *, start_date: str, end_date: str, exchange: str = "SSE") -> bool:
        requested_dates = _calendar_day_strings(start_date, end_date)
        existing_dates = set(self.get_calendar_dates(start_date=start_date, end_date=end_date, exchange=exchange))
        return all(day in existing_dates for day in requested_dates)

    @staticmethod
    def _parse_date(value: str) -> date:
        text = str(value).strip()
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Invalid trade date: {value}")

    @staticmethod
    def _serialize_market_daily(row: MarketDaily) -> Dict[str, Any]:
        return {
            "ts_code": row.ts_code,
            "trade_date": row.trade_date.strftime("%Y%m%d") if row.trade_date else None,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "pre_close": row.pre_close,
            "change": row.change,
            "pct_chg": row.pct_chg,
            "vol": row.vol,
            "amount": row.amount,
        }

    @staticmethod
    def _serialize_market_daily_basic(row: MarketDailyBasic) -> Dict[str, Any]:
        return {
            "ts_code": row.ts_code,
            "trade_date": row.trade_date.strftime("%Y%m%d") if row.trade_date else None,
            "close": row.close,
            "turnover_rate": row.turnover_rate,
            "turnover_rate_f": row.turnover_rate_f,
            "volume_ratio": row.volume_ratio,
            "pe": row.pe,
            "pe_ttm": row.pe_ttm,
            "pb": row.pb,
            "ps": row.ps,
            "ps_ttm": row.ps_ttm,
            "dv_ratio": row.dv_ratio,
            "dv_ttm": row.dv_ttm,
            "total_share": row.total_share,
            "float_share": row.float_share,
            "free_share": row.free_share,
            "total_mv": row.total_mv,
            "circ_mv": row.circ_mv,
        }

    @staticmethod
    def _serialize_market_limit_list_daily(row: MarketLimitListDaily) -> Dict[str, Any]:
        return {
            "trade_date": row.trade_date.strftime("%Y%m%d") if row.trade_date else None,
            "ts_code": row.ts_code,
            "industry": row.industry,
            "name": row.name,
            "close": row.close,
            "pct_chg": row.pct_chg,
            "amount": row.amount,
            "limit_amount": row.limit_amount,
            "float_mv": row.float_mv,
            "total_mv": row.total_mv,
            "turnover_ratio": row.turnover_ratio,
            "fd_amount": row.fd_amount,
            "first_time": row.first_time,
            "last_time": row.last_time,
            "open_times": row.open_times,
            "up_stat": row.up_stat,
            "limit_times": row.limit_times,
            "limit": row.limit,
        }

    @staticmethod
    def _serialize_market_top_list_daily(row: MarketTopListDaily) -> Dict[str, Any]:
        return {
            "trade_date": row.trade_date.strftime("%Y%m%d") if row.trade_date else None,
            "ts_code": row.ts_code,
            "name": row.name,
            "close": row.close,
            "pct_change": row.pct_change,
            "turnover_rate": row.turnover_rate,
            "amount": row.amount,
            "l_sell": row.l_sell,
            "l_buy": row.l_buy,
            "l_amount": row.l_amount,
            "net_amount": row.net_amount,
            "net_rate": row.net_rate,
            "amount_rate": row.amount_rate,
            "float_values": row.float_values,
            "reason": row.reason,
        }

    @staticmethod
    def _serialize_market_industry_moneyflow_daily(row: MarketIndustryMoneyflowDaily) -> Dict[str, Any]:
        return {
            "trade_date": row.trade_date.strftime("%Y%m%d") if row.trade_date else None,
            "ts_code": row.ts_code,
            "industry": row.industry,
            "lead_stock": row.lead_stock,
            "close": row.close,
            "pct_change": row.pct_change,
            "company_num": row.company_num,
            "pct_change_stock": row.pct_change_stock,
            "close_price": row.close_price,
            "net_buy_amount": row.net_buy_amount,
            "net_sell_amount": row.net_sell_amount,
            "net_amount": row.net_amount,
        }

    @staticmethod
    def _serialize_market_moneyflow_market_daily(row: MarketMoneyflowMarketDaily) -> Dict[str, Any]:
        return {
            "trade_date": row.trade_date.strftime("%Y%m%d") if row.trade_date else None,
            "close_sh": row.close_sh,
            "pct_change_sh": row.pct_change_sh,
            "close_sz": row.close_sz,
            "pct_change_sz": row.pct_change_sz,
            "net_amount": row.net_amount,
            "net_amount_rate": row.net_amount_rate,
            "buy_elg_amount": row.buy_elg_amount,
            "buy_elg_amount_rate": row.buy_elg_amount_rate,
            "buy_lg_amount": row.buy_lg_amount,
            "buy_lg_amount_rate": row.buy_lg_amount_rate,
            "buy_md_amount": row.buy_md_amount,
            "buy_md_amount_rate": row.buy_md_amount_rate,
            "buy_sm_amount": row.buy_sm_amount,
            "buy_sm_amount_rate": row.buy_sm_amount_rate,
        }


def _calendar_day_strings(start_date: str, end_date: str) -> List[str]:
    start = MarketRawDataRepository._parse_date(start_date)
    end = MarketRawDataRepository._parse_date(end_date)
    values: List[str] = []
    current = start
    while current <= end:
        values.append(current.strftime("%Y%m%d"))
        current = current.fromordinal(current.toordinal() + 1)
    return values

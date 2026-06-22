from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from octts.config import Settings
from octts.models.screening_models import (
    DatabaseManager,
    MarketAdjFactor,
    MarketDaily,
    MarketDailyBasic,
    MarketLimitListDaily,
    MarketTopListDaily,
)
from octts.tools.backfill_market_raw_data import _fetch_rows

logger = logging.getLogger(__name__)

DEFAULT_EXCHANGE = "SSE"
RAW_DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
RAW_DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
    "dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
)
RAW_ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"


class MarketDataSyncService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = DatabaseManager(settings.database_url)
        self._tushare_client = None

    @property
    def tushare_client(self):
        if self._tushare_client is None:
            from octts.clients.tushare_client import TushareClient

            self._tushare_client = TushareClient(self.settings)
        return self._tushare_client

    def ensure_trade_date_data(self, *, trade_date: str) -> Dict[str, Any]:
        normalized_trade_date = self._normalize_trade_date(trade_date)
        result = {
            "trade_date": normalized_trade_date,
            "daily": False,
            "daily_basic": False,
            "adj_factor": False,
            "limit_list": False,
            "top_list": False,
            "fetched": {"daily": 0, "daily_basic": 0, "adj_factor": 0, "limit_list": 0, "top_list": 0},
            "inserted": {"daily": 0, "daily_basic": 0, "adj_factor": 0, "limit_list": 0, "top_list": 0},
        }

        trade_date_value = datetime.strptime(normalized_trade_date, "%Y%m%d").date()
        if not self.db.has_market_data_for_trade_date(model=MarketDaily, trade_date=trade_date_value):
            rows = _fetch_rows(self.tushare_client._pro.daily(trade_date=normalized_trade_date, fields=RAW_DAILY_FIELDS))
            result["fetched"]["daily"] = len(rows)
            result["inserted"]["daily"] = self.db.upsert_market_daily(rows, force_refresh=False)
        result["daily"] = self.db.has_market_data_for_trade_date(model=MarketDaily, trade_date=trade_date_value)

        if not self.db.has_market_data_for_trade_date(model=MarketDailyBasic, trade_date=trade_date_value):
            rows = _fetch_rows(
                self.tushare_client._pro.daily_basic(trade_date=normalized_trade_date, fields=RAW_DAILY_BASIC_FIELDS)
            )
            result["fetched"]["daily_basic"] = len(rows)
            result["inserted"]["daily_basic"] = self.db.upsert_market_daily_basic(rows, force_refresh=False)
        result["daily_basic"] = self.db.has_market_data_for_trade_date(model=MarketDailyBasic, trade_date=trade_date_value)

        if not self.db.has_market_data_for_trade_date(model=MarketAdjFactor, trade_date=trade_date_value):
            rows = _fetch_rows(self.tushare_client._pro.adj_factor(trade_date=normalized_trade_date, fields=RAW_ADJ_FACTOR_FIELDS))
            result["fetched"]["adj_factor"] = len(rows)
            result["inserted"]["adj_factor"] = self.db.upsert_market_adj_factor(rows, force_refresh=False)
        result["adj_factor"] = self.db.has_market_data_for_trade_date(model=MarketAdjFactor, trade_date=trade_date_value)

        # limit_list: relay v2 模型核心数据源（prev_day_limit_up / open_times / up_stat 等）
        if not self.db.has_market_data_for_trade_date(model=MarketLimitListDaily, trade_date=trade_date_value):
            try:
                rows = _fetch_rows(self.tushare_client._pro.limit_list_d(trade_date=normalized_trade_date))
                result["fetched"]["limit_list"] = len(rows)
                result["inserted"]["limit_list"] = self.db.upsert_market_limit_list_daily(rows, force_refresh=False)
            except Exception:
                logger.exception("Failed to fetch limit_list for trade_date=%s", normalized_trade_date)
        result["limit_list"] = self.db.has_market_data_for_trade_date(model=MarketLimitListDaily, trade_date=trade_date_value)

        # top_list: 龙虎榜，报告展示用
        if not self.db.has_market_data_for_trade_date(model=MarketTopListDaily, trade_date=trade_date_value):
            try:
                rows = _fetch_rows(self.tushare_client._pro.top_list(trade_date=normalized_trade_date))
                result["fetched"]["top_list"] = len(rows)
                result["inserted"]["top_list"] = self.db.upsert_market_top_list_daily(rows, force_refresh=False)
            except Exception:
                logger.exception("Failed to fetch top_list for trade_date=%s", normalized_trade_date)
        result["top_list"] = self.db.has_market_data_for_trade_date(model=MarketTopListDaily, trade_date=trade_date_value)

        logger.info("Market data ensure complete: %s", result)
        return result

    @staticmethod
    def _normalize_trade_date(trade_date: str) -> str:
        text = str(trade_date).strip()
        if len(text) == 10 and text[4] == "-" and text[7] == "-":
            return text.replace("-", "")
        if len(text) == 8 and text.isdigit():
            return text
        raise ValueError(f"Invalid trade_date: {trade_date}")

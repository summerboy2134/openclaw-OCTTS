"""Stock screening result storage service."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from octts.config import Settings
from octts.schemas.screener import ScreenResult, TrackedRecommendationState


def _recommendation_pool_sort_key(item: Dict[str, Any]) -> tuple:
    return (
        item.get("recommend_rank") is None,
        int(item.get("recommend_rank") or 9999),
        -float(item.get("recommendation_score") or 0.0),
        -float(item.get("overall_score") or item.get("priority_score") or 0.0),
        item.get("ts_code") or "",
    )


class ScreeningStore:
    """选股结果存储服务"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_path = Path(settings.history_dir_path) / "screening"
        self.pool_base_path = Path(settings.history_dir_path) / "recommendation_pool"
        self._db_manager = None
        if not settings.use_database:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.pool_base_path.mkdir(parents=True, exist_ok=True)

    def _get_db_manager(self):
        if self._db_manager is None:
            from octts.models.screening_models import DatabaseManager

            self._db_manager = DatabaseManager(self.settings.database_url)
        return self._db_manager

    async def save_screening_result(
        self,
        strategy_id: str,
        result: ScreenResult,
        ai_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        保存选股结果

        Args:
            strategy_id: 策略ID
            result: 选股结果
        """
        if self.settings.use_database:
            self._get_db_manager().save_screening_result(
                strategy_id,
                result,
                ai_analysis=ai_analysis,
            )
            return

        # 构建文件路径: screening/strategy_id/YYYYMMDD.json
        strategy_path = self.base_path / strategy_id
        strategy_path.mkdir(parents=True, exist_ok=True)

        today = date.today().strftime("%Y%m%d")
        file_path = strategy_path / f"{today}.json"

        # 准备数据
        data = {
            "screen_id": result.screen_id,
            "strategy_id": strategy_id,
            "screen_date": today,
            "screen_time": result.screen_time.isoformat(),
            "criteria": result.criteria.model_dump(),
            "total_count": result.total_count,
            "execution_time": result.execution_time,
            "stocks": [stock.model_dump() for stock in result.stocks],
        }

        # 保存到文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_screening_history(
        self,
        strategy_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        获取历史选股结果

        Args:
            strategy_id: 策略ID
            days: 获取最近N天的结果

        Returns:
            历史结果列表
        """
        if self.settings.use_database:
            start_date = datetime.now() - timedelta(days=days)
            return self._get_db_manager().get_screening_history(
                strategy_id=strategy_id,
                start_date=start_date,
            )

        strategy_path = self.base_path / strategy_id
        if not strategy_path.exists():
            return []

        results = []
        files = sorted(strategy_path.glob("*.json"), reverse=True)[:days]

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                # 忽略读取错误的文件
                pass

        return results

    def get_stock_performance(
        self,
        ts_code: str,
        days: int = 30,
        strategy_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取股票在选股系统中的历史表现

        Args:
            ts_code: 股票代码
            strategy_id: 策略ID（可选）

        Returns:
            股票表现统计
        """
        if self.settings.use_database:
            return self._get_db_manager().get_stock_performance(ts_code, days)

        appearances = []
        cutoff_date = date.today() - timedelta(days=days)

        # 搜索所有策略或指定策略
        if strategy_id:
            strategy_paths = [self.base_path / strategy_id]
        else:
            strategy_paths = [p for p in self.base_path.iterdir() if p.is_dir()]

        for strategy_path in strategy_paths:
            for file_path in strategy_path.glob("*.json"):
                try:
                    file_date = datetime.strptime(file_path.stem, "%Y%m%d").date()
                except ValueError:
                    continue
                if file_date < cutoff_date:
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # 查找股票
                    for stock in data.get("stocks", []):
                        if stock.get("ts_code") == ts_code:
                            appearances.append({
                                "strategy_id": data.get("strategy_id"),
                                "screen_date": data.get("screen_date"),
                                "stock_data": stock,
                            })
                except Exception:
                    pass

        return {
            "ts_code": ts_code,
            "total_appearances": len(appearances),
            "appearances": appearances,
        }

    def get_screening_result(self, screen_id: str) -> Optional[ScreenResult]:
        """Fetch a screening result by its unique screen id."""
        if self.settings.use_database:
            return self._get_db_manager().get_screening_result(screen_id)

        if not self.base_path.exists():
            return None

        for strategy_path in self.base_path.iterdir():
            if not strategy_path.is_dir():
                continue
            for file_path in sorted(strategy_path.glob("*.json"), reverse=True):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                if data.get("screen_id") != screen_id:
                    continue

                return ScreenResult(
                    screen_id=data["screen_id"],
                    criteria=data.get("criteria", {}),
                    stocks=data.get("stocks", []),
                    total_count=data.get("total_count", 0),
                    screen_time=datetime.fromisoformat(data["screen_time"]),
                    execution_time=data.get("execution_time", 0.0),
                )

        return None

    def get_all_screening_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Return recent history across all strategies."""
        if self.settings.use_database:
            start_date = datetime.now() - timedelta(days=days)
            return self._get_db_manager().get_screening_history(
                strategy_id=None,
                start_date=start_date,
            )

        history = []
        cutoff_date = date.today() - timedelta(days=days)

        for strategy_path in self.base_path.iterdir():
            if not strategy_path.is_dir():
                continue

            for file_path in strategy_path.glob("*.json"):
                try:
                    file_date = datetime.strptime(file_path.stem, "%Y%m%d").date()
                except ValueError:
                    continue
                if file_date < cutoff_date:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                history.append({
                    "strategy_id": strategy_path.name,
                    "data": data,
                })

        history.sort(key=lambda item: item["data"].get("screen_date", ""), reverse=True)
        return history

    def get_previous_recommendation_pool_trade_date(self, trade_date: date) -> Optional[date]:
        if self.settings.use_database:
            return self._get_db_manager().get_previous_recommendation_pool_trade_date(trade_date)

        return self._find_previous_trade_date_from_files(trade_date)

    def _find_previous_trade_date_from_files(self, trade_date: date) -> Optional[date]:
        available_dates = sorted(self._iter_recommendation_pool_dates(), reverse=True)
        for candidate in available_dates:
            if candidate < trade_date:
                return candidate
        return None

    def _iter_recommendation_pool_dates(self) -> Iterable[date]:
        if not self.pool_base_path.exists():
            return []
        available_dates: List[date] = []
        for file_path in self.pool_base_path.glob("*.json"):
            try:
                available_dates.append(datetime.strptime(file_path.stem, "%Y%m%d").date())
            except ValueError:
                continue
        return available_dates

    def save_recommendation_run(
        self,
        *,
        run_id: str,
        trade_date: date,
        candidate_count: int,
        final_count: int,
        report_id: Optional[str],
        items: List[Dict[str, Any]],
        generated_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        return self._get_db_manager().save_recommendation_run(
            run_id=run_id,
            trade_date=trade_date,
            candidate_count=candidate_count,
            final_count=final_count,
            report_id=report_id,
            generated_at=generated_at,
            items=items,
        )

    def load_recommendation_pool_state(self, trade_date: Optional[date] = None) -> List[Dict[str, Any]]:
        if self.settings.use_database:
            return self._get_db_manager().load_recommendation_pool_state(trade_date=trade_date)

        target_date = trade_date
        if target_date is None:
            files = sorted(self.pool_base_path.glob("*.json"), reverse=True)
            if not files:
                return []
            target_date = datetime.strptime(files[0].stem, "%Y%m%d").date()

        file_path = self.pool_base_path / f"{target_date.strftime('%Y%m%d')}.json"
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("states", [])
        except Exception:
            return []

    def upsert_recommendation_pool_states(self, states: List[TrackedRecommendationState]) -> List[Dict[str, Any]]:
        if self.settings.use_database:
            return self._get_db_manager().upsert_recommendation_pool_states(states)

        if not states:
            return []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for state in states:
            key = state.trade_date.strftime("%Y%m%d")
            grouped.setdefault(key, []).append(state.model_dump(mode="json"))

        persisted: List[Dict[str, Any]] = []
        for trade_key, serialized_states in grouped.items():
            file_path = self.pool_base_path / f"{trade_key}.json"
            payload = {
                "trade_date": trade_key,
                "generated_at": datetime.now().isoformat(),
                "states": sorted(serialized_states, key=_recommendation_pool_sort_key),
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            persisted.extend(payload["states"])
        return persisted

    def list_recommendation_pool(
        self,
        trade_date: Optional[date] = None,
        front_only: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if self.settings.use_database:
            return self._get_db_manager().list_recommendation_pool(
                trade_date=trade_date,
                front_only=front_only,
                limit=limit,
            )

        states = self.load_recommendation_pool_state(trade_date=trade_date)
        if front_only is not None:
            states = [item for item in states if bool(item.get("in_frontlist")) == front_only]
        states = sorted(states, key=_recommendation_pool_sort_key)
        if limit is not None:
            states = states[:limit]
        return states

    def list_active_recommendations(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._get_db_manager().list_active_recommendations(limit=limit)

    def list_new_recommendations(self, trade_date: Optional[date] = None, limit: int = 20) -> List[Dict[str, Any]]:
        return self._get_db_manager().list_new_recommendations(trade_date=trade_date, limit=limit)

    def list_recommendation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._get_db_manager().list_recommendation_history(limit=limit)

    def list_pending_performance_updates(self, lookback_days: int = 15, limit: int = 100) -> List[Dict[str, Any]]:
        return self._get_db_manager().list_pending_performance_updates(lookback_days=lookback_days, limit=limit)

    def upsert_recommendation_performance(self, recommendation_item_id: int, performance: Dict[str, Any]) -> Dict[str, Any]:
        return self._get_db_manager().upsert_recommendation_performance(recommendation_item_id, performance)

    def get_recommendation_summary(self, lookback_days: int = 30, history_limit: int = 20) -> Dict[str, Any]:
        return self._get_db_manager().get_recommendation_summary(
            lookback_days=lookback_days,
            history_limit=history_limit,
        )

    def get_latest_results(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略的最新结果

        Returns:
            {strategy_id: latest_result}
        """
        if self.settings.use_database:
            latest_results = {}
            for item in self._get_db_manager().get_screening_history(limit=500):
                strategy_name = item.get("strategy")
                if strategy_name and strategy_name not in latest_results:
                    latest_results[strategy_name] = item
            return latest_results

        latest_results = {}

        for strategy_path in self.base_path.iterdir():
            if not strategy_path.is_dir():
                continue

            # 获取最新的文件
            files = sorted(strategy_path.glob("*.json"), reverse=True)
            if files:
                try:
                    with open(files[0], "r", encoding="utf-8") as f:
                        data = json.load(f)
                        latest_results[strategy_path.name] = data
                except Exception:
                    pass

        return latest_results

    def cleanup_old_results(self, keep_days: int = 90) -> int:
        """
        清理超过指定天数的旧结果

        Args:
            keep_days: 保留最近N天的数据

        Returns:
            删除的文件数
        """
        if self.settings.use_database:
            return 0

        cutoff_date = (date.today() - timedelta(days=keep_days)).strftime("%Y%m%d")
        deleted_count = 0

        for strategy_path in self.base_path.iterdir():
            if not strategy_path.is_dir():
                continue

            for file_path in strategy_path.glob("*.json"):
                # 从文件名提取日期
                file_date = file_path.stem
                if file_date < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1

        return deleted_count
"""Performance optimization for lightweight stock screening."""

import asyncio
import logging
from typing import List, Dict, Any, Set
from datetime import datetime, timedelta
from functools import lru_cache
import pickle
import hashlib

from octts.config import Settings

logger = logging.getLogger(__name__)


class ScreeningOptimizer:
    """
    选股性能优化器

    优化策略：
    1. 智能缓存 - 避免重复计算
    2. 批量处理 - 减少API调用
    3. 并发控制 - 合理利用资源
    4. 增量更新 - 只处理变化数据
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache = {}
        self._cache_ttl = 300  # 缓存5分钟

    # ========== 1. 智能缓存 ==========

    def get_cache_key(self, *args) -> str:
        """生成缓存键"""
        data = str(args).encode()
        return hashlib.md5(data).hexdigest()

    def get_from_cache(self, key: str) -> Any:
        """从缓存获取数据"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                return data
            else:
                del self._cache[key]
        return None

    def set_cache(self, key: str, data: Any):
        """设置缓存"""
        self._cache[key] = (data, datetime.now())

    @lru_cache(maxsize=1000)
    def calculate_technical_indicators(
        self,
        ts_code: str,
        indicators: tuple  # 使用tuple而不是list，因为tuple可哈希
    ) -> Dict[str, float]:
        """
        缓存技术指标计算结果

        使用lru_cache装饰器自动缓存最近1000个计算结果
        """
        # 实际计算逻辑
        result = {}
        for indicator in indicators:
            # 模拟计算
            result[indicator] = 0.0

        return result

    # ========== 2. 批量处理 ==========

    async def batch_fetch_data(
        self,
        ts_codes: List[str],
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        批量获取数据，减少API调用次数

        Args:
            ts_codes: 股票代码列表
            batch_size: 每批数量

        Returns:
            {ts_code: data}
        """
        results = {}

        # 分批处理
        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i + batch_size]

            # 检查缓存
            uncached = []
            for code in batch:
                cache_key = self.get_cache_key('daily_data', code)
                cached_data = self.get_from_cache(cache_key)
                if cached_data:
                    results[code] = cached_data
                else:
                    uncached.append(code)

            # 批量获取未缓存的数据
            if uncached:
                batch_data = await self._fetch_batch_data(uncached)
                for code, data in batch_data.items():
                    results[code] = data
                    # 设置缓存
                    cache_key = self.get_cache_key('daily_data', code)
                    self.set_cache(cache_key, data)

        return results

    async def _fetch_batch_data(
        self,
        ts_codes: List[str]
    ) -> Dict[str, Any]:
        """实际的批量获取逻辑"""
        # 这里模拟批量获取
        # 实际应该调用 Tushare 的批量接口
        return {code: {"close": 10.0} for code in ts_codes}

    # ========== 3. 并发控制 ==========

    async def concurrent_screen(
        self,
        strategies: List[Any],
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        并发执行多个策略，但控制并发数

        Args:
            strategies: 策略列表
            max_concurrent: 最大并发数

        Returns:
            {strategy_id: result}
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []

        async def screen_with_limit(strategy):
            async with semaphore:
                return await self._run_strategy(strategy)

        for strategy in strategies:
            task = asyncio.create_task(
                screen_with_limit(strategy)
            )
            tasks.append((strategy.id, task))

        results = {}
        for strategy_id, task in tasks:
            try:
                result = await task
                results[strategy_id] = result
            except Exception as e:
                logger.error(f"Strategy {strategy_id} failed: {e}")
                results[strategy_id] = None

        return results

    async def _run_strategy(self, strategy):
        """模拟运行策略"""
        await asyncio.sleep(0.1)  # 模拟耗时
        return {"stocks": []}

    # ========== 4. 增量更新 ==========

    def get_stocks_to_update(
        self,
        all_stocks: Set[str],
        last_update_time: datetime
    ) -> Set[str]:
        """
        获取需要更新的股票列表

        只更新：
        1. 新增的股票
        2. 最近有交易的股票
        3. 有重大事件的股票
        """
        now = datetime.now()
        stocks_to_update = set()

        # 如果距离上次更新超过1小时，全部更新
        if now - last_update_time > timedelta(hours=1):
            return all_stocks

        # 否则只更新活跃股票
        # 这里简化处理，实际应该根据成交量、涨跌幅等判断
        for stock in all_stocks:
            # 检查缓存中的最后更新时间
            cache_key = self.get_cache_key('last_trade', stock)
            last_trade = self.get_from_cache(cache_key)

            if not last_trade or now - last_trade > timedelta(minutes=5):
                stocks_to_update.add(stock)

        return stocks_to_update

    # ========== 5. 优化建议 ==========

    def get_optimization_suggestions(
        self,
        screening_stats: Dict[str, Any]
    ) -> List[str]:
        """
        根据使用统计给出优化建议

        Args:
            screening_stats: 筛选统计信息

        Returns:
            优化建议列表
        """
        suggestions = []

        # 分析耗时
        avg_time = screening_stats.get('avg_execution_time', 0)
        if avg_time > 10:
            suggestions.append(
                f"平均执行时间 {avg_time:.1f}秒 较长，"
                f"建议：减少选股数量或使用更严格的初筛条件"
            )

        # 分析命中率
        hit_rate = screening_stats.get('cache_hit_rate', 0)
        if hit_rate < 0.5:
            suggestions.append(
                f"缓存命中率 {hit_rate:.1%} 较低，"
                f"建议：增加缓存时间或预加载常用数据"
            )

        # 分析股票数量
        total_stocks = screening_stats.get('total_stocks', 0)
        if total_stocks > 3000:
            suggestions.append(
                f"扫描股票数 {total_stocks} 较多，"
                f"建议：添加市值、行业等预筛选条件"
            )

        # 分析策略复杂度
        avg_indicators = screening_stats.get('avg_indicators_per_strategy', 0)
        if avg_indicators > 10:
            suggestions.append(
                f"平均每个策略使用 {avg_indicators} 个指标，"
                f"建议：简化策略或分阶段筛选"
            )

        return suggestions


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.stats = {
            'total_requests': 0,
            'total_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }

    def record_request(self, duration: float, cache_hit: bool = False):
        """记录请求"""
        self.stats['total_requests'] += 1
        self.stats['total_time'] += duration
        if cache_hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1

    def record_error(self):
        """记录错误"""
        self.stats['errors'] += 1

    def get_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        total_requests = self.stats['total_requests']
        if total_requests == 0:
            return {}

        return {
            'total_requests': total_requests,
            'avg_response_time': self.stats['total_time'] / total_requests,
            'cache_hit_rate': self.stats['cache_hits'] / total_requests,
            'error_rate': self.stats['errors'] / total_requests,
            'total_errors': self.stats['errors']
        }

    def reset(self):
        """重置统计"""
        self.stats = {
            'total_requests': 0,
            'total_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }


# ========== 实用工具函数 ==========

def optimize_stock_list(
    all_stocks: List[Dict[str, Any]],
    pre_filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    优化股票列表，提前过滤明显不符合的股票

    Args:
        all_stocks: 所有股票
        pre_filters: 预过滤条件

    Returns:
        过滤后的股票列表
    """
    filtered = []

    for stock in all_stocks:
        # 市值过滤
        if 'min_market_cap' in pre_filters:
            if stock.get('total_mv', 0) < pre_filters['min_market_cap']:
                continue

        if 'max_market_cap' in pre_filters:
            if stock.get('total_mv', 0) > pre_filters['max_market_cap']:
                continue

        # ST股过滤
        if pre_filters.get('exclude_st', True):
            if 'ST' in stock.get('name', ''):
                continue

        # 停牌过滤
        if pre_filters.get('exclude_suspended', True):
            # 简化判断：如果最近没有成交量，可能停牌
            if stock.get('vol', 0) == 0:
                continue

        filtered.append(stock)

    logger.info(
        f"Pre-filtered stocks: {len(all_stocks)} -> {len(filtered)} "
        f"({len(filtered)/len(all_stocks):.1%})"
    )

    return filtered


def estimate_screening_time(
    num_stocks: int,
    num_indicators: int,
    cache_hit_rate: float = 0.5
) -> float:
    """
    估算筛选时间

    Args:
        num_stocks: 股票数量
        num_indicators: 指标数量
        cache_hit_rate: 缓存命中率

    Returns:
        预估时间（秒）
    """
    # 基础时间：每只股票每个指标 0.01 秒
    base_time = num_stocks * num_indicators * 0.01

    # 缓存加速
    actual_time = base_time * (1 - cache_hit_rate)

    # 批量处理加速（假设批量处理能减少50%时间）
    if num_stocks > 100:
        actual_time *= 0.5

    # 并发加速（假设3并发）
    actual_time /= 3

    return max(actual_time, 1.0)  # 至少1秒
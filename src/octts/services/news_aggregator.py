"""News collection and analysis system."""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

import httpx
from bs4 import BeautifulSoup

from octts.config import Settings
from octts.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


class NewsSource(Enum):
    """新闻源枚举"""
    CAILIAN = "财联社"
    EASTMONEY = "东方财富"
    SINA = "新浪财经"
    WALLSTREET = "华尔街见闻"
    YUNCAIJING = "云财经"
    JINSHI = "金十数据"
    XUEQIU = "雪球"
    PENGPAI = "澎湃新闻"
    IFENG = "凤凰财经"
    YICAI = "第一财经"


@dataclass
class NewsItem:
    """新闻条目"""
    source: NewsSource
    title: str
    content: str
    url: str
    publish_time: datetime
    tags: List[str] = None
    importance: float = 0.5  # 0-1 重要性评分
    sentiment: float = 0.0   # -1到1 情绪评分
    related_stocks: List[str] = None  # 相关股票代码


@dataclass
class NewsCluster:
    """新闻聚类"""
    cluster_id: str
    theme: str  # 主题
    news_items: List[NewsItem]
    importance: float
    summary: str
    key_stocks: List[str]  # 主要影响的股票


class NewsCollector:
    """新闻采集器基类"""

    def __init__(self, source: NewsSource):
        self.source = source
        self.logger = logging.getLogger(f"news.{source.value}")

    async def collect(self) -> List[NewsItem]:
        """采集新闻（子类实现）"""
        raise NotImplementedError


class CailianNewsCollector(NewsCollector):
    """财联社新闻采集器"""

    def __init__(self):
        super().__init__(NewsSource.CAILIAN)
        self.api_url = "https://www.cls.cn/nodeapi/updateTelegraphList"

    async def collect(self) -> List[NewsItem]:
        """采集财联社实时新闻"""
        news_items = []

        timeout = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            try:
                response = await client.get(
                    self.api_url,
                    params={
                        "app": "CailianpressWeb",
                        "os": "web",
                        "sv": "8.4.6",
                    },
                )
                response.raise_for_status()
                data = response.json()

                for item in data.get("data", {}).get("roll_data", []):
                    if item.get("is_ad"):
                        continue

                    title = item.get("title") or item.get("brief", "")
                    content = item.get("content", "") or item.get("brief", "")
                    if not title:
                        continue

                    ctime = item.get("ctime", 0)
                    news_item = NewsItem(
                        source=self.source,
                        title=title,
                        content=content,
                        url=f"https://www.cls.cn/detail/{item.get('id', '')}",
                        publish_time=datetime.fromtimestamp(ctime) if ctime else datetime.now(),
                        tags=["电报"] + (["重要"] if item.get("important") else []),
                    )
                    news_items.append(news_item)

            except Exception as e:
                self.logger.error(f"Failed to collect from {self.source.value}: {e}")

        return news_items


class EastMoneyNewsCollector(NewsCollector):
    """东方财富新闻采集器"""

    def __init__(self):
        super().__init__(NewsSource.EASTMONEY)
        self.api_url = "http://finance.eastmoney.com/api/news"

    async def collect(self) -> List[NewsItem]:
        """采集东方财富新闻"""
        # 预留接口，当前未接入具体源。
        return []


class NewsAggregator:
    """新闻聚合器"""

    _IMPORTANT_SOURCES = {
        NewsSource.CAILIAN,
        NewsSource.WALLSTREET,
        NewsSource.JINSHI,
        NewsSource.YICAI,
        NewsSource.PENGPAI,
    }
    _IMPORTANT_KEYWORDS = (
        "涨停",
        "跌停",
        "业绩",
        "预增",
        "预亏",
        "回购",
        "增持",
        "减持",
        "并购",
        "重组",
        "停牌",
        "复牌",
        "监管",
        "政策",
        "财报",
        "订单",
        "算力",
        "芯片",
        "机器人",
        "军工",
        "新能源",
        "医药",
        "银行",
        "地产",
    )

    def __init__(
        self,
        settings: Settings,
        llm_client: Optional[LLMClient] = None
    ):
        self.settings = settings
        self.llm_client = llm_client or LLMClient(settings)

        # 初始化采集器
        self.collectors = [
            CailianNewsCollector(),
            EastMoneyNewsCollector(),
            # 添加更多采集器...
        ]

        # 缓存已处理的新闻
        self._processed_hashes = set()

    async def collect_all(
        self,
        sources: Optional[List[NewsSource]] = None
    ) -> List[NewsItem]:
        """
        并发采集所有源的新闻

        Args:
            sources: 指定采集源，None表示全部

        Returns:
            所有新闻列表
        """
        # 过滤采集器
        collectors = self.collectors
        if sources:
            collectors = [c for c in collectors if c.source in sources]

        # 并发采集
        tasks = [collector.collect() for collector in collectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        all_news = []
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            else:
                logger.error(f"Collection error: {result}")

        # 去重
        return self._deduplicate_news(all_news)

    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """新闻去重"""
        unique_news = []

        for item in news_items:
            # 计算内容哈希
            content_hash = hashlib.md5(
                f"{item.title}{item.content}".encode()
            ).hexdigest()

            if content_hash not in self._processed_hashes:
                self._processed_hashes.add(content_hash)
                unique_news.append(item)

        return unique_news

    async def analyze_importance(
        self,
        news_items: List[NewsItem]
    ) -> List[NewsItem]:
        """
        使用LLM分析新闻重要性

        Args:
            news_items: 新闻列表

        Returns:
            标注了重要性的新闻列表
        """
        batch_size = 5
        filtered_items = self._prefilter_importance_candidates(news_items)
        total_batches = (len(filtered_items) + batch_size - 1) // batch_size if filtered_items else 0

        for item in news_items:
            item.importance = item.importance if item.importance is not None else 0.5
            item.sentiment = item.sentiment if item.sentiment is not None else 0.0
            item.related_stocks = item.related_stocks or []

        for i in range(0, len(filtered_items), batch_size):
            batch = filtered_items[i:i + batch_size]
            batch_number = i // batch_size + 1
            logger.info(
                "News importance batch %s/%s start: %s items",
                batch_number,
                total_batches,
                len(batch),
            )

            news_texts = []
            for idx, item in enumerate(batch):
                news_texts.append(
                    f"{idx + 1}. 【{item.source.value}】{item.title}\n"
                    f"   内容：{item.content[:150]}..."
                )

            prompt = f"""
            请分析以下财经新闻的重要性和市场影响：

            {chr(10).join(news_texts)}

            对每条新闻，请评估：
            1. 重要性评分（0-1）：0表示不重要，1表示非常重要
            2. 市场情绪（-1到1）：-1表示利空，1表示利好
            3. 相关股票代码（如果有）
            4. 简短理由

            返回JSON格式：
            {{
                "1": {{
                    "importance": 0.8,
                    "sentiment": 0.5,
                    "stocks": ["000001.SZ", "000002.SZ"],
                    "reason": "央行降息利好银行股"
                }},
                ...
            }}
            """

            try:
                response = await self.llm_client.complete(prompt)
                results = self._parse_json_response(response)
                logger.info(
                    "News importance batch %s/%s complete: parsed %s items",
                    batch_number,
                    total_batches,
                    len(results),
                )

                for idx, item in enumerate(batch):
                    key = str(idx + 1)
                    if key in results:
                        item.importance = results[key].get("importance", item.importance)
                        item.sentiment = results[key].get("sentiment", item.sentiment)
                        item.related_stocks = results[key].get("stocks", item.related_stocks)

            except Exception as e:
                logger.error(
                    "News importance batch %s/%s failed: %s",
                    batch_number,
                    total_batches,
                    e,
                )

        return news_items

    async def cluster_news(
        self,
        news_items: List[NewsItem],
        min_cluster_size: int = 3
    ) -> List[NewsCluster]:
        """
        新闻聚类分析

        Args:
            news_items: 新闻列表
            min_cluster_size: 最小聚类大小

        Returns:
            新闻聚类列表
        """
        if len(news_items) < min_cluster_size:
            return []

        # 使用LLM进行语义聚类
        prompt = f"""
        请对以下新闻进行主题聚类分析：

        {self._format_news_for_clustering(news_items[:50])}  # 限制数量

        要求：
        1. 将相似主题的新闻归为一类
        2. 每个聚类至少包含{min_cluster_size}条新闻
        3. 为每个聚类生成一个主题概括
        4. 识别每个聚类的核心股票

        返回JSON格式：
        {{
            "clusters": [
                {{
                    "theme": "央行货币政策放松",
                    "news_indices": [1, 5, 8, 12],
                    "importance": 0.9,
                    "key_stocks": ["银行股", "地产股"],
                    "summary": "央行释放流动性信号..."
                }},
                ...
            ]
        }}
        """

        try:
            response = await self.llm_client.complete(prompt)
            result = self._parse_json_response(response)

            clusters = []
            for cluster_data in result.get("clusters", []):
                # 收集该聚类的新闻
                indices = cluster_data.get("news_indices", [])
                cluster_news = [
                    news_items[i - 1] for i in indices
                    if 0 < i <= len(news_items)
                ]

                if len(cluster_news) >= min_cluster_size:
                    cluster = NewsCluster(
                        cluster_id=hashlib.md5(
                            cluster_data["theme"].encode()
                        ).hexdigest()[:8],
                        theme=cluster_data["theme"],
                        news_items=cluster_news,
                        importance=cluster_data.get("importance", 0.5),
                        summary=cluster_data.get("summary", ""),
                        key_stocks=cluster_data.get("key_stocks", [])
                    )
                    clusters.append(cluster)

        except Exception as e:
            logger.error(f"Failed to cluster news: {e}")
            return []

        return clusters

    def _prefilter_importance_candidates(self, news_items: List[NewsItem]) -> List[NewsItem]:
        selected: List[NewsItem] = []
        seen_keys = set()
        for item in news_items:
            text = f"{item.title} {item.content}".strip()
            normalized = "".join(text.split()).lower()
            if len(item.title.strip()) < 8:
                item.importance = 0.2
                item.sentiment = item.sentiment if item.sentiment is not None else 0.0
                item.related_stocks = item.related_stocks or []
                continue
            if len(item.content.strip()) < 20:
                item.importance = 0.2
                item.sentiment = item.sentiment if item.sentiment is not None else 0.0
                item.related_stocks = item.related_stocks or []
                continue
            dedupe_key = normalized[:120]
            if dedupe_key in seen_keys:
                item.importance = 0.2
                item.sentiment = item.sentiment if item.sentiment is not None else 0.0
                item.related_stocks = item.related_stocks or []
                continue
            seen_keys.add(dedupe_key)
            has_stock_hint = bool(item.related_stocks) or bool(self._extract_stock_codes(text))
            has_keyword = any(keyword in text for keyword in self._IMPORTANT_KEYWORDS)
            important_source = item.source in self._IMPORTANT_SOURCES
            has_tags = bool(item.tags and any(tag for tag in item.tags if tag in {"重要", "电报", "公告"}))
            if important_source or has_keyword or has_stock_hint or has_tags:
                selected.append(item)
            else:
                item.importance = 0.25
                item.sentiment = item.sentiment if item.sentiment is not None else 0.0
                item.related_stocks = item.related_stocks or []
        return selected

    def _extract_stock_codes(self, text: str) -> List[str]:
        import re

        return re.findall(r"\b\d{6}\.(?:SH|SZ)\b", text.upper())

    def _format_news_for_clustering(
        self,
        news_items: List[NewsItem]
    ) -> str:
        """格式化新闻用于聚类"""
        lines = []
        for idx, item in enumerate(news_items):
            lines.append(
                f"{idx + 1}. {item.title} ({item.source.value})"
            )
        return "\n".join(lines)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """解析JSON响应"""
        import re

        try:
            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return {}


class NewsScheduler:
    """新闻采集调度器"""

    def __init__(
        self,
        settings: Settings,
        aggregator: NewsAggregator
    ):
        self.settings = settings
        self.aggregator = aggregator
        self.logger = logging.getLogger("news.scheduler")

    async def run_scheduled_collection(self):
        """定时采集任务"""
        # 财经快讯类 - 每5分钟
        fast_sources = [
            NewsSource.CAILIAN,
            NewsSource.WALLSTREET,
            NewsSource.JINSHI
        ]

        # 综合资讯类 - 每30分钟
        general_sources = [
            NewsSource.SINA,
            NewsSource.EASTMONEY,
            NewsSource.YICAI
        ]

        # 深度分析类 - 每小时
        depth_sources = [
            NewsSource.PENGPAI,
            NewsSource.XUEQIU
        ]

        while True:
            try:
                # 根据时间判断采集哪些源
                current_minute = datetime.now().minute

                sources_to_collect = []

                # 每5分钟采集快讯
                if current_minute % 5 == 0:
                    sources_to_collect.extend(fast_sources)

                # 每30分钟采集综合资讯
                if current_minute % 30 == 0:
                    sources_to_collect.extend(general_sources)

                # 每小时采集深度分析
                if current_minute == 0:
                    sources_to_collect.extend(depth_sources)

                if sources_to_collect:
                    # 采集新闻
                    news_items = await self.aggregator.collect_all(
                        sources_to_collect
                    )

                    # 分析重要性
                    news_items = await self.aggregator.analyze_importance(
                        news_items
                    )

                    # 聚类分析
                    clusters = await self.aggregator.cluster_news(news_items)

                    # 存储结果
                    await self._save_results(news_items, clusters)

                    self.logger.info(
                        f"Collected {len(news_items)} news, "
                        f"formed {len(clusters)} clusters"
                    )

            except Exception as e:
                self.logger.error(f"Collection error: {e}")

            # 等待下一分钟
            await asyncio.sleep(60)

    async def _save_results(
        self,
        news_items: List[NewsItem],
        clusters: List[NewsCluster]
    ):
        """保存采集结果"""
        # TODO: 保存到数据库或文件
        pass
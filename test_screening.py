"""Test daily stock screening functionality."""

import asyncio
from octts.config import Settings
from octts.services.stock_screening_scheduler import create_screening_scheduler


async def test_screening():
    """测试选股功能"""
    # 创建测试配置
    settings = Settings(
        OCTTS_MEMORY_BACKEND="file",
        OCTTS_SCREENING_ENABLED=True,
        OCTTS_SCREENING_STRATEGIES="oversold_bounce,volume_breakout",
        OCTTS_SCREENING_NOTIFY=False,  # 测试时不发送通知
    )

    # 创建调度器
    scheduler = create_screening_scheduler(settings)

    # 运行选股
    print("开始运行选股任务...")
    result = await scheduler.run_daily_screening()

    print(f"\n选股完成！")
    print(f"运行策略数: {result['strategies_run']}")
    print(f"选出股票总数: {result['total_stocks']}")
    print(f"执行时间: {result['duration_seconds']:.2f}秒")


if __name__ == "__main__":
    asyncio.run(test_screening())
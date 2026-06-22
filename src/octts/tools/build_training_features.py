"""
预构建训练特征数据，保存到数据库中。

用法:
    python -m octts.tools.build_training_features \
        --start-date 2025-08-01 \
        --end-date 2026-04-22 \
        --min-history-days 20
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from octts.config import Settings
from octts.schemas.training import RawMarketTrainingSample
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# 与 train_raw_market_model.py 的默认训练特征保持一致，避免预计算表与训练/预测口径漂移。
FEATURE_COLUMNS = list(RAW_MARKET_FEATURE_COLUMNS)

# 目标列
TARGET_COLUMNS = [
    "return_1d", "return_3d", "return_5d", "return_10d",
    "vs_market_1d", "vs_market_3d", "vs_market_5d",
    "label_up_1d", "label_up_3d", "label_up_5d",
    "label_vs_market_1d", "label_vs_market_3d", "label_vs_market_5d",
    "label_strong_1d",
    "label_limit_relay_success_1d", "label_limit_relay_strong_1d",
    "label_limit_relay_success_3d", "label_limit_relay_limit_up_1d",
]

ALL_COLUMNS = ["trade_date", "ts_code"] + FEATURE_COLUMNS + TARGET_COLUMNS


def create_table_if_not_exists(repo: MarketRawDataRepository) -> None:
    """创建特征表（如果不存在），并为已有表补齐缺失列。"""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS training_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date DATE NOT NULL,
        ts_code VARCHAR(20) NOT NULL,
        {', '.join([f'{col} FLOAT' for col in FEATURE_COLUMNS])},
        {', '.join([f'{col} FLOAT' for col in TARGET_COLUMNS])},
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(trade_date, ts_code)
    )
    """
    with repo._db.get_session() as session:
        session.execute(text(create_sql))
        existing_columns = {
            row[1]
            for row in session.execute(text("PRAGMA table_info(training_features)")).fetchall()
        }
        required_feature_columns = {col: "FLOAT" for col in FEATURE_COLUMNS}
        required_target_columns = {col: "FLOAT" for col in TARGET_COLUMNS}
        required_columns = {
            **required_feature_columns,
            **required_target_columns,
        }
        for column_name, column_type in required_columns.items():
            if column_name in existing_columns:
                continue
            session.execute(text(f"ALTER TABLE training_features ADD COLUMN {column_name} {column_type}"))
            logger.info("training_features 表新增缺失列: %s %s", column_name, column_type)
        session.commit()
    logger.info("特征表已就绪")


def insert_features(repo: MarketRawDataRepository, samples: List[RawMarketTrainingSample], batch_size: int = 5000) -> int:
    """批量插入特征数据。"""
    if not samples:
        return 0

    # 使用命名参数
    col_names = ALL_COLUMNS
    placeholders = ", ".join([f":{col}" for col in col_names])
    insert_sql = f"INSERT OR REPLACE INTO training_features ({', '.join(col_names)}) VALUES ({placeholders})"

    total_inserted = 0
    batch = []

    for sample in samples:
        row = {
            "trade_date": sample.trade_date.strftime("%Y-%m-%d"),
            "ts_code": sample.ts_code,
        }
        # 特征列
        for col in FEATURE_COLUMNS:
            row[col] = getattr(sample, col, None)
        # 目标列
        for col in TARGET_COLUMNS:
            row[col] = getattr(sample, col, None)
        batch.append(row)

        if len(batch) >= batch_size:
            with repo._db.engine.connect() as conn:
                conn.execute(text(insert_sql), batch)
                conn.commit()
            total_inserted += len(batch)
            logger.info("已插入 %d 条记录...", total_inserted)
            batch = []

    # 插入剩余数据
    if batch:
        with repo._db.engine.connect() as conn:
            conn.execute(text(insert_sql), batch)
            conn.commit()
        total_inserted += len(batch)

    return total_inserted


def build_features(
    settings: Settings,
    start_date: date,
    end_date: date,
    min_history_days: int,
    exclude_bj: bool,
) -> Dict[str, Any]:
    """构建特征数据。"""
    repo = MarketRawDataRepository(settings.database_url)
    builder = RawMarketTrainingDatasetBuilder(settings)

    # 创建表
    create_table_if_not_exists(repo)

    # 构建样本
    logger.info("开始构建特征数据: %s ~ %s", start_date, end_date)
    logger.info("参数: min_history_days=%d, exclude_bj=%s", min_history_days, exclude_bj)
    start_time = time.time()

    # 获取交易日列表（用于进度显示）
    trading_dates = builder.repo.list_trading_dates(
        start_date=(start_date - timedelta(days=60)).strftime("%Y%m%d"),
        end_date=(end_date + timedelta(days=20)).strftime("%Y%m%d"),
    )
    sample_dates = [
        d for d in trading_dates
        if start_date <= datetime.strptime(d, "%Y%m%d").date() <= end_date
    ]
    logger.info("交易日数量: %d 天 (样本日期范围)", len(sample_dates))

    logger.info("正在加载市场数据...")
    samples = builder.build_samples(
        start_date=start_date,
        end_date=end_date,
        min_history_days=min_history_days,
        exclude_bj=exclude_bj,
    )

    build_time = time.time() - start_time
    logger.info("特征构建完成: 样本数=%d, 耗时=%.2f秒", len(samples), build_time)
    if len(samples) == 0:
        logger.warning("警告: 未生成任何样本，请检查日期范围和数据完整性")
        return {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "total_samples": 0,
            "inserted": 0,
        }

    # 统计信息
    trade_dates = sorted(set(s.trade_date for s in samples))
    ts_codes = sorted(set(s.ts_code for s in samples))
    logger.info("日期范围: %s ~ %s, 共 %d 个交易日", trade_dates[0], trade_dates[-1], len(trade_dates))
    logger.info("股票数量: %d 只", len(ts_codes))

    # 插入数据库
    logger.info("开始写入数据库...")
    insert_start = time.time()
    inserted = insert_features(repo, samples)
    insert_time = time.time() - insert_start
    logger.info("写入完成: 插入 %d 条记录, 耗时=%.2f秒", inserted, insert_time)

    # 统计各字段的空值情况
    logger.info("\n=== 字段统计 ===")
    for col in FEATURE_COLUMNS[:10]:
        values = [getattr(s, col, None) for s in samples]
        valid = sum(1 for v in values if v is not None)
        logger.info("%s: 有效值=%d (%.1f%%)", col, valid, valid / len(samples) * 100)
    logger.info("... (共 %d 个特征列)", len(FEATURE_COLUMNS))

    logger.info("\n=== 目标变量统计 ===")
    for col in TARGET_COLUMNS:
        values = [getattr(s, col, None) for s in samples]
        valid = sum(1 for v in values if v is not None)
        logger.info("%s: 有效值=%d (%.1f%%)", col, valid, valid / len(samples) * 100)

    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "min_history_days": min_history_days,
        "exclude_bj": exclude_bj,
        "total_samples": len(samples),
        "trade_dates": len(trade_dates),
        "ts_codes": len(ts_codes),
        "inserted": inserted,
        "build_time_seconds": round(build_time, 2),
        "insert_time_seconds": round(insert_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="预构建训练特征数据")
    parser.add_argument("--start-date", type=str, required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--min-history-days", type=int, default=20, help="最小历史天数")
    parser.add_argument("--exclude-bj", action="store_true", help="排除北交所股票")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    settings = Settings()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    result = build_features(
        settings,
        start_date=start_date,
        end_date=end_date,
        min_history_days=args.min_history_days,
        exclude_bj=args.exclude_bj,
    )

    print("\n" + "=" * 50)
    print("特征构建完成")
    print("=" * 50)
    print(f"日期范围: {result['start_date']} ~ {result['end_date']}")
    print(f"样本总数: {result['total_samples']:,}")
    print(f"交易日数: {result['trade_dates']}")
    print(f"股票数量: {result['ts_codes']}")
    print(f"插入记录: {result['inserted']:,}")
    print(f"构建耗时: {result['build_time_seconds']}秒")
    print(f"写入耗时: {result['insert_time_seconds']}秒")


if __name__ == "__main__":
    main()

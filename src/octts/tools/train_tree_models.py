"""
训练并对比树模型效果。

用法:
    python -m octts.tools.train_tree_models \
        --train-start 2025-10-01 --train-end 2026-02-28 \
        --test-start 2026-03-01 --test-end 2026-03-15 \
        --output tmp/model_comparison_tree.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from octts.config import Settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder

logger = logging.getLogger(__name__)

# 34个固定特征
FEATURE_COLUMNS = [
    "close", "pct_change", "turnover_rate", "volume_ratio", "market_cap",
    "pe_ttm", "pb", "amount", "vol", "volatility_5d", "volatility_10d",
    "max_drawdown_10d_past", "close_to_ma5", "close_to_ma10",
    "price_position_20d", "price_position_10d", "avg_turnover_rate_5d",
    "avg_volume_ratio_5d", "market_return_1d", "market_return_3d",
    "market_return_5d", "market_up_ratio_1d", "market_up_ratio_3d_avg",
    "market_up_days_5d", "stock_vs_market_return_1d", "stock_vs_market_return_3d",
    "stock_vs_market_return_10d", "pct_change_rank_pct", "turnover_rate_rank_pct",
    "volume_ratio_rank_pct", "up_days_3d", "up_days_5d",
    "new_high_gap_20d", "new_low_gap_20d",
]

# 目标变量
TARGET_COLUMNS = [
    "return_1d",
    "return_3d",
    "vs_market_1d",
    "label_up_1d",
    "label_strong_1d",
]

ACTUAL_RETURN_COLUMNS = ["return_1d", "return_3d", "return_5d"]

# 可选导入树模型
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRanker, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# 树模型配置
TREE_MODEL_CONFIGS = {
    "rf": {
        "model": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 100, "max_depth": 5},
    },
}
if HAS_XGBOOST:
    TREE_MODEL_CONFIGS["xgb"] = {
        "model": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    }
if HAS_LIGHTGBM:
    TREE_MODEL_CONFIGS["lgbm"] = {
        "model": LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    }

# LambdaRank 模型配置（需要单独处理）
RANK_MODEL_CONFIGS = {}
if HAS_XGBOOST:
    RANK_MODEL_CONFIGS["xgb_lambdarank"] = {
        "model": XGBRanker(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective="rank:pairwise",
            random_state=42,
            n_jobs=-1,
        ),
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "objective": "rank:pairwise"},
    }


@dataclass
class ModelResult:
    model_name: str
    target_name: str
    train_samples: int
    test_samples: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    accuracy: Optional[float] = None
    precision_pos: Optional[float] = None
    recall_pos: Optional[float] = None
    f1_pos: Optional[float] = None
    backtest_return_1d: Optional[float] = None
    backtest_return_3d: Optional[float] = None
    backtest_return_5d: Optional[float] = None
    top3_accuracy: Optional[float] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)


def _build_dataset_from_samples(
    settings: Settings,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    builder = RawMarketTrainingDatasetBuilder(settings)
    samples = builder.build_samples(
        start_date=start_date,
        end_date=end_date,
        min_history_days=20,
        exclude_bj=True,
    )

    records = []
    for sample in samples:
        record = {"ts_code": sample.ts_code, "trade_date": sample.trade_date}
        for col in FEATURE_COLUMNS:
            record[col] = getattr(sample, col, None)
        for col in TARGET_COLUMNS:
            record[col] = getattr(sample, col, None)
        record["return_1d_actual"] = sample.return_1d
        record["return_3d_actual"] = sample.return_3d
        record["return_5d_actual"] = sample.return_5d
        records.append(record)

    return pd.DataFrame(records)


def _load_dataset_from_training_features(
    settings: Settings,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    repo = MarketRawDataRepository(settings.database_url)
    inspector = inspect(repo._db.engine)
    if not inspector.has_table("training_features"):
        return pd.DataFrame()

    required_columns = ["trade_date", "ts_code"] + FEATURE_COLUMNS + TARGET_COLUMNS + ACTUAL_RETURN_COLUMNS
    available_columns = {column["name"] for column in inspector.get_columns("training_features")}
    if any(column not in available_columns for column in required_columns):
        return pd.DataFrame()

    selected_columns = list(dict.fromkeys(["trade_date", "ts_code"] + FEATURE_COLUMNS + TARGET_COLUMNS + ACTUAL_RETURN_COLUMNS))
    query = text(
        "SELECT "
        + ", ".join(selected_columns)
        + " FROM training_features WHERE trade_date >= :start_date AND trade_date <= :end_date ORDER BY trade_date, ts_code"
    )
    with repo._db.engine.connect() as conn:
        frame = pd.read_sql_query(
            query,
            conn,
            params={
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            },
        )

    if frame.empty:
        return frame

    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["return_1d_actual"] = frame["return_1d"]
    frame["return_3d_actual"] = frame["return_3d"]
    frame["return_5d_actual"] = frame["return_5d"]
    return frame


def build_dataset(
    settings: Settings,
    start_date: date,
    end_date: date,
    *,
    prefer_training_features: bool = True,
    training_features_only: bool = False,
) -> pd.DataFrame:
    """优先从 training_features 读取，失败时回退到动态构建。"""
    if prefer_training_features or training_features_only:
        frame = _load_dataset_from_training_features(settings, start_date, end_date)
        if not frame.empty:
            logger.info("从 training_features 读取数据: %s ~ %s, 样本数=%d", start_date, end_date, len(frame))
            return frame
        if training_features_only:
            logger.warning("training_features_only 已启用，但 training_features 区间为空或不可用: %s ~ %s", start_date, end_date)
            return frame

    logger.info("training_features 不可用或区间为空，回退到动态构建: %s ~ %s", start_date, end_date)
    return _build_dataset_from_samples(settings, start_date, end_date)


def train_model(
    train_df: pd.DataFrame,
    target: str,
    model_name: str,
) -> tuple[Any, ModelResult]:
    """训练单个树模型。"""
    X = train_df[FEATURE_COLUMNS].fillna(0.0)
    y = train_df[target].values

    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]

    model_config = TREE_MODEL_CONFIGS[model_name]
    model = model_config["model"]
    model.fit(X, y)

    # 特征重要性
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    else:
        importance = {}

    result = ModelResult(
        model_name=model_name,
        target_name=target,
        train_samples=len(y),
        test_samples=0,
        feature_importance=importance,
    )

    return model, result


def prepare_ranking_data(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """准备排序学习数据。

    按交易日分组，每组内的股票按目标值排序。

    Returns:
        X: 特征矩阵
        y: 目标值
        group: 每组样本数量
    """
    df = df.copy()
    df = df.dropna(subset=[target])

    X_list = []
    y_list = []
    group_list = []

    for trade_date in sorted(df["trade_date"].unique()):
        day_df = df[df["trade_date"] == trade_date]
        if len(day_df) < 3:
            continue

        X_list.append(day_df[FEATURE_COLUMNS].fillna(0.0).values)
        y_list.append(day_df[target].values)
        group_list.append(len(day_df))

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.array(group_list)

    return X, y, groups


def train_rank_model(
    train_df: pd.DataFrame,
    target: str,
    model_name: str,
) -> tuple[Any, ModelResult]:
    """训练 LambdaRank 模型。"""
    X, y, groups = prepare_ranking_data(train_df, target)

    if len(X) == 0:
        logger.warning("无有效数据用于 LambdaRank 训练")
        return None, ModelResult(
            model_name=model_name,
            target_name=target,
            train_samples=0,
            test_samples=0,
        )

    model_config = RANK_MODEL_CONFIGS[model_name]
    model = model_config["model"]
    model.fit(X, y, group=groups)

    # 特征重要性
    if hasattr(model, "feature_importances_"):
        importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    else:
        importance = {}

    result = ModelResult(
        model_name=model_name,
        target_name=target,
        train_samples=len(y),
        test_samples=0,
        feature_importance=importance,
    )

    return model, result


def evaluate_model(
    model: Any,
    test_df: pd.DataFrame,
    target: str,
    result: ModelResult,
) -> ModelResult:
    """评估模型效果。"""
    X_test = test_df[FEATURE_COLUMNS].fillna(0.0)
    y_test = test_df[target].values

    valid_mask = ~pd.isna(y_test)
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]

    y_pred = model.predict(X_test)

    result.test_samples = len(y_test)

    if target in ["label_up_1d", "label_strong_1d"]:
        # 分类任务
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_test_binary = y_test.astype(int)

        result.accuracy = float((y_pred_binary == y_test_binary).mean())

        tp = ((y_pred_binary == 1) & (y_test_binary == 1)).sum()
        fp = ((y_pred_binary == 1) & (y_test_binary == 0)).sum()
        fn = ((y_pred_binary == 0) & (y_test_binary == 1)).sum()

        result.precision_pos = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        result.recall_pos = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if result.precision_pos + result.recall_pos > 0:
            result.f1_pos = 2 * result.precision_pos * result.recall_pos / (result.precision_pos + result.recall_pos)
        else:
            result.f1_pos = 0.0
    else:
        # 回归任务
        result.mae = float(mean_absolute_error(y_test, y_pred))
        result.rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        result.r2 = float(r2_score(y_test, y_pred))

    return result


def backtest_top3(
    model: Any,
    test_df: pd.DataFrame,
    target: str,
    result: ModelResult,
) -> ModelResult:
    """回测：每天选预测最高的3只股票，计算实际收益。"""
    test_df = test_df.copy()
    test_df["pred"] = model.predict(test_df[FEATURE_COLUMNS].fillna(0.0))

    daily_returns_1d = []
    daily_returns_3d = []
    daily_returns_5d = []

    for trade_date in test_df["trade_date"].unique():
        day_df = test_df[test_df["trade_date"] == trade_date]
        if len(day_df) < 3:
            continue

        top3 = day_df.nlargest(3, "pred")

        for _, row in top3.iterrows():
            if pd.notna(row["return_1d_actual"]):
                daily_returns_1d.append(row["return_1d_actual"])
            if pd.notna(row["return_3d_actual"]):
                daily_returns_3d.append(row["return_3d_actual"])
            if pd.notna(row["return_5d_actual"]):
                daily_returns_5d.append(row["return_5d_actual"])

    if daily_returns_1d:
        result.backtest_return_1d = float(np.mean(daily_returns_1d))
    if daily_returns_3d:
        result.backtest_return_3d = float(np.mean(daily_returns_3d))
    if daily_returns_5d:
        result.backtest_return_5d = float(np.mean(daily_returns_5d))

    # Top3准确率
    if target in ["return_1d", "return_3d", "vs_market_1d"]:
        correct = 0
        total = 0
        for trade_date in test_df["trade_date"].unique():
            day_df = test_df[test_df["trade_date"] == trade_date]
            if len(day_df) < 3:
                continue
            top3 = day_df.nlargest(3, "pred")
            for _, row in top3.iterrows():
                if pd.notna(row["return_1d_actual"]):
                    total += 1
                    if row["return_1d_actual"] > 0:
                        correct += 1
        if total > 0:
            result.top3_accuracy = float(correct / total)

    return result


def main():
    parser = argparse.ArgumentParser(description="训练并对比树模型")
    parser.add_argument("--train-start", type=str, required=True, help="训练开始日期")
    parser.add_argument("--train-end", type=str, required=True, help="训练结束日期")
    parser.add_argument("--test-start", type=str, required=True, help="测试开始日期")
    parser.add_argument("--test-end", type=str, required=True, help="测试结束日期")
    parser.add_argument("--output", type=str, default="tmp/model_comparison_tree.json", help="输出文件路径")
    parser.add_argument("--no-training-features", action="store_true", help="不读取 training_features，强制动态构建")
    parser.add_argument("--training-features-only", action="store_true", help="只读取 training_features，不回退到动态构建")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    train_start = datetime.strptime(args.train_start, "%Y-%m-%d").date()
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date()
    test_start = datetime.strptime(args.test_start, "%Y-%m-%d").date()
    test_end = datetime.strptime(args.test_end, "%Y-%m-%d").date()

    settings = Settings()
    prefer_training_features = not args.no_training_features
    training_features_only = bool(args.training_features_only)

    logger.info("可用回归模型: %s", list(TREE_MODEL_CONFIGS.keys()))
    logger.info("可用排序模型: %s", list(RANK_MODEL_CONFIGS.keys()))

    logger.info("构建训练数据集: %s ~ %s", train_start, train_end)
    train_df = build_dataset(
        settings,
        train_start,
        train_end,
        prefer_training_features=prefer_training_features,
        training_features_only=training_features_only,
    )
    logger.info("训练样本数: %d", len(train_df))

    logger.info("构建测试数据集: %s ~ %s", test_start, test_end)
    test_df = build_dataset(
        settings,
        test_start,
        test_end,
        prefer_training_features=prefer_training_features,
        training_features_only=training_features_only,
    )
    logger.info("测试样本数: %d", len(test_df))

    results = []

    for target in TARGET_COLUMNS:
        logger.info("\n=== 目标变量: %s ===", target)

        # 训练回归模型
        for model_name in TREE_MODEL_CONFIGS.keys():
            logger.info("训练回归模型: %s", model_name)
            model, result = train_model(train_df, target, model_name)

            logger.info("评估模型: %s", model_name)
            result = evaluate_model(model, test_df, target, result)

            logger.info("回测模型: %s", model_name)
            result = backtest_top3(model, test_df, target, result)

            results.append(result)
            logger.info(
                "结果: train=%d, test=%d, mae=%.4f, r2=%.4f, top3_return_5d=%.2f%%",
                result.train_samples,
                result.test_samples,
                result.mae or 0,
                result.r2 or 0,
                (result.backtest_return_5d or 0) * 100,
            )

        # 训练 LambdaRank 模型
        for model_name in RANK_MODEL_CONFIGS.keys():
            logger.info("训练排序模型: %s", model_name)
            model, result = train_rank_model(train_df, target, model_name)

            if model is None:
                logger.warning("跳过模型: %s (训练失败)", model_name)
                continue

            logger.info("评估模型: %s", model_name)
            result = evaluate_model(model, test_df, target, result)

            logger.info("回测模型: %s", model_name)
            result = backtest_top3(model, test_df, target, result)

            results.append(result)
            logger.info(
                "结果: train=%d, test=%d, top3_return_5d=%.2f%%, top3_accuracy=%.2f%%",
                result.train_samples,
                result.test_samples,
                (result.backtest_return_5d or 0) * 100,
                (result.top3_accuracy or 0) * 100,
            )

    # 输出结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "train_period": {"start": str(train_start), "end": str(train_end)},
        "test_period": {"start": str(test_start), "end": str(test_end)},
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "available_regressor_models": list(TREE_MODEL_CONFIGS.keys()),
        "available_rank_models": list(RANK_MODEL_CONFIGS.keys()),
        "results": [
            {
                "model": r.model_name,
                "target": r.target_name,
                "train_samples": r.train_samples,
                "test_samples": r.test_samples,
                "mae": r.mae,
                "rmse": r.rmse,
                "r2": r.r2,
                "accuracy": r.accuracy,
                "precision_pos": r.precision_pos,
                "recall_pos": r.recall_pos,
                "f1_pos": r.f1_pos,
                "backtest_return_1d": r.backtest_return_1d,
                "backtest_return_3d": r.backtest_return_3d,
                "backtest_return_5d": r.backtest_return_5d,
                "top3_accuracy": r.top3_accuracy,
                "feature_importance": dict(sorted(r.feature_importance.items(), key=lambda x: -x[1])[:10]),
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info("结果已保存到: %s", output_path)

    # 打印汇总表格
    print("\n" + "=" * 100)
    print("树模型对比汇总")
    print("=" * 100)
    print(f"{'模型':<15} {'目标':<18} {'MAE':<12} {'R2':<12} {'5日收益':<10} {'Top3准确率':<10}")
    print("-" * 100)
    for r in results:
        mae_str = f"{r.mae:.4f}" if r.mae is not None else "N/A"
        r2_str = f"{r.r2:.4f}" if r.r2 is not None else "N/A"
        print(
            f"{r.model_name:<15} {r.target_name:<18} "
            f"{mae_str:<12} "
            f"{r2_str:<12} "
            f"{(r.backtest_return_5d or 0) * 100:>8.2f}% "
            f"{(r.top3_accuracy or 0) * 100:>8.2f}%"
        )


if __name__ == "__main__":
    main()

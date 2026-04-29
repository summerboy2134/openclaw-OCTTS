"""
超参数调优测试。

用法:
    python -m octts.tools.train_tuned_models \
        --train-start 2025-10-01 --train-end 2026-02-28 \
        --test-start 2026-03-01 --test-end 2026-03-15 \
        --output tmp/model_comparison_tuned.json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from octts.config import Settings
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder
from octts.tools.modeling import save_model_artifact

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

# 目标变量 - 默认保留完整短期收益目标集
TARGET_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
]

ACTUAL_RETURN_COLUMNS = ["return_1d", "return_3d", "return_5d"]

# 导入树模型
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    HAS_RF = True
    HAS_EXTRA_TREES = True
except ImportError:
    HAS_RF = False
    HAS_EXTRA_TREES = False


# 超参数配置组合
HYPERPARAM_CONFIGS = {
    # LightGBM 基线与邻域精调
    "lgbm_default": {
        "model_class": "lgbm",
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    },
    "lgbm_slow": {
        "model_class": "lgbm",
        "params": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    },
    "lgbm_deep_slow": {
        "model_class": "lgbm",
        "params": {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05},
    },
    "lgbm_more_trees": {
        "model_class": "lgbm",
        "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    },
    "lgbm_deep_slow_leaves31": {
        "model_class": "lgbm",
        "params": {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    },
    "lgbm_deep_slow_leaves63": {
        "model_class": "lgbm",
        "params": {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    },
    "lgbm_deep_slow_regularized": {
        "model_class": "lgbm",
        "params": {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    },
    "lgbm_deep_slow_longer": {
        "model_class": "lgbm",
        "params": {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    },
    "lgbm_deep_slow_longer_mild_regularized": {
        "model_class": "lgbm",
        "params": {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 0.5,
        },
    },
    "lgbm_deep_slow_longer_regularized": {
        "model_class": "lgbm",
        "params": {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    },

    # XGBoost 基线与邻域精调
    "xgb_default": {
        "model_class": "xgb",
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
    },
    "xgb_more_trees": {
        "model_class": "xgb",
        "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    },
    "xgb_slow": {
        "model_class": "xgb",
        "params": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    },
    "xgb_slow_subsample": {
        "model_class": "xgb",
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
        },
    },
    "xgb_slow_regularized": {
        "model_class": "xgb",
        "params": {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
    },
    "xgb_slow_deeper": {
        "model_class": "xgb",
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
        },
    },
    "xgb_slow_longer": {
        "model_class": "xgb",
        "params": {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
        },
    },

    # RandomForest 配置（默认不跑，只有显式指定 --model-classes rf 或 --model-names 时才会进入）
    "rf_default": {
        "model_class": "rf",
        "params": {"n_estimators": 100, "max_depth": 5},
    },

    # ExtraTrees：低依赖、金融里常见的树集成对照组
    "et_default": {
        "model_class": "et",
        "params": {"n_estimators": 300, "max_depth": 7, "min_samples_leaf": 5},
    },
    "et_deep": {
        "model_class": "et",
        "params": {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 3},
    },
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
    frame["return_1d_actual"] = frame["return_1d"] if "return_1d" in frame.columns else np.nan
    frame["return_3d_actual"] = frame["return_3d"] if "return_3d" in frame.columns else np.nan
    frame["return_5d_actual"] = frame["return_5d"] if "return_5d" in frame.columns else np.nan
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


def create_model(model_class: str, params: Dict[str, Any]) -> Any:
    """根据配置创建模型实例。"""
    if model_class == "lgbm" and HAS_LIGHTGBM:
        return LGBMRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            num_leaves=params.get("num_leaves", 31),
            min_child_samples=params.get("min_child_samples", 20),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 0.0),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_class == "xgb" and HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            min_child_weight=params.get("min_child_weight", 1),
            reg_alpha=params.get("reg_alpha", 0.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            gamma=params.get("gamma", 0.0),
            random_state=42,
            n_jobs=-1,
        )
    elif model_class == "rf" and HAS_RF:
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=42,
            n_jobs=-1,
        )
    elif model_class == "et" and HAS_EXTRA_TREES:
        return ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 7),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown or unavailable model class: {model_class}")


def train_model(
    train_df: pd.DataFrame,
    target: str,
    model_name: str,
) -> tuple[Any, ModelResult]:
    """训练单个模型。"""
    config = HYPERPARAM_CONFIGS[model_name]
    model = create_model(config["model_class"], config["params"])

    X = train_df[FEATURE_COLUMNS].fillna(0.0)
    y = train_df[target].values

    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]

    model.fit(X, y)

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
    result.mae = float(mean_absolute_error(y_test, y_pred))
    result.rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    result.r2 = float(r2_score(y_test, y_pred))

    return result


def backtest_top3(
    model: Any,
    test_df: pd.DataFrame,
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


def export_model_artifact(
    *,
    settings: Settings,
    model: Any,
    model_name: str,
    target: str,
    train_start: date,
    train_end: date,
) -> Path:
    output_dir = Path(settings.history_dir_path) / "short_term_models"
    model_class = HYPERPARAM_CONFIGS[model_name]["model_class"]
    suffix_map = {
        "lgbm": "lightgbm",
        "xgb": "xgboost",
        "rf": "randomforest",
        "et": "extratrees",
    }
    output_name = (
        f"raw_market_{train_start.strftime('%Y%m')}_{train_end.strftime('%Y%m')}_{target}_{model_name}.{suffix_map.get(model_class, model_class)}.pkl"
    )
    artifact_path = output_dir / output_name
    payload = {
        "model": model,
        "model_name": model_name,
        "model_class": model_class,
        "target": target,
        "feature_columns": list(FEATURE_COLUMNS),
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "schema": "raw_market_training_v1",
    }
    save_model_artifact(artifact_path, payload)
    return artifact_path


def main():
    parser = argparse.ArgumentParser(description="超参数调优测试")
    parser.add_argument("--train-start", type=str, required=True, help="训练开始日期")
    parser.add_argument("--train-end", type=str, required=True, help="训练结束日期")
    parser.add_argument("--test-start", type=str, required=True, help="测试开始日期")
    parser.add_argument("--test-end", type=str, required=True, help="测试结束日期")
    parser.add_argument("--output", type=str, default="tmp/model_comparison_tuned.json", help="输出文件路径")
    parser.add_argument("--targets", type=str, default="return_1d,return_3d,return_5d", help="逗号分隔目标变量，默认 return_1d,return_3d,return_5d")
    parser.add_argument("--model-classes", type=str, default="lgbm,xgb", help="逗号分隔模型类别，默认 lgbm,xgb；可选加 et,rf")
    parser.add_argument("--model-names", type=str, default="", help="逗号分隔具体配置名，为空则按 model-classes 过滤")
    parser.add_argument("--top-k", type=int, default=10, help="汇总时显示前 K 个配置")
    parser.add_argument("--export-artifacts", action="store_true", help="将训练完成的模型导出到 history/short_term_models")
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

    # 过滤可用的配置
    available_configs = {}
    for name, config in HYPERPARAM_CONFIGS.items():
        model_class = config["model_class"]
        if model_class == "lgbm" and HAS_LIGHTGBM:
            available_configs[name] = config
        elif model_class == "xgb" and HAS_XGBOOST:
            available_configs[name] = config
        elif model_class == "rf" and HAS_RF:
            available_configs[name] = config
        elif model_class == "et" and HAS_EXTRA_TREES:
            available_configs[name] = config

    requested_targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    active_targets = [target for target in requested_targets if target in TARGET_COLUMNS] or list(TARGET_COLUMNS)

    requested_model_classes = {value.strip() for value in args.model_classes.split(",") if value.strip()}
    if requested_model_classes:
        available_configs = {
            name: config
            for name, config in available_configs.items()
            if config["model_class"] in requested_model_classes
        }

    requested_model_names = [value.strip() for value in args.model_names.split(",") if value.strip()]
    if requested_model_names:
        requested_name_set = set(requested_model_names)
        available_configs = {
            name: config
            for name, config in available_configs.items()
            if name in requested_name_set
        }

    logger.info("激活目标: %s", active_targets)
    logger.info("可用配置数量: %d", len(available_configs))
    logger.info("配置列表: %s", list(available_configs.keys()))
    if not available_configs:
        raise ValueError("No available hyperparameter configs after filtering. Check --model-classes or --model-names.")

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
    exported_artifacts: list[dict[str, str]] = []

    for target in active_targets:
        logger.info("\n=== 目标变量: %s ===", target)
        for model_name in available_configs.keys():
            logger.info("训练模型: %s", model_name)
            model, result = train_model(train_df, target, model_name)

            logger.info("评估模型: %s", model_name)
            result = evaluate_model(model, test_df, target, result)

            logger.info("回测模型: %s", model_name)
            result = backtest_top3(model, test_df, result)
            if args.export_artifacts:
                artifact_path = export_model_artifact(
                    settings=settings,
                    model=model,
                    model_name=model_name,
                    target=target,
                    train_start=train_start,
                    train_end=train_end,
                )
                exported_artifacts.append(
                    {
                        "model": model_name,
                        "target": target,
                        "path": str(artifact_path),
                    }
                )
                logger.info("已导出模型 artifact: %s", artifact_path)

            results.append(result)
            logger.info(
                "结果: train=%d, test=%d, mae=%.4f, r2=%.4f, top3_return_5d=%.2f%%, accuracy=%.2f%%",
                result.train_samples,
                result.test_samples,
                result.mae or 0,
                result.r2 or 0,
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
        "results": [
            {
                "model": r.model_name,
                "target": r.target_name,
                "train_samples": r.train_samples,
                "test_samples": r.test_samples,
                "mae": r.mae,
                "rmse": r.rmse,
                "r2": r.r2,
                "backtest_return_1d": r.backtest_return_1d,
                "backtest_return_3d": r.backtest_return_3d,
                "backtest_return_5d": r.backtest_return_5d,
                "top3_accuracy": r.top3_accuracy,
                "feature_importance": dict(sorted(r.feature_importance.items(), key=lambda x: -x[1])[:10]),
            }
            for r in results
        ],
        "exported_artifacts": exported_artifacts,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info("结果已保存到: %s", output_path)

    # 打印汇总表格 - 按5日收益排序
    results_sorted = sorted(results, key=lambda r: r.backtest_return_5d or 0, reverse=True)

    print("\n" + "=" * 110)
    print("超参数调优结果汇总（按5日收益排序）")
    print("=" * 110)
    print(f"{'模型':<20} {'目标':<12} {'MAE':<10} {'R2':<10} {'5日收益':<10} {'准确率':<10}")
    print("-" * 110)
    for r in results_sorted[: max(int(args.top_k), 1)]:
        mae_str = f"{r.mae:.4f}" if isinstance(r.mae, (int, float)) else "N/A"
        r2_str = f"{r.r2:.4f}" if isinstance(r.r2, (int, float)) else "N/A"
        print(
            f"{r.model_name:<20} {r.target_name:<12} "
            f"{mae_str:<10} "
            f"{r2_str:<10} "
            f"{(r.backtest_return_5d or 0) * 100:>8.2f}% "
            f"{(r.top3_accuracy or 0) * 100:>8.2f}%"
        )

    # 打印最佳配置
    best = results_sorted[0]
    print("\n" + "=" * 110)
    print(f"最佳配置: {best.model_name} + {best.target_name}")
    print(f"5日收益: {(best.backtest_return_5d or 0) * 100:.2f}%")
    print(f"Top3准确率: {(best.top3_accuracy or 0) * 100:.2f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()

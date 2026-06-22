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
from octts.tools.modeling_weights import (
    ANTI_CHASE_PROFILES,
    SAMPLE_WEIGHT_PROFILES,
    apply_limit_up_downweight,
    build_high_position_acceleration_mask,
    build_high_position_mask,
    build_limit_up_mask,
    build_regime_high_position_risk_mask,
    build_sample_weights,
    clip_return_target,
    count_clipped_return_target,
    numeric_column,
)

logger = logging.getLogger(__name__)

# 原始34个特征 + 新增6个风险交互特征
LEGACY_FEATURE_COLUMNS = [
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

NEW_INTERACTION_FEATURE_COLUMNS = [
    "weak_market_flag", "high_position_flag", "high_position_acceleration_flag",
    "weak_market_high_position_flag", "recent_runup_5d", "turnover_spike_ratio",
    "prev_day_limit_up", "prev_day_limit_open_times", "prev_day_limit_first_time",
    "prev_day_limit_last_time", "prev_day_limit_amount", "prev_day_fd_amount",
    "prev_day_limit_times", "prev_day_up_stat_success", "prev_day_up_stat_total",
    "prev_day_up_stat_ratio", "prev_day_one_word_limit_flag",
    "limit_chase_failure_risk_score",
]

FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS + NEW_INTERACTION_FEATURE_COLUMNS
BEST35_FEATURE_COLUMNS = [
    "market_return_3d",
    "market_return_1d",
    "market_return_5d",
    "market_up_ratio_3d_avg",
    "market_up_ratio_1d",
    "close",
    "market_cap",
    "new_low_gap_20d",
    "avg_turnover_rate_5d",
    "pb",
    "amount",
    "pct_change",
    "max_drawdown_10d_past",
    "volatility_10d",
    "stock_vs_market_return_1d",
    "recent_runup_5d",
    "stock_vs_market_return_10d",
    "volatility_5d",
    "stock_vs_market_return_3d",
    "pct_change_rank_pct",
    "turnover_rate",
    "vol",
    "avg_volume_ratio_5d",
    "close_to_ma5",
    "pe_ttm",
    "market_up_days_5d",
    "new_high_gap_20d",
    "close_to_ma10",
    "volume_ratio",
    "turnover_rate_rank_pct",
    "price_position_20d",
    "turnover_spike_ratio",
    "volume_ratio_rank_pct",
    "price_position_10d",
    "up_days_5d",
]

# 目标变量 - 默认保留完整短期收益目标集
TARGET_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "vs_market_1d",
    "vs_market_3d",
    "vs_market_5d",
]

ACTUAL_RETURN_COLUMNS = ["return_1d", "return_3d", "return_5d"]

RISK_ADJUSTED_SCORE_PENALTIES = {
    "top3_high_position": 0.035,
    "weak_market_top3_high_position": 0.055,
    "top3_high_position_acceleration": 0.08,
    "top3_loss_5pct_3d": 0.06,
}

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
    sample_weight_mode: str = "none"
    anti_chase_profile: str = "default"
    sample_weight_profile: str = "balanced"
    train_anti_chase_rows: int = 0
    train_anti_chase_bad_rows: int = 0
    train_regime_risk_rows: int = 0
    train_regime_risk_bad_rows: int = 0
    train_avg_sample_weight: Optional[float] = None
    limit_up_sample_mode: str = "none"
    limit_up_pct_threshold: float = 9.5
    limit_up_sample_weight: Optional[float] = None
    train_limit_up_sample_count: int = 0
    limit_up_samples_dropped: int = 0
    return_clip_enabled: bool = False
    return_clip_low: Optional[float] = None
    return_clip_high: Optional[float] = None
    return_clip_count: int = 0
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    backtest_return_1d: Optional[float] = None
    backtest_return_3d: Optional[float] = None
    backtest_return_5d: Optional[float] = None
    backtest_vs_market_1d: Optional[float] = None
    backtest_vs_market_3d: Optional[float] = None
    backtest_vs_market_5d: Optional[float] = None
    top3_accuracy: Optional[float] = None
    top3_return_1d_win_rate: Optional[float] = None
    top3_return_3d_win_rate: Optional[float] = None
    top3_vs_market_3d_win_rate: Optional[float] = None
    top3_loss_3d_rate: Optional[float] = None
    top3_loss_5pct_3d_rate: Optional[float] = None
    top3_high_position_acceleration_exposure: Optional[float] = None
    top20_high_position_acceleration_exposure: Optional[float] = None
    top3_high_position_exposure: Optional[float] = None
    weak_market_top3_count: int = 0
    weak_market_top3_return_3d: Optional[float] = None
    weak_market_top3_vs_market_3d: Optional[float] = None
    weak_market_top3_vs_market_3d_win_rate: Optional[float] = None
    weak_market_top3_loss_3d_rate: Optional[float] = None
    weak_market_top3_loss_5pct_3d_rate: Optional[float] = None
    weak_market_top3_high_position_exposure: Optional[float] = None
    weak_market_top3_high_position_acceleration_exposure: Optional[float] = None
    top3_count: int = 0
    top20_count: int = 0
    risk_adjusted_selection_score: Optional[float] = None
    feature_subset_name: str = "all"
    feature_count: int = 0
    feature_columns: List[str] = field(default_factory=list)
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
    *,
    feature_columns: Optional[List[str]] = None,
    feature_subset_name: str = "all",
    sample_weight_mode: str = "none",
    anti_chase_profile: str = "default",
    sample_weight_profile: str = "balanced",
    limit_up_sample_mode: str = "none",
    limit_up_pct_threshold: float = 9.5,
    limit_up_sample_weight: float = 0.1,
    enable_return_clip: bool = False,
    return_clip_low: float = -0.15,
    return_clip_high: float = 0.20,
) -> tuple[Any, ModelResult]:
    """训练单个模型。"""
    config = HYPERPARAM_CONFIGS[model_name]
    model = create_model(config["model_class"], config["params"])
    active_feature_columns = _normalize_feature_columns(feature_columns)

    working_df = train_df.copy()
    limit_up_mask_all = build_limit_up_mask(working_df, threshold=limit_up_pct_threshold)
    limit_up_sample_count = int(limit_up_mask_all.sum())
    limit_up_samples_dropped = 0
    if limit_up_sample_mode == "drop" and limit_up_sample_count:
        working_df = working_df.loc[~limit_up_mask_all].copy()
        limit_up_samples_dropped = limit_up_sample_count

    X = working_df[active_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_series = pd.to_numeric(working_df[target], errors="coerce")
    y_before_clip = y_series.copy()
    y_series = clip_return_target(
        y_series,
        target_name=target,
        enabled=enable_return_clip,
        lower=return_clip_low,
        upper=return_clip_high,
    )
    return_clip_count = count_clipped_return_target(y_before_clip, y_series)
    y = y_series.values

    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    valid_train_df = working_df.loc[valid_mask].copy()
    sample_weight = build_sample_weights(
        valid_train_df,
        target_values=pd.Series(y, index=valid_train_df.index),
        mode=sample_weight_mode,
        profile=anti_chase_profile,
        weight_profile=sample_weight_profile,
    )
    if limit_up_sample_mode == "downweight":
        sample_weight = apply_limit_up_downweight(
            sample_weight,
            valid_train_df,
            threshold=limit_up_pct_threshold,
            weight=limit_up_sample_weight,
        )
    anti_chase_mask = build_high_position_acceleration_mask(valid_train_df, anti_chase_profile)
    regime_risk_mask = build_regime_high_position_risk_mask(valid_train_df, anti_chase_profile)

    if sample_weight is not None:
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            logger.warning("Model %s does not accept sample_weight; falling back to unweighted fit", model_name)
            model.fit(X, y)
    else:
        model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        importance = dict(zip(active_feature_columns, model.feature_importances_.tolist()))
    else:
        importance = {}

    result = ModelResult(
        model_name=model_name,
        target_name=target,
        train_samples=len(y),
        test_samples=0,
        sample_weight_mode=sample_weight_mode,
        anti_chase_profile=anti_chase_profile,
        sample_weight_profile=sample_weight_profile,
        train_anti_chase_rows=int(anti_chase_mask.sum()),
        train_anti_chase_bad_rows=int(
            (anti_chase_mask & (pd.Series(y, index=valid_train_df.index) <= 0)).sum()
        ),
        train_regime_risk_rows=int(regime_risk_mask.sum()),
        train_regime_risk_bad_rows=int(
            (regime_risk_mask & (pd.Series(y, index=valid_train_df.index) <= 0)).sum()
        ),
        train_avg_sample_weight=float(sample_weight.mean()) if sample_weight is not None and len(sample_weight) else None,
        limit_up_sample_mode=limit_up_sample_mode,
        limit_up_pct_threshold=limit_up_pct_threshold,
        limit_up_sample_weight=limit_up_sample_weight if limit_up_sample_mode == "downweight" else None,
        train_limit_up_sample_count=int(
            build_limit_up_mask(valid_train_df, threshold=limit_up_pct_threshold).sum()
        ),
        limit_up_samples_dropped=limit_up_samples_dropped,
        return_clip_enabled=bool(enable_return_clip),
        return_clip_low=return_clip_low if enable_return_clip else None,
        return_clip_high=return_clip_high if enable_return_clip else None,
        return_clip_count=return_clip_count,
        feature_subset_name=feature_subset_name,
        feature_count=len(active_feature_columns),
        feature_columns=list(active_feature_columns),
        feature_importance=importance,
    )

    return model, result


def _normalize_feature_columns(feature_columns: Optional[List[str]]) -> List[str]:
    if not feature_columns:
        return list(FEATURE_COLUMNS)
    valid_feature_columns = [column for column in feature_columns if column in FEATURE_COLUMNS]
    if not valid_feature_columns:
        raise ValueError("feature_columns resolved to an empty set")
    return list(dict.fromkeys(valid_feature_columns))


def _build_sample_weights(
    frame: pd.DataFrame,
    *,
    target_values: pd.Series,
    mode: str,
    profile: str,
    weight_profile: str,
) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    if mode not in {"anti_chase", "regime_anti_chase"}:
        raise ValueError(f"Unsupported sample_weight_mode: {mode}")
    weights_config = SAMPLE_WEIGHT_PROFILES.get(weight_profile)
    if weights_config is None:
        raise ValueError(f"Unsupported sample_weight_profile: {weight_profile}")

    weights = pd.Series(1.0, index=frame.index)
    anti_chase_mask = _build_high_position_acceleration_mask(frame, profile)
    high_position_mask = _build_high_position_mask(frame, profile)
    bad_forward_mask = anti_chase_mask & (target_values <= 0)
    strong_forward_mask = anti_chase_mask & (target_values >= 0.05)
    positive_forward_mask = anti_chase_mask & (target_values > 0) & ~strong_forward_mask

    # Emphasize failed chase patterns while damping runaway high-position winners
    # that would otherwise dominate daily Top3 selection.
    weights.loc[bad_forward_mask] = weights_config["anti_bad"]
    weights.loc[positive_forward_mask] = weights_config["anti_positive"]
    weights.loc[strong_forward_mask] = weights_config["anti_strong"]

    high_position_bad_mask = high_position_mask & ~anti_chase_mask & (target_values <= 0)
    high_position_strong_mask = high_position_mask & ~anti_chase_mask & (target_values >= 0.05)
    high_position_positive_mask = (
        high_position_mask
        & ~anti_chase_mask
        & (target_values > 0)
        & ~high_position_strong_mask
    )
    weights.loc[high_position_bad_mask] = np.maximum(
        weights.loc[high_position_bad_mask],
        weights_config["high_position_bad"],
    )
    weights.loc[high_position_positive_mask] = np.minimum(
        weights.loc[high_position_positive_mask],
        weights_config["high_position_positive"],
    )
    weights.loc[high_position_strong_mask] = np.minimum(
        weights.loc[high_position_strong_mask],
        weights_config["high_position_strong"],
    )

    if mode == "regime_anti_chase":
        regime_risk_mask = _build_regime_high_position_risk_mask(frame, profile)
        regime_bad_mask = regime_risk_mask & (target_values <= 0)
        regime_strong_mask = regime_risk_mask & (target_values >= 0.05)
        regime_positive_mask = regime_risk_mask & (target_values > 0) & ~regime_strong_mask
        weights.loc[regime_bad_mask] = np.maximum(weights.loc[regime_bad_mask], weights_config["regime_bad"])
        weights.loc[regime_positive_mask] = np.minimum(weights.loc[regime_positive_mask], weights_config["regime_positive"])
        weights.loc[regime_strong_mask] = np.minimum(weights.loc[regime_strong_mask], weights_config["regime_strong"])
    return weights.to_numpy(dtype=float)


def _build_high_position_acceleration_mask(frame: pd.DataFrame, profile: str = "default") -> pd.Series:
    thresholds = ANTI_CHASE_PROFILES.get(profile)
    if thresholds is None:
        raise ValueError(f"Unsupported anti_chase_profile: {profile}")

    if "high_position_acceleration_flag" in frame.columns:
        stored_flag = _numeric_column(frame, "high_position_acceleration_flag")
        if stored_flag.notna().any():
            return (stored_flag > 0).fillna(False)

    price_position = _numeric_column(frame, "price_position_20d")
    pct_change = _numeric_column(frame, "pct_change")
    volume_ratio = _numeric_column(frame, "volume_ratio")
    turnover_rate = _numeric_column(frame, "turnover_rate")
    recent_runup_5d = _numeric_column(frame, "recent_runup_5d")
    turnover_spike_ratio = _numeric_column(frame, "turnover_spike_ratio")

    high_position = price_position >= thresholds["price_position"]
    same_day_acceleration = (
        (pct_change >= thresholds["pct_change"])
        & (volume_ratio >= thresholds["volume_ratio"])
        & (turnover_rate >= thresholds["turnover_rate"])
    )
    runup_acceleration = (
        (recent_runup_5d >= thresholds["return_5d_past"] * 100.0)
        & (volume_ratio >= thresholds["volume_ratio"])
    )
    turnover_acceleration = (
        (recent_runup_5d >= thresholds["return_5d_past"] * 100.0)
        & (turnover_spike_ratio >= 1.6)
    )
    return (high_position & (same_day_acceleration | runup_acceleration | turnover_acceleration)).fillna(False)


def _build_high_position_mask(frame: pd.DataFrame, profile: str = "default") -> pd.Series:
    thresholds = ANTI_CHASE_PROFILES.get(profile)
    if thresholds is None:
        raise ValueError(f"Unsupported anti_chase_profile: {profile}")
    if "high_position_flag" in frame.columns:
        stored_flag = _numeric_column(frame, "high_position_flag")
        if stored_flag.notna().any():
            return (stored_flag > 0).fillna(False)
    return (_numeric_column(frame, "price_position_20d") >= thresholds["price_position"]).fillna(False)


def _build_weak_market_mask(frame: pd.DataFrame) -> pd.Series:
    market_return_1d = _numeric_column(frame, "market_return_1d")
    market_return_3d = _numeric_column(frame, "market_return_3d")
    market_up_ratio_1d = _numeric_column(frame, "market_up_ratio_1d")
    market_up_ratio_3d_avg = _numeric_column(frame, "market_up_ratio_3d_avg")
    market_up_days_5d = _numeric_column(frame, "market_up_days_5d")
    weak_mask = (
        (market_return_1d < 0)
        | (market_return_3d < 0)
        | (market_up_ratio_1d < 0.45)
        | (market_up_ratio_3d_avg < 0.45)
        | (market_up_days_5d <= 2)
    )
    return weak_mask.fillna(False)


def _build_regime_high_position_risk_mask(frame: pd.DataFrame, profile: str = "default") -> pd.Series:
    if "weak_market_high_position_flag" in frame.columns:
        stored_flag = _numeric_column(frame, "weak_market_high_position_flag")
        if stored_flag.notna().any():
            return (stored_flag > 0).fillna(False)
    weak_market = _build_weak_market_mask(frame)
    high_position = _build_high_position_mask(frame, profile)
    recent_strength = (
        (_numeric_column(frame, "recent_runup_5d") > 0)
        | (_numeric_column(frame, "pct_change") > 0)
        | (_numeric_column(frame, "price_position_20d") >= 0.95)
    )
    return (weak_market & high_position & recent_strength).fillna(False)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def evaluate_model(
    model: Any,
    test_df: pd.DataFrame,
    target: str,
    result: ModelResult,
) -> ModelResult:
    """评估模型效果。"""
    feature_columns = _normalize_feature_columns(result.feature_columns)
    X_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
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
    *,
    anti_chase_profile: str = "default",
) -> ModelResult:
    """回测：每天选预测最高的3只股票，计算实际收益。"""
    test_df = test_df.copy()
    feature_columns = _normalize_feature_columns(result.feature_columns)
    test_df["pred"] = model.predict(test_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0))

    daily_returns_1d = []
    daily_returns_3d = []
    daily_returns_5d = []
    daily_vs_market_1d = []
    daily_vs_market_3d = []
    daily_vs_market_5d = []
    top3_anti_chase_flags: List[bool] = []
    top20_anti_chase_flags: List[bool] = []
    top3_high_position_flags: List[bool] = []
    weak_market_returns_3d: List[float] = []
    weak_market_vs_market_3d: List[float] = []
    weak_market_anti_chase_flags: List[bool] = []
    weak_market_high_position_flags: List[bool] = []

    for trade_date in test_df["trade_date"].unique():
        day_df = test_df[test_df["trade_date"] == trade_date]
        if len(day_df) < 3:
            continue

        top3 = day_df.nlargest(3, "pred")
        top20 = day_df.nlargest(min(20, len(day_df)), "pred")
        weak_market_day = bool(_build_weak_market_mask(day_df).any())
        top3_anti_chase_flags.extend(_build_high_position_acceleration_mask(top3, anti_chase_profile).tolist())
        top20_anti_chase_flags.extend(_build_high_position_acceleration_mask(top20, anti_chase_profile).tolist())
        top3_high_position_flags.extend(_build_high_position_mask(top3, anti_chase_profile).tolist())
        top3_anti_flags_for_day = _build_high_position_acceleration_mask(top3, anti_chase_profile).tolist()
        top3_high_flags_for_day = _build_high_position_mask(top3, anti_chase_profile).tolist()
        if weak_market_day:
            weak_market_anti_chase_flags.extend(top3_anti_flags_for_day)
            weak_market_high_position_flags.extend(top3_high_flags_for_day)

        for _, row in top3.iterrows():
            if pd.notna(row["return_1d_actual"]):
                daily_returns_1d.append(row["return_1d_actual"])
            if pd.notna(row["return_3d_actual"]):
                daily_returns_3d.append(row["return_3d_actual"])
                if weak_market_day:
                    weak_market_returns_3d.append(row["return_3d_actual"])
            if pd.notna(row["return_5d_actual"]):
                daily_returns_5d.append(row["return_5d_actual"])
            if "vs_market_1d" in row and pd.notna(row["vs_market_1d"]):
                daily_vs_market_1d.append(row["vs_market_1d"])
            if "vs_market_3d" in row and pd.notna(row["vs_market_3d"]):
                daily_vs_market_3d.append(row["vs_market_3d"])
                if weak_market_day:
                    weak_market_vs_market_3d.append(row["vs_market_3d"])
            if "vs_market_5d" in row and pd.notna(row["vs_market_5d"]):
                daily_vs_market_5d.append(row["vs_market_5d"])

    if daily_returns_1d:
        result.backtest_return_1d = float(np.mean(daily_returns_1d))
        result.top3_return_1d_win_rate = float(np.mean([value > 0 for value in daily_returns_1d]))
    if daily_returns_3d:
        result.backtest_return_3d = float(np.mean(daily_returns_3d))
        result.top3_return_3d_win_rate = float(np.mean([value > 0 for value in daily_returns_3d]))
        result.top3_loss_3d_rate = float(np.mean([value < 0 for value in daily_returns_3d]))
        result.top3_loss_5pct_3d_rate = float(np.mean([value <= -0.05 for value in daily_returns_3d]))
    if daily_returns_5d:
        result.backtest_return_5d = float(np.mean(daily_returns_5d))
    if daily_vs_market_1d:
        result.backtest_vs_market_1d = float(np.mean(daily_vs_market_1d))
    if daily_vs_market_3d:
        result.backtest_vs_market_3d = float(np.mean(daily_vs_market_3d))
        result.top3_vs_market_3d_win_rate = float(np.mean([value > 0 for value in daily_vs_market_3d]))
    if daily_vs_market_5d:
        result.backtest_vs_market_5d = float(np.mean(daily_vs_market_5d))
    if top3_anti_chase_flags:
        result.top3_count = len(top3_anti_chase_flags)
        result.top3_high_position_acceleration_exposure = float(np.mean(top3_anti_chase_flags))
    if top3_high_position_flags:
        result.top3_high_position_exposure = float(np.mean(top3_high_position_flags))
    if top20_anti_chase_flags:
        result.top20_count = len(top20_anti_chase_flags)
        result.top20_high_position_acceleration_exposure = float(np.mean(top20_anti_chase_flags))
    if weak_market_returns_3d:
        result.weak_market_top3_count = len(weak_market_returns_3d)
        result.weak_market_top3_return_3d = float(np.mean(weak_market_returns_3d))
        result.weak_market_top3_loss_3d_rate = float(np.mean([value < 0 for value in weak_market_returns_3d]))
        result.weak_market_top3_loss_5pct_3d_rate = float(np.mean([value <= -0.05 for value in weak_market_returns_3d]))
    if weak_market_vs_market_3d:
        result.weak_market_top3_vs_market_3d = float(np.mean(weak_market_vs_market_3d))
        result.weak_market_top3_vs_market_3d_win_rate = float(np.mean([value > 0 for value in weak_market_vs_market_3d]))
    if weak_market_high_position_flags:
        result.weak_market_top3_high_position_exposure = float(np.mean(weak_market_high_position_flags))
    if weak_market_anti_chase_flags:
        result.weak_market_top3_high_position_acceleration_exposure = float(np.mean(weak_market_anti_chase_flags))

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

    result.risk_adjusted_selection_score = _calculate_risk_adjusted_selection_score(result)
    return result


def _calculate_risk_adjusted_selection_score(result: ModelResult) -> float:
    base_score = float(result.backtest_vs_market_3d or 0.0)
    risk_penalty = (
        float(result.top3_high_position_exposure or 0.0) * RISK_ADJUSTED_SCORE_PENALTIES["top3_high_position"]
        + float(result.weak_market_top3_high_position_exposure or 0.0)
        * RISK_ADJUSTED_SCORE_PENALTIES["weak_market_top3_high_position"]
        + float(result.top3_high_position_acceleration_exposure or 0.0)
        * RISK_ADJUSTED_SCORE_PENALTIES["top3_high_position_acceleration"]
        + float(result.top3_loss_5pct_3d_rate or 0.0)
        * RISK_ADJUSTED_SCORE_PENALTIES["top3_loss_5pct_3d"]
    )
    return float(base_score - risk_penalty)


def _parse_feature_topn_values(raw_value: str) -> List[int | str]:
    values: List[int | str] = []
    for item in raw_value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "all":
            values.append("all")
            continue
        number = int(token)
        if number <= 0:
            raise ValueError(f"--feature-topn values must be positive or all: {raw_value}")
        values.append(number)
    return values or ["all"]


def _select_feature_importance_order(
    train_df: pd.DataFrame,
    *,
    target: str,
    selector_model_name: str,
    sample_weight_mode: str,
    anti_chase_profile: str,
    sample_weight_profile: str,
    limit_up_sample_mode: str,
    limit_up_pct_threshold: float,
    limit_up_sample_weight: float,
    enable_return_clip: bool,
    return_clip_low: float,
    return_clip_high: float,
) -> List[str]:
    selector_model, selector_result = train_model(
        train_df,
        target,
        selector_model_name,
        feature_columns=list(FEATURE_COLUMNS),
        feature_subset_name="selector_all",
        sample_weight_mode=sample_weight_mode,
        anti_chase_profile=anti_chase_profile,
        sample_weight_profile=sample_weight_profile,
        limit_up_sample_mode=limit_up_sample_mode,
        limit_up_pct_threshold=limit_up_pct_threshold,
        limit_up_sample_weight=limit_up_sample_weight,
        enable_return_clip=enable_return_clip,
        return_clip_low=return_clip_low,
        return_clip_high=return_clip_high,
    )
    del selector_model
    if not selector_result.feature_importance:
        logger.warning("特征选择模型 %s 未输出 feature_importance，回退到原始特征顺序", selector_model_name)
        return list(FEATURE_COLUMNS)
    importance_order = [
        feature
        for feature, _ in sorted(
            selector_result.feature_importance.items(),
            key=lambda item: (-float(item[1] or 0.0), item[0]),
        )
    ]
    return importance_order or list(FEATURE_COLUMNS)


def _build_feature_subsets(
    *,
    mode: str,
    topn_values: List[int | str],
    importance_order: Optional[List[str]],
) -> List[tuple[str, List[str]]]:
    if mode == "none":
        return [("all", list(FEATURE_COLUMNS))]
    if mode == "best35":
        return [("best35", list(BEST35_FEATURE_COLUMNS))]
    if mode == "focused_combo":
        return [
            ("legacy34", list(LEGACY_FEATURE_COLUMNS)),
            ("legacy34_plus_recent_runup_5d", list(LEGACY_FEATURE_COLUMNS) + ["recent_runup_5d"]),
            ("legacy34_plus_turnover_spike_ratio", list(LEGACY_FEATURE_COLUMNS) + ["turnover_spike_ratio"]),
            (
                "legacy34_plus_recent_runup_5d_turnover_spike_ratio",
                list(LEGACY_FEATURE_COLUMNS) + ["recent_runup_5d", "turnover_spike_ratio"],
            ),
        ]
    if mode == "ablation":
        subsets: List[tuple[str, List[str]]] = [
            ("legacy34", list(LEGACY_FEATURE_COLUMNS)),
            ("legacy34_plus_all6", list(FEATURE_COLUMNS)),
        ]
        subsets.extend(
            (
                f"legacy34_plus_{feature}",
                list(LEGACY_FEATURE_COLUMNS) + [feature],
            )
            for feature in NEW_INTERACTION_FEATURE_COLUMNS
        )
        return subsets
    if mode != "topn":
        raise ValueError(f"Unsupported feature_subset_mode: {mode}")

    ordered_features = list(importance_order or FEATURE_COLUMNS)
    subsets: List[tuple[str, List[str]]] = []
    seen_labels: set[str] = set()
    for value in topn_values:
        if value == "all":
            label = "all"
            columns = list(FEATURE_COLUMNS)
        else:
            topn = min(int(value), len(ordered_features))
            label = f"top{topn}"
            columns = ordered_features[:topn]
        if label in seen_labels:
            continue
        seen_labels.add(label)
        subsets.append((label, columns))
    return subsets or [("all", list(FEATURE_COLUMNS))]


def export_model_artifact(
    *,
    settings: Settings,
    model: Any,
    model_name: str,
    target: str,
    train_start: date,
    train_end: date,
    feature_columns: Optional[List[str]] = None,
    feature_subset_name: str = "all",
    sample_weight_mode: str = "none",
    anti_chase_profile: str = "default",
    sample_weight_profile: str = "balanced",
    limit_up_sample_mode: str = "none",
    limit_up_pct_threshold: float = 9.5,
    limit_up_sample_weight: Optional[float] = None,
    return_clip_enabled: bool = False,
    return_clip_low: Optional[float] = None,
    return_clip_high: Optional[float] = None,
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
        f"raw_market_{train_start.strftime('%Y%m')}_{train_end.strftime('%Y%m')}_{target}_{model_name}_{feature_subset_name}.{suffix_map.get(model_class, model_class)}.pkl"
    )
    artifact_path = output_dir / output_name
    active_feature_columns = _normalize_feature_columns(feature_columns)
    payload = {
        "model": model,
        "model_name": model_name,
        "model_class": model_class,
        "target": target,
        "feature_columns": list(active_feature_columns),
        "feature_subset_name": feature_subset_name,
        "feature_count": len(active_feature_columns),
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "sample_weight_mode": sample_weight_mode,
        "anti_chase_profile": anti_chase_profile,
        "sample_weight_profile": sample_weight_profile,
        "limit_up_sample_mode": limit_up_sample_mode,
        "limit_up_pct_threshold": limit_up_pct_threshold,
        "limit_up_sample_weight": limit_up_sample_weight,
        "return_clip_enabled": return_clip_enabled,
        "return_clip_low": return_clip_low,
        "return_clip_high": return_clip_high,
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
    parser.add_argument("--sample-weight-mode", choices=["none", "anti_chase", "regime_anti_chase"], default="none", help="样本权重模式")
    parser.add_argument("--anti-chase-profile", choices=sorted(ANTI_CHASE_PROFILES.keys()), default="default", help="高位加速识别阈值配置")
    parser.add_argument("--sample-weight-profile", choices=sorted(SAMPLE_WEIGHT_PROFILES.keys()), default="balanced", help="样本权重强度配置")
    parser.add_argument("--limit-up-sample-mode", choices=["none", "drop", "downweight"], default="none", help="涨停/近涨停训练样本处理方式")
    parser.add_argument("--limit-up-pct-threshold", type=float, default=9.5, help="识别涨停/近涨停样本的 pct_change 阈值")
    parser.add_argument("--limit-up-sample-weight", type=float, default=0.1, help="limit-up-sample-mode=downweight 时的权重乘数")
    parser.add_argument("--enable-return-clip", action="store_true", help="对 return_* 回归目标做极端值裁剪")
    parser.add_argument("--return-clip-low", type=float, default=-0.15, help="return_* 目标裁剪下界")
    parser.add_argument("--return-clip-high", type=float, default=0.20, help="return_* 目标裁剪上界")
    parser.add_argument("--feature-subset-mode", choices=["none", "best35", "topn", "ablation", "focused_combo"], default="none", help="特征子集搜索模式；best35 使用原始34+recent_runup_5d，topn 按重要性测试前N子集，ablation 对比原始34与新增6个特征，focused_combo 仅比较 recent_runup_5d/turnover_spike_ratio 组合")
    parser.add_argument("--feature-topn", type=str, default="10,15,20,25,30,all", help="topn 模式下测试的特征数量列表，逗号分隔，可包含 all")
    parser.add_argument("--feature-selector-model", type=str, default="lgbm_more_trees", help="topn 模式下用于生成特征重要性排序的模型配置名")
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
    invalid_targets = [target for target in requested_targets if target not in TARGET_COLUMNS]
    if invalid_targets:
        raise ValueError(f"Unsupported targets: {invalid_targets}. Available targets: {TARGET_COLUMNS}")
    active_targets = requested_targets or list(TARGET_COLUMNS)

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

    feature_topn_values = _parse_feature_topn_values(args.feature_topn)
    if args.feature_subset_mode == "topn" and args.feature_selector_model not in HYPERPARAM_CONFIGS:
        raise ValueError(f"Unsupported feature selector model: {args.feature_selector_model}")

    logger.info("激活目标: %s", active_targets)
    logger.info(
        "样本权重模式: %s, anti_chase_profile=%s, sample_weight_profile=%s",
        args.sample_weight_mode,
        args.anti_chase_profile,
        args.sample_weight_profile,
    )
    logger.info(
        "训练样本追高处理: limit_up_mode=%s, limit_up_threshold=%.2f, limit_up_weight=%.4f, return_clip=%s [%.4f, %.4f]",
        args.limit_up_sample_mode,
        args.limit_up_pct_threshold,
        args.limit_up_sample_weight,
        bool(args.enable_return_clip),
        args.return_clip_low,
        args.return_clip_high,
    )
    logger.info(
        "特征子集模式: %s, feature_topn=%s, selector=%s",
        args.feature_subset_mode,
        feature_topn_values,
        args.feature_selector_model,
    )
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
        importance_order: Optional[List[str]] = None
        if args.feature_subset_mode == "topn":
            logger.info("训练特征选择器: target=%s, model=%s", target, args.feature_selector_model)
            importance_order = _select_feature_importance_order(
                train_df,
                target=target,
                selector_model_name=args.feature_selector_model,
                sample_weight_mode=args.sample_weight_mode,
                anti_chase_profile=args.anti_chase_profile,
                sample_weight_profile=args.sample_weight_profile,
                limit_up_sample_mode=args.limit_up_sample_mode,
                limit_up_pct_threshold=args.limit_up_pct_threshold,
                limit_up_sample_weight=args.limit_up_sample_weight,
                enable_return_clip=bool(args.enable_return_clip),
                return_clip_low=args.return_clip_low,
                return_clip_high=args.return_clip_high,
            )
            logger.info("特征重要性Top10: %s", importance_order[:10])
        feature_subsets = _build_feature_subsets(
            mode=args.feature_subset_mode,
            topn_values=feature_topn_values,
            importance_order=importance_order,
        )
        logger.info("特征子集列表: %s", [(name, len(columns)) for name, columns in feature_subsets])

        for feature_subset_name, feature_columns in feature_subsets:
            for model_name in available_configs.keys():
                logger.info("训练模型: %s, feature_subset=%s, feature_count=%d", model_name, feature_subset_name, len(feature_columns))
                model, result = train_model(
                    train_df,
                    target,
                    model_name,
                    feature_columns=feature_columns,
                    feature_subset_name=feature_subset_name,
                    sample_weight_mode=args.sample_weight_mode,
                    anti_chase_profile=args.anti_chase_profile,
                    sample_weight_profile=args.sample_weight_profile,
                    limit_up_sample_mode=args.limit_up_sample_mode,
                    limit_up_pct_threshold=args.limit_up_pct_threshold,
                    limit_up_sample_weight=args.limit_up_sample_weight,
                    enable_return_clip=bool(args.enable_return_clip),
                    return_clip_low=args.return_clip_low,
                    return_clip_high=args.return_clip_high,
                )

                logger.info("评估模型: %s, feature_subset=%s", model_name, feature_subset_name)
                result = evaluate_model(model, test_df, target, result)

                logger.info("回测模型: %s, feature_subset=%s", model_name, feature_subset_name)
                result = backtest_top3(model, test_df, result, anti_chase_profile=args.anti_chase_profile)
                if args.export_artifacts:
                    artifact_path = export_model_artifact(
                        settings=settings,
                        model=model,
                        model_name=model_name,
                        target=target,
                        train_start=train_start,
                        train_end=train_end,
                        feature_columns=feature_columns,
                        feature_subset_name=feature_subset_name,
                        sample_weight_mode=args.sample_weight_mode,
                        anti_chase_profile=args.anti_chase_profile,
                        sample_weight_profile=args.sample_weight_profile,
                        limit_up_sample_mode=args.limit_up_sample_mode,
                        limit_up_pct_threshold=args.limit_up_pct_threshold,
                        limit_up_sample_weight=args.limit_up_sample_weight if args.limit_up_sample_mode == "downweight" else None,
                        return_clip_enabled=bool(args.enable_return_clip),
                        return_clip_low=args.return_clip_low if args.enable_return_clip else None,
                        return_clip_high=args.return_clip_high if args.enable_return_clip else None,
                    )
                    exported_artifacts.append(
                        {
                            "model": model_name,
                            "target": target,
                            "feature_subset_name": feature_subset_name,
                            "feature_count": str(len(feature_columns)),
                            "sample_weight_mode": args.sample_weight_mode,
                            "sample_weight_profile": args.sample_weight_profile,
                            "anti_chase_profile": args.anti_chase_profile,
                            "limit_up_sample_mode": args.limit_up_sample_mode,
                            "limit_up_pct_threshold": str(args.limit_up_pct_threshold),
                            "limit_up_sample_weight": str(args.limit_up_sample_weight) if args.limit_up_sample_mode == "downweight" else "",
                            "return_clip_enabled": str(bool(args.enable_return_clip)),
                            "return_clip_low": str(args.return_clip_low) if args.enable_return_clip else "",
                            "return_clip_high": str(args.return_clip_high) if args.enable_return_clip else "",
                            "path": str(artifact_path),
                        }
                    )
                    logger.info("已导出模型 artifact: %s", artifact_path)

                results.append(result)
                logger.info(
                    "结果: subset=%s, features=%d, train=%d, test=%d, mae=%.4f, r2=%.4f, top3_return_5d=%.2f%%, accuracy=%.2f%%",
                    result.feature_subset_name,
                    result.feature_count,
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
        "sample_weight_mode": args.sample_weight_mode,
        "anti_chase_profile": args.anti_chase_profile,
        "sample_weight_profile": args.sample_weight_profile,
        "limit_up_sample_mode": args.limit_up_sample_mode,
        "limit_up_pct_threshold": args.limit_up_pct_threshold,
        "limit_up_sample_weight": args.limit_up_sample_weight if args.limit_up_sample_mode == "downweight" else None,
        "return_clip_enabled": bool(args.enable_return_clip),
        "return_clip_low": args.return_clip_low if args.enable_return_clip else None,
        "return_clip_high": args.return_clip_high if args.enable_return_clip else None,
        "feature_subset_mode": args.feature_subset_mode,
        "feature_topn": feature_topn_values,
        "feature_selector_model": args.feature_selector_model,
        "results": [
            {
                "model": r.model_name,
                "target": r.target_name,
                "feature_subset_name": r.feature_subset_name,
                "feature_count": r.feature_count,
                "feature_columns": r.feature_columns,
                "sample_weight_mode": r.sample_weight_mode,
                "anti_chase_profile": r.anti_chase_profile,
                "sample_weight_profile": r.sample_weight_profile,
                "limit_up_sample_mode": r.limit_up_sample_mode,
                "limit_up_pct_threshold": r.limit_up_pct_threshold,
                "limit_up_sample_weight": r.limit_up_sample_weight,
                "train_limit_up_sample_count": r.train_limit_up_sample_count,
                "limit_up_samples_dropped": r.limit_up_samples_dropped,
                "return_clip_enabled": r.return_clip_enabled,
                "return_clip_low": r.return_clip_low,
                "return_clip_high": r.return_clip_high,
                "return_clip_count": r.return_clip_count,
                "train_samples": r.train_samples,
                "test_samples": r.test_samples,
                "train_anti_chase_rows": r.train_anti_chase_rows,
                "train_anti_chase_bad_rows": r.train_anti_chase_bad_rows,
                "train_regime_risk_rows": r.train_regime_risk_rows,
                "train_regime_risk_bad_rows": r.train_regime_risk_bad_rows,
                "train_avg_sample_weight": r.train_avg_sample_weight,
                "mae": r.mae,
                "rmse": r.rmse,
                "r2": r.r2,
                "backtest_return_1d": r.backtest_return_1d,
                "backtest_return_3d": r.backtest_return_3d,
                "backtest_return_5d": r.backtest_return_5d,
                "backtest_vs_market_1d": r.backtest_vs_market_1d,
                "backtest_vs_market_3d": r.backtest_vs_market_3d,
                "backtest_vs_market_5d": r.backtest_vs_market_5d,
                "top3_accuracy": r.top3_accuracy,
                "top3_return_1d_win_rate": r.top3_return_1d_win_rate,
                "top3_return_3d_win_rate": r.top3_return_3d_win_rate,
                "top3_vs_market_3d_win_rate": r.top3_vs_market_3d_win_rate,
                "top3_loss_3d_rate": r.top3_loss_3d_rate,
                "top3_loss_5pct_3d_rate": r.top3_loss_5pct_3d_rate,
                "top3_high_position_exposure": r.top3_high_position_exposure,
                "top3_high_position_acceleration_exposure": r.top3_high_position_acceleration_exposure,
                "top20_high_position_acceleration_exposure": r.top20_high_position_acceleration_exposure,
                "weak_market_top3_count": r.weak_market_top3_count,
                "weak_market_top3_return_3d": r.weak_market_top3_return_3d,
                "weak_market_top3_vs_market_3d": r.weak_market_top3_vs_market_3d,
                "weak_market_top3_vs_market_3d_win_rate": r.weak_market_top3_vs_market_3d_win_rate,
                "weak_market_top3_loss_3d_rate": r.weak_market_top3_loss_3d_rate,
                "weak_market_top3_loss_5pct_3d_rate": r.weak_market_top3_loss_5pct_3d_rate,
                "weak_market_top3_high_position_exposure": r.weak_market_top3_high_position_exposure,
                "weak_market_top3_high_position_acceleration_exposure": r.weak_market_top3_high_position_acceleration_exposure,
                "top3_count": r.top3_count,
                "top20_count": r.top20_count,
                "risk_adjusted_selection_score": r.risk_adjusted_selection_score,
                "feature_importance": dict(sorted(r.feature_importance.items(), key=lambda x: -x[1])[:10]),
            }
            for r in results
        ],
        "exported_artifacts": exported_artifacts,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info("结果已保存到: %s", output_path)

    # 打印汇总表格 - 默认按风险调整分排序，避免机器只追逐高位收益样本。
    results_sorted = sorted(
        results,
        key=lambda r: (
            r.risk_adjusted_selection_score
            if r.risk_adjusted_selection_score is not None
            else _calculate_risk_adjusted_selection_score(r),
            r.backtest_vs_market_3d or 0,
            r.backtest_return_3d or 0,
        ),
        reverse=True,
    )

    print("\n" + "=" * 168)
    print("超参数调优结果汇总（按风险调整分排序）")
    print("=" * 168)
    print(
        f"{'模型':<20} {'目标':<12} {'特征':<10} {'数量':<6} {'权重':<18} {'风险分':<10} {'MAE':<10} {'R2':<10} "
        f"{'1日胜率':<10} {'3日胜率':<10} {'超额胜率':<10} {'弱市超额胜率':<14} "
        f"{'3日收益':<10} {'超额3日':<10} {'弱市超额3日':<12} {'高位Top3':<10} {'弱市高位Top3':<14} {'3日大亏':<10}"
    )
    print("-" * 168)
    for r in results_sorted[: max(int(args.top_k), 1)]:
        mae_str = f"{r.mae:.4f}" if isinstance(r.mae, (int, float)) else "N/A"
        r2_str = f"{r.r2:.4f}" if isinstance(r.r2, (int, float)) else "N/A"
        weight_label = (
            r.sample_weight_mode
            if r.sample_weight_mode == "none"
            else f"{r.sample_weight_mode}:{r.sample_weight_profile}"
        )
        print(
            f"{r.model_name:<20} {r.target_name:<12} {r.feature_subset_name:<10} {r.feature_count:<6} {weight_label:<18} "
            f"{(r.risk_adjusted_selection_score if r.risk_adjusted_selection_score is not None else 0.0):<10.4f} "
            f"{mae_str:<10} "
            f"{r2_str:<10} "
            f"{(r.top3_return_1d_win_rate or 0) * 100:>8.2f}% "
            f"{(r.top3_return_3d_win_rate or 0) * 100:>8.2f}% "
            f"{(r.top3_vs_market_3d_win_rate or 0) * 100:>8.2f}% "
            f"{(r.weak_market_top3_vs_market_3d_win_rate or 0) * 100:>12.2f}% "
            f"{(r.backtest_return_3d or 0) * 100:>8.2f}% "
            f"{(r.backtest_vs_market_3d or 0) * 100:>8.2f}% "
            f"{(r.weak_market_top3_vs_market_3d or 0) * 100:>10.2f}% "
            f"{(r.top3_high_position_exposure or 0) * 100:>8.2f}% "
            f"{(r.weak_market_top3_high_position_exposure or 0) * 100:>12.2f}% "
            f"{(r.top3_loss_5pct_3d_rate or 0) * 100:>8.2f}%"
        )

    # 打印最佳配置
    best = results_sorted[0]
    print("\n" + "=" * 168)
    print(f"最佳配置: {best.model_name} + {best.target_name}")
    print(f"最佳特征子集: {best.feature_subset_name} ({best.feature_count} features)")
    print(f"风险调整分: {(best.risk_adjusted_selection_score if best.risk_adjusted_selection_score is not None else 0.0):.4f}")
    print(f"Top3 1日胜率: {(best.top3_return_1d_win_rate or 0) * 100:.2f}%")
    print(f"Top3 3日胜率: {(best.top3_return_3d_win_rate or 0) * 100:.2f}%")
    print(f"Top3 3日超额胜率: {(best.top3_vs_market_3d_win_rate or 0) * 100:.2f}%")
    print(f"弱市Top3 3日超额胜率: {(best.weak_market_top3_vs_market_3d_win_rate or 0) * 100:.2f}%")
    print(f"3日收益: {(best.backtest_return_3d or 0) * 100:.2f}%")
    print(f"3日超额收益: {(best.backtest_vs_market_3d or 0) * 100:.2f}%")
    print(f"弱市3日超额收益: {(best.weak_market_top3_vs_market_3d or 0) * 100:.2f}%")
    print(f"Top3高位暴露: {(best.top3_high_position_exposure or 0) * 100:.2f}%")
    print(f"弱市Top3高位暴露: {(best.weak_market_top3_high_position_exposure or 0) * 100:.2f}%")
    print(f"Top3高位加速暴露: {(best.top3_high_position_acceleration_exposure or 0) * 100:.2f}%")
    print(f"Top3 3日大亏率: {(best.top3_loss_5pct_3d_rate or 0) * 100:.2f}%")
    print(f"弱市Top3 3日大亏率: {(best.weak_market_top3_loss_5pct_3d_rate or 0) * 100:.2f}%")
    print(f"特征列: {', '.join(best.feature_columns)}")
    print("=" * 168)


if __name__ == "__main__":
    main()

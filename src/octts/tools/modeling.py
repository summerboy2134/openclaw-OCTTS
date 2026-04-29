from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


FEATURE_COLUMNS = [
    "in_frontlist",
    "recommend_rank",
    "strategy_count",
    "is_repeat_pick",
    "news_mentioned",
    "entry_price",
    "close",
    "pct_change",
    "volume_ratio",
    "turnover_rate",
    "recommendation_score",
    "overall_score",
    "technical_score",
    "fundamental_score",
    "sentiment_score",
    "news_score",
    "base_score",
    "sentiment_adjustment",
    "news_adjustment",
    "industry_heat_score",
    "distribution_risk_score",
    "moneyflow_3d_value",
    "turnover_spike_ratio",
    "recent_runup_5d",
    "continuation_bias_score",
    "top3_risk_penalty",
    "short_term_contradiction_penalty",
    "late_stage_momentum_flag",
    "candidate_risk_blocked",
    "previous_recommendation_score",
    "previous_overall_score",
    "score_change",
]

CATEGORICAL_COLUMNS = [
    "source_tag",
    "technical_signal",
    "industry",
    "industry_flow_bias",
]


def build_training_frame(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        row = {key: sample.get(key) for key in FEATURE_COLUMNS + CATEGORICAL_COLUMNS}
        row["distribution_risk_flag_count"] = len(sample.get("distribution_risk_flags") or [])
        row["continuation_positive_count"] = len(sample.get("continuation_positive_flags") or [])
        row["continuation_negative_count"] = len(sample.get("continuation_negative_flags") or [])
        row["label_up_1d"] = sample.get("label_up_1d")
        row["trade_date"] = sample.get("trade_date")
        row["ts_code"] = sample.get("ts_code")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in FEATURE_COLUMNS + [
        "distribution_risk_flag_count",
        "continuation_positive_count",
        "continuation_negative_count",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("unknown").astype(str)
    return frame


def build_feature_matrix(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if frame.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    y = frame["label_up_1d"].astype(int)
    x = frame.drop(columns=["label_up_1d", "trade_date", "ts_code"])
    x = pd.get_dummies(x, columns=[column for column in CATEGORICAL_COLUMNS if column in x.columns], dummy_na=False)
    return x, y


def save_model_artifact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_model_artifact(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)

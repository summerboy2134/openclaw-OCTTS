from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from octts.tools.common import print_json
from octts.tools.train_raw_market_model import RAW_MARKET_FEATURE_COLUMNS


HIGH_CORRELATION_THRESHOLD = 0.95
TOP_FEATURE_LIMIT = 15
TOP_CORRELATION_LIMIT = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw-market training features against a selected target.")
    parser.add_argument("--input", required=True, help="CSV dataset path")
    parser.add_argument("--target", default="vs_market_3d", help="Target column to audit")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if frame.empty:
        print_json({"audited": False, "reason": "empty_dataset"})
        return
    if args.target not in frame.columns:
        print_json({"audited": False, "reason": "missing_target", "target": args.target})
        return

    audited = frame[frame[args.target].notna()].copy()
    if audited.empty:
        print_json({"audited": False, "reason": "no_labeled_rows", "target": args.target})
        return

    feature_columns = [column for column in RAW_MARKET_FEATURE_COLUMNS if column in audited.columns]
    feature_frame = audited[feature_columns].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(audited[args.target], errors="coerce")

    missing_rates = feature_frame.isna().mean()
    filled_features = feature_frame.fillna(0.0)
    variances = filled_features.var()
    zero_rates = (filled_features == 0).mean()
    target_correlations = filled_features.corrwith(target).fillna(0.0)

    top_positive = _series_records(target_correlations.sort_values(ascending=False).head(TOP_FEATURE_LIMIT))
    top_negative = _series_records(target_correlations.sort_values().head(TOP_FEATURE_LIMIT))
    lowest_variance = _feature_stat_records(feature_columns, missing_rates, zero_rates, variances)[:TOP_FEATURE_LIMIT]
    high_missing = [record for record in _feature_stat_records(feature_columns, missing_rates, zero_rates, variances) if record["missing_rate"] > 0.2][:TOP_FEATURE_LIMIT]
    high_zero = [record for record in _feature_stat_records(feature_columns, missing_rates, zero_rates, variances) if record["zero_rate"] > 0.8][:TOP_FEATURE_LIMIT]

    correlation_pairs = _high_correlation_pairs(filled_features)

    coefficients = _linear_coefficients(filled_features, target)

    result: Dict[str, Any] = {
        "audited": True,
        "input": args.input,
        "target": args.target,
        "row_count": int(len(audited)),
        "feature_count": int(len(feature_columns)),
        "top_positive_target_correlations": top_positive,
        "top_negative_target_correlations": top_negative,
        "lowest_variance_features": lowest_variance,
        "high_missing_features": high_missing,
        "high_zero_features": high_zero,
        "high_correlation_pairs": correlation_pairs[:TOP_CORRELATION_LIMIT],
        "top_positive_coefficients": coefficients["positive"],
        "top_negative_coefficients": coefficients["negative"],
    }
    print_json(result)


def _series_records(series: pd.Series) -> List[Dict[str, Any]]:
    return [
        {"feature": str(index), "value": float(value)}
        for index, value in series.items()
    ]


def _feature_stat_records(
    feature_columns: List[str],
    missing_rates: pd.Series,
    zero_rates: pd.Series,
    variances: pd.Series,
) -> List[Dict[str, Any]]:
    records = [
        {
            "feature": feature,
            "missing_rate": float(missing_rates.get(feature, 0.0)),
            "zero_rate": float(zero_rates.get(feature, 0.0)),
            "variance": float(variances.get(feature, 0.0)),
        }
        for feature in feature_columns
    ]
    return sorted(records, key=lambda item: item["variance"])


def _high_correlation_pairs(feature_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    correlation_matrix = feature_frame.corr().abs()
    pairs: List[Dict[str, Any]] = []
    columns = list(correlation_matrix.columns)
    for idx, left in enumerate(columns):
        for right in columns[idx + 1:]:
            corr_value = correlation_matrix.loc[left, right]
            if pd.isna(corr_value) or corr_value < HIGH_CORRELATION_THRESHOLD:
                continue
            pairs.append(
                {
                    "left_feature": left,
                    "right_feature": right,
                    "correlation": float(corr_value),
                }
            )
    return sorted(pairs, key=lambda item: item["correlation"], reverse=True)


def _linear_coefficients(feature_frame: pd.DataFrame, target: pd.Series) -> Dict[str, List[Dict[str, Any]]]:
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return {"positive": [], "negative": []}

    model = LinearRegression()
    model.fit(feature_frame, target)
    coefficients = pd.Series(model.coef_, index=feature_frame.columns)
    positive = _series_records(coefficients.sort_values(ascending=False).head(TOP_FEATURE_LIMIT))
    negative = _series_records(coefficients.sort_values().head(TOP_FEATURE_LIMIT))
    return {"positive": positive, "negative": negative}


if __name__ == "__main__":
    main()

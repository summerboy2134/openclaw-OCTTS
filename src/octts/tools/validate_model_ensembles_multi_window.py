from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from octts.config import Settings
from octts.tools.train_tuned_models import BEST35_FEATURE_COLUMNS, FEATURE_COLUMNS, HYPERPARAM_CONFIGS, build_dataset, create_model

logger = logging.getLogger(__name__)
DEFAULT_TARGET = "return_3d"
DEFAULT_MODEL_A = ["lgbm_more_trees"]
DEFAULT_MODEL_B = ["xgb_slow"]
DEFAULT_METHODS = ["rank_mean", "weighted_rank"]
DEFAULT_WEIGHT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


@dataclass
class WindowSpec:
    label: str
    start: str
    end: str


@dataclass
class RunResult:
    window_label: str
    strategy: str
    strategy_type: str
    model_a: str
    model_b: str | None
    method: str
    weight_a: float | None
    weight_b: float | None
    train_samples: int
    test_samples: int
    mae: float | None
    rmse: float | None
    r2: float | None
    backtest_return_1d: float | None
    backtest_return_3d: float | None
    backtest_return_5d: float | None
    top3_accuracy: float | None


def parse_window(raw: str, idx: int) -> WindowSpec:
    start, end = [p.strip() for p in raw.split(":", 1)]
    sdt = datetime.strptime(start, "%Y-%m-%d").date()
    edt = datetime.strptime(end, "%Y-%m-%d").date()
    if sdt > edt:
        raise ValueError(f"Test window start must be <= end: {raw}")
    return WindowSpec(f"window_{idx}_{start}_to_{end}", start, end)


def parse_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_weight_grid(raw: str) -> list[float]:
    vals = sorted(set(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not vals:
        raise ValueError("weight grid cannot be empty")
    for v in vals:
        if not (0 < v < 1):
            raise ValueError(f"weight must be in (0,1): {v}")
    return vals


def feature_columns_for_subset(subset: str) -> list[str]:
    normalized = str(subset or "all").strip().lower()
    if normalized == "all":
        return list(FEATURE_COLUMNS)
    if normalized == "best35":
        return list(BEST35_FEATURE_COLUMNS)
    raise ValueError(f"Unsupported feature subset: {subset}. Expected all or best35")


def parse_feature_subsets(raw: str) -> list[str]:
    subsets = [item.strip().lower() for item in raw.split(",") if item.strip()]
    for subset in subsets:
        feature_columns_for_subset(subset)
    return subsets or ["all"]


def prediction_key(model_name: str, feature_subset: str) -> str:
    return f"pred::{model_name}::{feature_subset}"


def strategy_model_token(model_name: str, feature_subset: str) -> str:
    return f"{model_name}[{feature_subset}]"


def train_one(train_df: pd.DataFrame, target: str, model_name: str, feature_subset: str):
    cfg = HYPERPARAM_CONFIGS[model_name]
    model = create_model(cfg["model_class"], cfg["params"])
    feature_columns = feature_columns_for_subset(feature_subset)
    x = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = train_df[target].values
    mask = ~pd.isna(y)
    x = x[mask]
    y = y[mask]
    model.fit(x, y)
    return model, int(len(y)), feature_columns


def daily_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def fuse(df: pd.DataFrame, pred_a: str, pred_b: str, method: str, weight_a: float | None) -> pd.Series:
    ranked = df.groupby("trade_date", group_keys=False)[[pred_a, pred_b]].transform(daily_rank)
    if method == "rank_mean":
        return ranked[[pred_a, pred_b]].mean(axis=1)
    if method == "weighted_rank":
        assert weight_a is not None
        return ranked[pred_a] * weight_a + ranked[pred_b] * (1.0 - weight_a)
    raise ValueError(f"Unknown method: {method}")


def eval_preds(df: pd.DataFrame, target: str, pred_col: str, eval_reg: bool) -> tuple[float | None, float | None, float | None, int]:
    y = df[target].values
    p = df[pred_col].values
    mask = ~pd.isna(y)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None, None, None, 0
    if not eval_reg:
        return None, None, None, int(len(y))
    return float(mean_absolute_error(y, p)), float(np.sqrt(mean_squared_error(y, p))), float(r2_score(y, p)), int(len(y))


def backtest(df: pd.DataFrame, pred_col: str) -> tuple[float | None, float | None, float | None, float | None]:
    r1, r3, r5 = [], [], []
    correct = total = 0
    for trade_date in df["trade_date"].unique():
        day = df[df["trade_date"] == trade_date]
        if len(day) < 3:
            continue
        top3 = day.nlargest(3, pred_col)
        for _, row in top3.iterrows():
            if pd.notna(row["return_1d_actual"]):
                r1.append(float(row["return_1d_actual"])); total += 1
                if row["return_1d_actual"] > 0: correct += 1
            if pd.notna(row["return_3d_actual"]): r3.append(float(row["return_3d_actual"]))
            if pd.notna(row["return_5d_actual"]): r5.append(float(row["return_5d_actual"]))
    return float(np.mean(r1)) if r1 else None, float(np.mean(r3)) if r3 else None, float(np.mean(r5)) if r5 else None, float(correct / total) if total > 0 else None


def build_specs(
    model_as: list[str],
    model_bs: list[str],
    methods: list[str],
    weight_grid: list[float],
    single_feature_subsets: list[str],
    ensemble_feature_subsets: list[str],
):
    specs = []
    for name in sorted(set(model_as + model_bs)):
        for feature_subset in single_feature_subsets:
            token = strategy_model_token(name, feature_subset)
            specs.append((f"single::{token}", "single", name, None, "raw_mean", None, feature_subset, feature_subset))
    for ensemble_feature_subset in ensemble_feature_subsets:
        for a, b in itertools.product(model_as, model_bs):
            pair = f"{strategy_model_token(a, ensemble_feature_subset)}+{strategy_model_token(b, ensemble_feature_subset)}"
            for method in methods:
                if method == "rank_mean":
                    specs.append((f"ensemble::rank_mean::{pair}", "ensemble", a, b, method, None, ensemble_feature_subset, ensemble_feature_subset))
                elif method == "weighted_rank":
                    for wa in weight_grid:
                        specs.append((f"ensemble::weighted_rank::{wa:.2f}/{1-wa:.2f}::{pair}", "ensemble", a, b, method, wa, ensemble_feature_subset, ensemble_feature_subset))
    return specs


def summarize(results: list[RunResult]) -> dict:
    grouped: dict[str, list[RunResult]] = {}
    for r in results: grouped.setdefault(r.strategy, []).append(r)
    rows = []
    for strategy, items in grouped.items():
        first = items[0]
        r3 = [x.backtest_return_3d for x in items if x.backtest_return_3d is not None]
        r5 = [x.backtest_return_5d for x in items if x.backtest_return_5d is not None]
        acc = [x.top3_accuracy for x in items if x.top3_accuracy is not None]
        mae = [x.mae for x in items if x.mae is not None]
        r2 = [x.r2 for x in items if x.r2 is not None]
        rows.append({"strategy": strategy, "strategy_type": first.strategy_type, "model_a": first.model_a, "model_b": first.model_b, "method": first.method, "weight_a": first.weight_a, "weight_b": first.weight_b, "windows": len(items), "avg_mae": sum(mae)/len(mae) if mae else None, "avg_r2": sum(r2)/len(r2) if r2 else None, "avg_backtest_return_3d": sum(r3)/len(r3) if r3 else None, "min_backtest_return_3d": min(r3) if r3 else None, "max_backtest_return_3d": max(r3) if r3 else None, "avg_backtest_return_5d": sum(r5)/len(r5) if r5 else None, "min_backtest_return_5d": min(r5) if r5 else None, "max_backtest_return_5d": max(r5) if r5 else None, "avg_top3_accuracy": sum(acc)/len(acc) if acc else None})
    rows.sort(key=lambda x: x["avg_backtest_return_3d"] or float("-inf"), reverse=True)
    return {"by_strategy": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused two-model fusion search")
    parser.add_argument("--train-start", type=str, required=True)
    parser.add_argument("--train-end", type=str, required=True)
    parser.add_argument("--test-window", dest="test_windows", action="append", required=True)
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--model-a-candidates", type=str, default=",".join(DEFAULT_MODEL_A))
    parser.add_argument("--model-b-candidates", type=str, default=",".join(DEFAULT_MODEL_B))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--weight-grid", type=str, default=",".join(str(x) for x in DEFAULT_WEIGHT_GRID))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=str, default="tmp/ensemble_multi_window.json")
    parser.add_argument("--single-feature-subsets", type=str, default="all", help="Comma-separated feature subsets for single-model baselines: all,best35")
    parser.add_argument("--ensemble-feature-subsets", type=str, default="all", help="Comma-separated feature subsets used by ensemble members: all,best35")
    parser.add_argument("--no-training-features", action="store_true")
    parser.add_argument("--training-features-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    model_as = parse_list(args.model_a_candidates)
    model_bs = parse_list(args.model_b_candidates)
    methods = parse_list(args.methods)
    weight_grid = parse_weight_grid(args.weight_grid)
    single_feature_subsets = parse_feature_subsets(args.single_feature_subsets)
    ensemble_feature_subsets = parse_feature_subsets(args.ensemble_feature_subsets)
    bad_models = [x for x in model_as + model_bs if x not in HYPERPARAM_CONFIGS]
    if bad_models: raise ValueError(f"Unknown model config(s): {sorted(set(bad_models))}")
    bad_methods = [x for x in methods if x not in {"rank_mean", "weighted_rank"}]
    if bad_methods: raise ValueError(f"Unknown method(s): {bad_methods}")

    specs = build_specs(model_as, model_bs, methods, weight_grid, single_feature_subsets, ensemble_feature_subsets)
    windows = [parse_window(raw, i + 1) for i, raw in enumerate(args.test_windows)]
    settings = Settings()
    prefer_training_features = not args.no_training_features
    training_features_only = bool(args.training_features_only)

    train_start = datetime.strptime(args.train_start, "%Y-%m-%d").date()
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date()
    train_df = build_dataset(settings, train_start, train_end, prefer_training_features=prefer_training_features, training_features_only=training_features_only)
    required_trained_keys = sorted({(a, fa) for _, stype, a, _, _, _, fa, _ in specs} | {(b, fb) for _, stype, _, b, _, _, _, fb in specs if stype == "ensemble" and b is not None})
    trained = {key: train_one(train_df, args.target, key[0], key[1]) for key in required_trained_keys}
    train_samples = min(v[1] for v in trained.values()) if trained else 0

    results: list[RunResult] = []
    for window in windows:
        test_start = datetime.strptime(window.start, "%Y-%m-%d").date()
        test_end = datetime.strptime(window.end, "%Y-%m-%d").date()
        base = build_dataset(settings, test_start, test_end, prefer_training_features=prefer_training_features, training_features_only=training_features_only)
        if base.empty: continue
        df = base.copy()
        for (name, feature_subset), (model, _, feature_columns) in trained.items():
            df[prediction_key(name, feature_subset)] = model.predict(df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0))
        for strategy, stype, a, b, method, wa, feature_subset_a, feature_subset_b in specs:
            fused = f"fused::{strategy}"
            if stype == "single":
                df[fused] = df[prediction_key(a, feature_subset_a)]
                mae, rmse, r2, test_samples = eval_preds(df, args.target, fused, True)
                ret1, ret3, ret5, acc = backtest(df, fused)
                results.append(RunResult(window.label, strategy, stype, strategy_model_token(a, feature_subset_a), None, method, None, None, train_samples, test_samples, mae, rmse, r2, ret1, ret3, ret5, acc))
            else:
                df[fused] = fuse(df, prediction_key(a, feature_subset_a), prediction_key(b, feature_subset_b), method, wa)
                mae, rmse, r2, test_samples = eval_preds(df, args.target, fused, False)
                ret1, ret3, ret5, acc = backtest(df, fused)
                results.append(RunResult(window.label, strategy, stype, strategy_model_token(a, feature_subset_a), strategy_model_token(b, feature_subset_b), method, wa, 1.0-wa if wa is not None else None, train_samples, test_samples, mae, rmse, r2, ret1, ret3, ret5, acc))

    output = {"target": args.target, "model_a_candidates": model_as, "model_b_candidates": model_bs, "methods": methods, "weight_grid": weight_grid, "single_feature_subsets": single_feature_subsets, "ensemble_feature_subsets": ensemble_feature_subsets, "test_windows": [asdict(w) for w in windows], "results": [asdict(r) for r in results], "summary": summarize(results)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f: json.dump(output, f, indent=2, default=str)

    rows = output["summary"]["by_strategy"]
    print("\n" + "=" * 185)
    print("双模型融合多窗口汇总（按平均3日收益排序，5日收益辅助参考）")
    print("=" * 185)
    print(f"{'策略':<70} {'类型':<10} {'窗口数':<8} {'平均MAE':<10} {'平均R2':<10} {'平均3日收益':<12} {'最差3日收益':<12} {'平均5日收益':<12} {'最好5日收益':<12} {'平均准确率':<10}")
    print("-" * 185)
    for row in rows[: max(int(args.top_k), 1)]:
        avg_mae = f"{row['avg_mae']:.4f}" if isinstance(row['avg_mae'], (int, float)) else "N/A"
        avg_r2 = f"{row['avg_r2']:.4f}" if isinstance(row['avg_r2'], (int, float)) else "N/A"
        avg3 = f"{row['avg_backtest_return_3d'] * 100:.2f}%" if isinstance(row['avg_backtest_return_3d'], (int, float)) else "N/A"
        min3 = f"{row['min_backtest_return_3d'] * 100:.2f}%" if isinstance(row['min_backtest_return_3d'], (int, float)) else "N/A"
        avg5 = f"{row['avg_backtest_return_5d'] * 100:.2f}%" if isinstance(row['avg_backtest_return_5d'], (int, float)) else "N/A"
        max5 = f"{row['max_backtest_return_5d'] * 100:.2f}%" if isinstance(row['max_backtest_return_5d'], (int, float)) else "N/A"
        avg_acc = f"{row['avg_top3_accuracy'] * 100:.2f}%" if isinstance(row['avg_top3_accuracy'], (int, float)) else "N/A"
        print(f"{row['strategy']:<70} {row['strategy_type']:<10} {row['windows']:<8} {avg_mae:<10} {avg_r2:<10} {avg3:<12} {min3:<12} {avg5:<12} {max5:<12} {avg_acc:<10}")


if __name__ == "__main__":
    main()

"""
多窗口验证指定候选模型配置。

用法:
    python -m octts.tools.validate_tuned_models_multi_window \
        --train-start 2025-09-01 \
        --train-end 2026-02-28 \
        --test-window 2026-03-01:2026-03-15 \
        --test-window 2026-03-16:2026-03-31 \
        --test-window 2026-04-01:2026-04-17 \
        --training-features-only \
        --output tmp/multi_window_validation.json
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from octts.config import Settings
from octts.tools.train_tuned_models import (
    HYPERPARAM_CONFIGS,
    ModelResult,
    build_dataset,
    evaluate_model,
    backtest_top3,
    train_model,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAMES = [
    "lgbm_more_trees",
    "xgb_slow",
    "lgbm_deep_slow_longer",
    "lgbm_deep_slow_longer_mild_regularized",
    "lgbm_deep_slow_longer_regularized",
    "lgbm_deep_slow_regularized",
]
DEFAULT_TARGET = "return_3d"


@dataclass
class WindowSpec:
    label: str
    start: str
    end: str


@dataclass
class WindowRunResult:
    window_label: str
    model: str
    target: str
    train_samples: int
    test_samples: int
    mae: float | None
    rmse: float | None
    r2: float | None
    backtest_return_1d: float | None
    backtest_return_3d: float | None
    backtest_return_5d: float | None
    top3_accuracy: float | None


def parse_window_spec(raw: str, index: int) -> WindowSpec:
    try:
        start_str, end_str = [part.strip() for part in raw.split(":", 1)]
        start_dt = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid --test-window value: {raw}. Expected YYYY-MM-DD:YYYY-MM-DD") from exc

    if start_dt > end_dt:
        raise ValueError(f"Test window start must be <= end: {raw}")

    return WindowSpec(
        label=f"window_{index}_{start_str}_to_{end_str}",
        start=start_str,
        end=end_str,
    )


def summarize_runs(results: List[WindowRunResult]) -> dict:
    grouped: dict[str, list[WindowRunResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

    summary = []
    for model_name, items in grouped.items():
        backtest_3d_values = [item.backtest_return_3d for item in items if item.backtest_return_3d is not None]
        backtest_5d_values = [item.backtest_return_5d for item in items if item.backtest_return_5d is not None]
        accuracy_values = [item.top3_accuracy for item in items if item.top3_accuracy is not None]
        mae_values = [item.mae for item in items if item.mae is not None]
        r2_values = [item.r2 for item in items if item.r2 is not None]

        summary.append(
            {
                "model": model_name,
                "windows": len(items),
                "avg_mae": sum(mae_values) / len(mae_values) if mae_values else None,
                "avg_r2": sum(r2_values) / len(r2_values) if r2_values else None,
                "avg_backtest_return_3d": sum(backtest_3d_values) / len(backtest_3d_values) if backtest_3d_values else None,
                "min_backtest_return_3d": min(backtest_3d_values) if backtest_3d_values else None,
                "max_backtest_return_3d": max(backtest_3d_values) if backtest_3d_values else None,
                "avg_backtest_return_5d": sum(backtest_5d_values) / len(backtest_5d_values) if backtest_5d_values else None,
                "min_backtest_return_5d": min(backtest_5d_values) if backtest_5d_values else None,
                "max_backtest_return_5d": max(backtest_5d_values) if backtest_5d_values else None,
                "avg_top3_accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else None,
            }
        )

    summary.sort(key=lambda item: item["avg_backtest_return_3d"] or float("-inf"), reverse=True)
    return {"by_model": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="多窗口验证指定候选模型配置")
    parser.add_argument("--train-start", type=str, required=True, help="训练开始日期")
    parser.add_argument("--train-end", type=str, required=True, help="训练结束日期")
    parser.add_argument(
        "--test-window",
        dest="test_windows",
        action="append",
        required=True,
        help="测试窗口，格式 YYYY-MM-DD:YYYY-MM-DD，可重复传入多次",
    )
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="目标变量，默认 return_3d")
    parser.add_argument(
        "--model-names",
        type=str,
        default=",".join(DEFAULT_MODEL_NAMES),
        help="逗号分隔模型配置名",
    )
    parser.add_argument("--output", type=str, default="tmp/multi_window_validation.json", help="输出文件路径")
    parser.add_argument("--no-training-features", action="store_true", help="不读取 training_features，强制动态构建")
    parser.add_argument("--training-features-only", action="store_true", help="只读取 training_features，不回退到动态构建")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    train_start = datetime.strptime(args.train_start, "%Y-%m-%d").date()
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date()

    model_names = [name.strip() for name in args.model_names.split(",") if name.strip()]
    unknown_model_names = [name for name in model_names if name not in HYPERPARAM_CONFIGS]
    if unknown_model_names:
        raise ValueError(f"Unknown model config(s): {unknown_model_names}")

    test_windows = [parse_window_spec(raw, index + 1) for index, raw in enumerate(args.test_windows)]
    settings = Settings()
    prefer_training_features = not args.no_training_features
    training_features_only = bool(args.training_features_only)

    logger.info("构建训练数据集: %s ~ %s", train_start, train_end)
    train_df = build_dataset(
        settings,
        train_start,
        train_end,
        prefer_training_features=prefer_training_features,
        training_features_only=training_features_only,
    )
    logger.info("训练样本数: %d", len(train_df))

    all_results: list[WindowRunResult] = []
    for window in test_windows:
        test_start = datetime.strptime(window.start, "%Y-%m-%d").date()
        test_end = datetime.strptime(window.end, "%Y-%m-%d").date()
        logger.info("构建测试数据集: %s ~ %s", test_start, test_end)
        test_df = build_dataset(
            settings,
            test_start,
            test_end,
            prefer_training_features=prefer_training_features,
            training_features_only=training_features_only,
        )
        logger.info("窗口 %s 测试样本数: %d", window.label, len(test_df))

        for model_name in model_names:
            logger.info("窗口 %s 训练模型: %s", window.label, model_name)
            model, result = train_model(train_df, args.target, model_name)
            result = evaluate_model(model, test_df, args.target, result)
            result = backtest_top3(model, test_df, result)
            all_results.append(
                WindowRunResult(
                    window_label=window.label,
                    model=result.model_name,
                    target=result.target_name,
                    train_samples=result.train_samples,
                    test_samples=result.test_samples,
                    mae=result.mae,
                    rmse=result.rmse,
                    r2=result.r2,
                    backtest_return_1d=result.backtest_return_1d,
                    backtest_return_3d=result.backtest_return_3d,
                    backtest_return_5d=result.backtest_return_5d,
                    top3_accuracy=result.top3_accuracy,
                )
            )
            logger.info(
                "窗口 %s 结果: model=%s, mae=%.4f, r2=%.4f, top3_return_3d=%.2f%%, top3_return_5d=%.2f%%, accuracy=%.2f%%",
                window.label,
                result.model_name,
                result.mae or 0,
                result.r2 or 0,
                (result.backtest_return_3d or 0) * 100,
                (result.backtest_return_5d or 0) * 100,
                (result.top3_accuracy or 0) * 100,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "train_period": {"start": str(train_start), "end": str(train_end)},
        "target": args.target,
        "model_names": model_names,
        "test_windows": [asdict(window) for window in test_windows],
        "results": [asdict(result) for result in all_results],
        "summary": summarize_runs(all_results),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info("结果已保存到: %s", output_path)

    summary_rows = output_data["summary"]["by_model"]
    print("\n" + "=" * 155)
    print("多窗口验证汇总（按平均3日收益排序，5日收益辅助参考）")
    print("=" * 155)
    print(
        f"{'模型':<28} {'窗口数':<8} {'平均MAE':<10} {'平均R2':<10} {'平均3日收益':<12} {'最差3日收益':<12} {'平均5日收益':<12} {'最好5日收益':<12} {'平均准确率':<10}"
    )
    print("-" * 155)
    for row in summary_rows:
        avg_mae = f"{row['avg_mae']:.4f}" if isinstance(row["avg_mae"], (int, float)) else "N/A"
        avg_r2 = f"{row['avg_r2']:.4f}" if isinstance(row["avg_r2"], (int, float)) else "N/A"
        avg_ret_3d = f"{row['avg_backtest_return_3d'] * 100:.2f}%" if isinstance(row["avg_backtest_return_3d"], (int, float)) else "N/A"
        min_ret_3d = f"{row['min_backtest_return_3d'] * 100:.2f}%" if isinstance(row["min_backtest_return_3d"], (int, float)) else "N/A"
        avg_ret_5d = f"{row['avg_backtest_return_5d'] * 100:.2f}%" if isinstance(row["avg_backtest_return_5d"], (int, float)) else "N/A"
        max_ret_5d = f"{row['max_backtest_return_5d'] * 100:.2f}%" if isinstance(row["max_backtest_return_5d"], (int, float)) else "N/A"
        avg_acc = f"{row['avg_top3_accuracy'] * 100:.2f}%" if isinstance(row["avg_top3_accuracy"], (int, float)) else "N/A"
        print(
            f"{row['model']:<28} {row['windows']:<8} {avg_mae:<10} {avg_r2:<10} {avg_ret_3d:<12} {min_ret_3d:<12} {avg_ret_5d:<12} {max_ret_5d:<12} {avg_acc:<10}"
        )


if __name__ == "__main__":
    main()

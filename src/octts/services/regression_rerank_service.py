from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from octts.config import Settings
from octts.schemas.screener import ScreenResult
from octts.services.market_raw_data_repository import MarketRawDataRepository
from octts.services.raw_market_training_dataset import RawMarketTrainingDatasetBuilder
from octts.tools.modeling import load_model_artifact

logger = logging.getLogger(__name__)


@dataclass
class RegressionRerankResult:
    candidate_codes: List[str]
    analysis_codes: List[str]
    metadata_by_code: Dict[str, Dict[str, Any]]
    artifact_path: Optional[str]
    fallback_reason: Optional[str] = None
    error_message: Optional[str] = None


class RegressionRerankService:
    DEFAULT_MODEL_SPECS = [
        ("lgbm_more_trees", "raw_market_202509_202602_return_3d_lgbm_more_trees.lightgbm.pkl", 0.60),
        ("xgb_slow", "raw_market_202509_202602_return_3d_xgb_slow.xgboost.pkl", 0.40),
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_builder = RawMarketTrainingDatasetBuilder(settings)
        self.market_raw_data_repo = MarketRawDataRepository(settings.database_url)

    def rank_candidates(
        self,
        screening_results: Dict[str, ScreenResult],
        *,
        trade_date: date,
        coarse_limit: int,
        analysis_limit: int,
        exclude_bj: bool,
        rule_weight: float,
    ) -> RegressionRerankResult:
        rule_ranked = self._build_rule_ranked_candidates(
            screening_results,
            trade_date=trade_date,
            coarse_limit=coarse_limit,
            exclude_bj=exclude_bj,
        )
        if not rule_ranked:
            return RegressionRerankResult(
                candidate_codes=[],
                analysis_codes=[],
                metadata_by_code={},
                artifact_path=None,
                fallback_reason="empty_rule_pool",
            )

        artifact_paths = self._resolve_default_artifact_paths()
        if not artifact_paths:
            logger.warning("Regression rerank skipped: model artifact missing")
            metadata = {item["ts_code"]: item for item in rule_ranked}
            ordered_codes = [item["ts_code"] for item in rule_ranked]
            return RegressionRerankResult(
                candidate_codes=ordered_codes,
                analysis_codes=ordered_codes[:analysis_limit],
                metadata_by_code=metadata,
                artifact_path=None,
                fallback_reason="missing_model_artifact",
            )

        try:
            artifacts = self._load_default_artifacts(artifact_paths)
            enriched_candidates = self._apply_model_scores(
                rule_ranked,
                trade_date=trade_date,
                artifacts=artifacts,
                rule_weight=rule_weight,
            )
        except Exception:
            logger.exception("Regression rerank failed, fallback to rule-ranked candidates")
            metadata = {item["ts_code"]: item for item in rule_ranked}
            ordered_codes = [item["ts_code"] for item in rule_ranked]
            return RegressionRerankResult(
                candidate_codes=ordered_codes,
                analysis_codes=ordered_codes[:analysis_limit],
                metadata_by_code=metadata,
                artifact_path=",".join(str(path) for _, path, _ in artifact_paths),
                fallback_reason="rerank_exception",
            )

        ordered_codes = [item["ts_code"] for item in enriched_candidates]
        analysis_codes = self._build_analysis_codes_with_moneyflow_veto(
            enriched_candidates,
            analysis_limit=analysis_limit,
        )
        return RegressionRerankResult(
            candidate_codes=ordered_codes,
            analysis_codes=analysis_codes,
            metadata_by_code={item["ts_code"]: item for item in enriched_candidates},
            artifact_path=",".join(str(path) for _, path, _ in artifact_paths),
            fallback_reason=None,
        )

    def rank_market_universe(
        self,
        *,
        trade_date: date,
        candidate_limit: int,
        analysis_limit: int,
        exclude_bj: bool,
    ) -> RegressionRerankResult:
        artifact_paths = self._resolve_default_artifact_paths()
        if not artifact_paths:
            logger.warning("Market universe rank skipped: model artifact missing")
            return RegressionRerankResult(
                candidate_codes=[],
                analysis_codes=[],
                metadata_by_code={},
                artifact_path=None,
                fallback_reason="missing_model_artifact",
            )

        try:
            samples = self.dataset_builder.build_samples(
                start_date=trade_date,
                end_date=trade_date,
                exclude_bj=exclude_bj,
            )
            sample_map = {
                sample.ts_code.strip().upper(): sample.model_dump(mode="python")
                for sample in samples
            }
            candidates = self._build_universe_candidates(sample_map)
            artifacts = self._load_default_artifacts(artifact_paths)
            enriched_candidates = self._apply_model_scores(
                candidates,
                trade_date=trade_date,
                artifacts=artifacts,
                rule_weight=0.0,
                sample_map=sample_map,
            )
        except Exception as exc:
            logger.exception("Market universe rank failed")
            return RegressionRerankResult(
                candidate_codes=[],
                analysis_codes=[],
                metadata_by_code={},
                artifact_path=",".join(str(path) for _, path, _ in artifact_paths),
                fallback_reason="universe_rank_exception",
                error_message=str(exc),
            )

        top_candidates = enriched_candidates[:candidate_limit]
        ordered_codes = [item["ts_code"] for item in top_candidates]
        for item in top_candidates:
            item["technical_score"] = round(float(item.get("model_score_norm") or 0.0) * 100.0, 4)
            item["recommendation_score"] = round(float(item.get("blend_score") or 0.0) * 100.0, 4)
            item["score_mode"] = "model_universe_rank"
        logger.info(
            "Market universe rank complete: samples=%s, ranked=%s, candidate_limit=%s, feature_model=%s",
            len(samples),
            len(enriched_candidates),
            candidate_limit,
            ",".join(str(path.name) for _, path, _ in artifact_paths),
        )
        return RegressionRerankResult(
            candidate_codes=ordered_codes,
            analysis_codes=ordered_codes[:analysis_limit],
            metadata_by_code={item["ts_code"]: item for item in top_candidates},
            artifact_path=",".join(str(path) for _, path, _ in artifact_paths),
            fallback_reason=None,
        )

    @staticmethod
    def _build_universe_candidates(sample_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for ts_code, sample in sample_map.items():
            candidates.append(
                {
                    "ts_code": ts_code,
                    "name": ts_code,
                    "count": 1,
                    "best_rank": 0,
                    "technical_score": None,
                    "technical_score_min": None,
                    "technical_score_max": None,
                    "recommendation_score": None,
                    "pct_change": float(sample.get("pct_change") or 0.0),
                    "volume_ratio": float(sample.get("volume_ratio") or 0.0),
                    "turnover_rate": float(sample.get("turnover_rate") or 0.0),
                    "rsi": None,
                    "close": sample.get("close"),
                    "market_cap": sample.get("market_cap"),
                    "pe_ttm": sample.get("pe_ttm"),
                    "pb": sample.get("pb"),
                    "amount": sample.get("amount"),
                    "vol": sample.get("vol"),
                    "price_position_20d": sample.get("price_position_20d"),
                    "recent_3d_net_inflow": 0.0,
                    "recent_large_order_net_inflow": 0.0,
                    "recent_super_large_order_net_inflow": 0.0,
                    "moneyflow_positive_flag": 0.0,
                    "moneyflow_summary_rows": 0,
                    "moneyflow_missing_for_top3": True,
                    "moneyflow_signal_score": None,
                    "moneyflow_model_combo_bucket": None,
                    "divergence_score": 0.0,
                    "rule_score": 0.0,
                    "score_mode": "model_universe_rank",
                    "rule_weight": 0.0,
                    "model_target": None,
                    "model_score": None,
                    "model_score_norm": None,
                    "rule_score_norm": None,
                    "blend_score": 0.0,
                    "rerank_pool_rank": None,
                    "moneyflow_vetoed_for_top3": False,
                }
            )
        return candidates

    def _build_rule_ranked_candidates(
        self,
        screening_results: Dict[str, ScreenResult],
        *,
        trade_date: date,
        coarse_limit: int,
        exclude_bj: bool,
    ) -> List[Dict[str, Any]]:
        stock_scores: Dict[str, Dict[str, Any]] = {}
        for result in screening_results.values():
            if not result:
                continue
            for rank_index, stock in enumerate(result.stocks):
                ts_code = str(stock.ts_code).strip().upper()
                if exclude_bj and ts_code.endswith(".BJ"):
                    continue
                technical_score = float(stock.technical_score or 0.0)
                recommendation_score = float(stock.recommendation_score or stock.score or 0.0)
                pct_change = float(stock.pct_change or 0.0) if stock.pct_change is not None else 0.0
                volume_ratio = float(stock.volume_ratio or 0.0)
                turnover_rate = float(stock.turnover_rate or 0.0)
                rsi = float(stock.rsi) if stock.rsi is not None else None
                item = stock_scores.setdefault(
                    ts_code,
                    {
                        "ts_code": ts_code,
                        "name": getattr(stock, "name", ts_code),
                        "count": 0,
                        "best_rank": float("inf"),
                        "technical_score": technical_score,
                        "technical_score_min": technical_score,
                        "technical_score_max": technical_score,
                        "recommendation_score": recommendation_score,
                        "pct_change": pct_change,
                        "volume_ratio": volume_ratio,
                        "turnover_rate": turnover_rate,
                        "rsi": rsi,
                    },
                )
                item["count"] += 1
                item["best_rank"] = min(float(item["best_rank"]), float(rank_index))
                item["technical_score_min"] = min(float(item["technical_score_min"]), technical_score)
                item["technical_score_max"] = max(float(item["technical_score_max"]), technical_score)
                representative_score = float(item["recommendation_score"] or 0.0)
                representative_technical = float(item["technical_score"] or 0.0)
                if recommendation_score > representative_score or (
                    recommendation_score == representative_score and technical_score > representative_technical
                ):
                    item["name"] = getattr(stock, "name", item["name"])
                    item["technical_score"] = technical_score
                    item["recommendation_score"] = recommendation_score
                    item["pct_change"] = pct_change
                    item["volume_ratio"] = volume_ratio
                    item["turnover_rate"] = turnover_rate
                    item["rsi"] = rsi

        filtered: List[Dict[str, Any]] = []
        moneyflow_summaries = self.market_raw_data_repo.get_moneyflow_summaries_by_trade_date(
            ts_codes=stock_scores.keys(),
            trade_date=trade_date.strftime("%Y%m%d"),
            lookback_days=3,
        )
        relaxed_reject_reasons = {
            "rsi": 0,
        }
        for item in stock_scores.values():
            moneyflow_summary = moneyflow_summaries.get(item["ts_code"], {})
            recent_3d_net_inflow = float(moneyflow_summary.get("recent_3d_net_inflow") or 0.0)
            recent_large_order_net_inflow = float(moneyflow_summary.get("recent_large_order_net_inflow") or 0.0)
            recent_super_large_order_net_inflow = float(moneyflow_summary.get("recent_super_large_order_net_inflow") or 0.0)
            rsi = item.get("rsi")
            if rsi is not None and (float(rsi) > 92 or float(rsi) < 8):
                relaxed_reject_reasons["rsi"] += 1
                continue
            divergence_score = max(
                0.0,
                float(item["technical_score_max"] or 0.0) - float(item["technical_score_min"] or 0.0),
            )
            rule_score = (
                float(item["count"] or 0.0) * 100.0
                + float(item["technical_score"] or 0.0) * 1.2
                + float(item["recommendation_score"] or 0.0)
                + float(item["volume_ratio"] or 0.0) * 6.0
                + float(item["turnover_rate"] or 0.0) * 0.8
                + float(item["pct_change"] or 0.0) * 1.5
                - divergence_score * 0.25
                - float(item["best_rank"] or 0.0) * 1.5
            )
            filtered.append(
                {
                    **item,
                    "recent_3d_net_inflow": round(recent_3d_net_inflow, 2),
                    "recent_large_order_net_inflow": round(recent_large_order_net_inflow, 2),
                    "recent_super_large_order_net_inflow": round(recent_super_large_order_net_inflow, 2),
                    "moneyflow_positive_flag": float(moneyflow_summary.get("positive_flag") or 0.0),
                    "moneyflow_summary_rows": int(moneyflow_summary.get("rows") or 0),
                    "moneyflow_missing_for_top3": int(moneyflow_summary.get("rows") or 0) <= 0,
                    "moneyflow_signal_score": None,
                    "moneyflow_model_combo_bucket": None,
                    "divergence_score": round(divergence_score, 4),
                    "rule_score": round(rule_score, 6),
                    "score_mode": "rule_only",
                    "rule_weight": None,
                    "model_target": None,
                    "model_score": None,
                    "model_score_norm": None,
                    "rule_score_norm": None,
                    "blend_score": round(rule_score, 6),
                    "rerank_pool_rank": None,
                    "moneyflow_vetoed_for_top3": False,
                }
            )

        logger.info(
            "Regression rerank rule pool: raw_candidates=%s, filtered_candidates=%s, rejects=%s, coarse_limit=%s, effective_input_count=%s, exclude_bj=%s",
            len(stock_scores),
            len(filtered),
            relaxed_reject_reasons,
            coarse_limit,
            min(len(filtered), coarse_limit),
            exclude_bj,
        )

        filtered.sort(
            key=lambda item: (
                -float(item["count"] or 0.0),
                -float(item["technical_score"] or 0.0),
                -float(item["volume_ratio"] or 0.0),
                -float(item["pct_change"] or 0.0),
                item["ts_code"],
            )
        )
        return filtered[:coarse_limit]

    def _apply_model_scores(
        self,
        candidates: List[Dict[str, Any]],
        *,
        trade_date: date,
        artifacts: List[Dict[str, Any]],
        rule_weight: float,
        sample_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        candidate_codes = [item["ts_code"] for item in candidates]
        if sample_map is None:
            samples = self.dataset_builder.build_samples_for_codes(
                candidate_codes, start_date=trade_date, end_date=trade_date
            )
            sample_map = {sample.ts_code.strip().upper(): sample.model_dump(mode="python") for sample in samples}

        prediction_maps: Dict[str, Dict[str, float]] = {}
        for artifact_spec in artifacts:
            model_name = str(artifact_spec["model_name"])
            artifact = artifact_spec["artifact"]
            feature_columns = list(artifact.get("feature_columns") or [])
            model = artifact.get("model")
            if not feature_columns or model is None:
                raise RuntimeError(f"artifact missing feature_columns or model: {model_name}")

            rows: List[Dict[str, Any]] = []
            predicted_codes: List[str] = []
            for item in candidates:
                ts_code = item["ts_code"]
                sample = sample_map.get(ts_code)
                if not sample:
                    continue
                rows.append({column: sample.get(column, 0.0) for column in feature_columns})
                predicted_codes.append(ts_code)

            prediction_map: Dict[str, float] = {}
            if rows:
                frame = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce").fillna(0.0)
                predictions = model.predict(frame)
                prediction_map = {code: float(score) for code, score in zip(predicted_codes, predictions)}
            prediction_maps[model_name] = prediction_map

        enriched = [dict(item) for item in candidates]
        self._normalize_in_place(enriched, "rule_score", "rule_score_norm")

        for artifact_spec in artifacts:
            model_name = str(artifact_spec["model_name"])
            score_key = f"model_score__{model_name}"
            norm_key = f"model_score_norm__{model_name}"
            prediction_map = prediction_maps.get(model_name, {})
            for item in enriched:
                item[score_key] = prediction_map.get(item["ts_code"])
            self._normalize_in_place(enriched, score_key, norm_key)

        ranked_frames: list[pd.Series] = []
        for artifact_spec in artifacts:
            model_name = str(artifact_spec["model_name"])
            norm_key = f"model_score_norm__{model_name}"
            weight = float(artifact_spec["weight"])
            series = pd.Series(
                {item["ts_code"]: float(item.get(norm_key) or 0.0) for item in enriched},
                dtype=float,
            )
            ranked = series.rank(method="average", pct=True)
            ranked_frames.append(ranked * weight)

        ensemble_rank_score_map: Dict[str, float] = {}
        if ranked_frames:
            combined = ranked_frames[0]
            for extra in ranked_frames[1:]:
                combined = combined.add(extra, fill_value=0.0)
            ensemble_rank_score_map = {str(code): round(float(score), 6) for code, score in combined.items()}

        bounded_rule_weight = max(0.0, min(1.0, float(rule_weight)))
        model_weight = 1.0 - bounded_rule_weight
        target_names = [str(spec["artifact"].get("target") or "return_3d") for spec in artifacts]
        for item in enriched:
            item["model_target"] = "+".join(target_names)
            item["model_score"] = ensemble_rank_score_map.get(item["ts_code"])
            item["model_score_norm"] = item["model_score"]
            item["model_ensemble_method"] = "weighted_rank"
            item["model_ensemble_weights"] = {str(spec["model_name"]): float(spec["weight"]) for spec in artifacts}
            model_score_norm = float(item.get("model_score_norm") or 0.0)
            rule_score_norm = float(item.get("rule_score_norm") or 0.0)
            if item.get("model_score") is None:
                blend_score = rule_score_norm
            else:
                blend_score = model_weight * model_score_norm + bounded_rule_weight * rule_score_norm
            item["blend_score"] = round(float(blend_score), 6)
            item["score_mode"] = "model_rule_blend"
            item["rule_weight"] = bounded_rule_weight

        enriched.sort(
            key=lambda item: (
                -float(item["blend_score"] or 0.0),
                -float(item.get("model_score_norm") or 0.0),
                -float(item.get("rule_score_norm") or 0.0),
                item["ts_code"],
            )
        )
        for index, item in enumerate(enriched, start=1):
            item["rerank_pool_rank"] = index
            item["selected_for_analysis"] = False
            item["moneyflow_vetoed_for_top3"] = False
            item["model_score_vetoed_for_top3"] = False
            item["veto_reason_for_top3"] = None
            item["model_signal_positive"] = item.get("model_score") is not None and float(item.get("model_score") or 0.0) >= 0.0
            item["moneyflow_signal_score"] = self._compute_moneyflow_signal_score(item)
            item["moneyflow_model_combo_bucket"] = self._build_moneyflow_model_combo_bucket(item)
            soft_filter_penalty = self._compute_soft_filter_penalty(item)
            item["soft_filter_penalty"] = soft_filter_penalty
            item["soft_filter_score"] = round(float(item.get("blend_score") or 0.0) + soft_filter_penalty, 6)
        return enriched

    def _build_analysis_codes_with_moneyflow_veto(self, candidates: List[Dict[str, Any]], *, analysis_limit: int) -> List[str]:
        selected: List[str] = []
        vetoed_count = 0
        veto_counts = {
            "moneyflow": 0,
            "moneyflow_missing": 0,
            "model_score": 0,
        }
        for item in candidates:
            ts_code = str(item.get("ts_code") or "").strip().upper()
            if not ts_code:
                continue
            item["selected_for_analysis"] = False
            item["moneyflow_vetoed_for_top3"] = False
            item["model_score_vetoed_for_top3"] = False
            item["veto_reason_for_top3"] = None

            moneyflow_summary_rows = int(item.get("moneyflow_summary_rows") or 0)
            moneyflow_missing = moneyflow_summary_rows <= 0
            item["moneyflow_missing_for_top3"] = moneyflow_missing
            recent_3d_net_inflow = float(item.get("recent_3d_net_inflow") or 0.0)
            recent_large_order_net_inflow = float(item.get("recent_large_order_net_inflow") or 0.0)
            recent_super_large_order_net_inflow = float(item.get("recent_super_large_order_net_inflow") or 0.0)
            moneyflow_veto = (not moneyflow_missing) and (
                (recent_3d_net_inflow < 0.0 and recent_super_large_order_net_inflow <= 0.0)
                or recent_3d_net_inflow < -3000.0
                or (recent_large_order_net_inflow <= 0.0 and recent_super_large_order_net_inflow <= 0.0)
            )
            if moneyflow_veto:
                item["moneyflow_vetoed_for_top3"] = True
                item["veto_reason_for_top3"] = "moneyflow"
                vetoed_count += 1
                veto_counts["moneyflow"] += 1
                continue
            if moneyflow_missing:
                veto_counts["moneyflow_missing"] += 1

            model_score = item.get("model_score")
            if model_score is not None and float(model_score) < -0.07:
                item["model_score_vetoed_for_top3"] = True
                item["veto_reason_for_top3"] = "model_score"
                vetoed_count += 1
                veto_counts["model_score"] += 1
                continue

            item["selected_for_analysis"] = True
            selected.append(ts_code)
            if len(selected) >= analysis_limit:
                logger.info(
                    "Regression rerank analysis selection complete: selected=%s, vetoed=%s, veto_counts=%s, analysis_limit=%s",
                    len(selected),
                    vetoed_count,
                    veto_counts,
                    analysis_limit,
                )
                return selected

        logger.info(
            "Regression rerank analysis selection exhausted: selected=%s, vetoed=%s, veto_counts=%s, analysis_limit=%s",
            len(selected),
            vetoed_count,
            veto_counts,
            analysis_limit,
        )
        return selected[:analysis_limit]

    @staticmethod
    def _compute_moneyflow_signal_score(item: Dict[str, Any]) -> Optional[float]:
        rows = int(item.get("moneyflow_summary_rows") or 0)
        if rows <= 0:
            return None
        recent_3d_net_inflow = float(item.get("recent_3d_net_inflow") or 0.0)
        recent_large_order_net_inflow = float(item.get("recent_large_order_net_inflow") or 0.0)
        recent_super_large_order_net_inflow = float(item.get("recent_super_large_order_net_inflow") or 0.0)
        score = 0.0
        if recent_3d_net_inflow > 0.0:
            score += 1.0
        elif recent_3d_net_inflow < -3000.0:
            score -= 1.0
        elif recent_3d_net_inflow < 0.0:
            score -= 0.5
        if recent_large_order_net_inflow > 0.0:
            score += 0.5
        elif recent_large_order_net_inflow < 0.0:
            score -= 0.5
        if recent_super_large_order_net_inflow > 0.0:
            score += 0.8
        elif recent_super_large_order_net_inflow < 0.0:
            score -= 0.8
        return round(score, 4)

    @staticmethod
    def _build_moneyflow_model_combo_bucket(item: Dict[str, Any]) -> str:
        rows = int(item.get("moneyflow_summary_rows") or 0)
        model_score = item.get("model_score")
        if rows <= 0:
            moneyflow_state = "unknown"
        else:
            moneyflow_signal_score = RegressionRerankService._compute_moneyflow_signal_score(item)
            moneyflow_state = "good" if (moneyflow_signal_score or 0.0) > 0 else "weak"
        if model_score is None:
            model_state = "unknown"
        elif float(model_score) >= 0.0:
            model_state = "good"
        else:
            model_state = "weak"
        return f"moneyflow_{moneyflow_state}__model_{model_state}"

    @staticmethod
    def _compute_soft_filter_penalty(item: Dict[str, Any]) -> float:
        combo_bucket = RegressionRerankService._build_moneyflow_model_combo_bucket(item)
        adjustments = {
            "moneyflow_good__model_good": 0.04,
            "moneyflow_good__model_weak": -0.03,
            "moneyflow_good__model_unknown": -0.02,
            "moneyflow_weak__model_good": -0.06,
            "moneyflow_weak__model_weak": -0.14,
            "moneyflow_weak__model_unknown": -0.08,
            "moneyflow_unknown__model_good": -0.04,
            "moneyflow_unknown__model_weak": -0.12,
            "moneyflow_unknown__model_unknown": -0.06,
        }
        return float(adjustments.get(combo_bucket, 0.0))

    @staticmethod
    def _normalize_in_place(items: List[Dict[str, Any]], source_key: str, target_key: str) -> None:
        values = [float(item[source_key]) for item in items if item.get(source_key) is not None]
        if not values:
            for item in items:
                item[target_key] = None
            return
        min_value = min(values)
        max_value = max(values)
        scale = max_value - min_value
        for item in items:
            raw_value = item.get(source_key)
            if raw_value is None:
                item[target_key] = None
            elif scale <= 0:
                item[target_key] = 0.0
            else:
                item[target_key] = round((float(raw_value) - min_value) / scale, 6)

    def _load_default_artifacts(self, artifact_paths: List[tuple[str, Path, float]]) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        for model_name, path, weight in artifact_paths:
            artifacts.append(
                {
                    "model_name": model_name,
                    "path": path,
                    "weight": weight,
                    "artifact": load_model_artifact(path),
                }
            )
        return artifacts

    def _resolve_default_artifact_paths(self) -> List[tuple[str, Path, float]]:
        resolved: List[tuple[str, Path, float]] = []
        for model_name, filename, weight in self.DEFAULT_MODEL_SPECS:
            candidate = Path(self.settings.history_dir_path) / "short_term_models" / filename
            if not candidate.exists():
                return []
            resolved.append((model_name, candidate, weight))
        return resolved

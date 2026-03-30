"""智能选股结果验证器"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from octts.clients.tushare_client import TushareClient
from octts.config import Settings

logger = logging.getLogger(__name__)


class ScreeningValidator:
    """验证选股结果的准确性"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tushare_client = TushareClient(settings)

    async def validate_recommendations(
        self,
        recommendations: Dict[str, Dict[str, Any]],
        check_days: int = 5,
    ) -> Dict[str, Any]:
        """
        验证推荐股票在未来几天的表现

        Args:
            recommendations: 推荐结果 {股票代码: {推荐信息}}
            check_days: 检查未来几天的表现

        Returns:
            验证结果
        """
        if not recommendations:
            return {
                "total_recommendations": 0,
                "validation_date": datetime.now().isoformat(),
                "results": [],
            }

        # 获取推荐时的价格
        recommendation_date = datetime.now().strftime("%Y%m%d")
        future_date = (datetime.now() + timedelta(days=check_days)).strftime("%Y%m%d")

        validation_results = []
        win_count = 0
        total_gain = 0.0

        for code, rec_info in recommendations.items():
            try:
                # 获取推荐时的收盘价
                rec_price_data = self.tushare_client.fetch_daily_data(
                    ts_code=code,
                    start_date=recommendation_date,
                    end_date=recommendation_date,
                )
                if not rec_price_data:
                    logger.warning(f"No price data for {code} on {recommendation_date}")
                    continue

                rec_price = float(rec_price_data[0].get("close", 0))

                # 获取未来的最高价
                future_data = self.tushare_client.fetch_daily_data(
                    ts_code=code,
                    start_date=recommendation_date,
                    end_date=future_date,
                )
                if not future_data:
                    logger.warning(f"No future data for {code}")
                    continue

                # 计算最高涨幅
                max_price = max(float(d.get("high", 0)) for d in future_data)
                gain_pct = (max_price - rec_price) / rec_price * 100 if rec_price > 0 else 0

                is_win = gain_pct > 0
                if is_win:
                    win_count += 1
                total_gain += gain_pct

                validation_results.append({
                    "code": code,
                    "name": rec_info.get("name", ""),
                    "recommendation": rec_info.get("recommendation", ""),
                    "score": rec_info.get("score", 0),
                    "rec_price": rec_price,
                    "max_price": max_price,
                    "gain_pct": round(gain_pct, 2),
                    "is_win": is_win,
                })

            except Exception as e:
                logger.error(f"Failed to validate {code}: {e}")

        total_count = len(validation_results)
        win_rate = (win_count / total_count * 100) if total_count > 0 else 0
        avg_gain = (total_gain / total_count) if total_count > 0 else 0

        return {
            "total_recommendations": len(recommendations),
            "validated_count": total_count,
            "validation_date": datetime.now().isoformat(),
            "check_days": check_days,
            "win_count": win_count,
            "win_rate": round(win_rate, 2),
            "avg_gain_pct": round(avg_gain, 2),
            "total_gain_pct": round(total_gain, 2),
            "results": sorted(
                validation_results,
                key=lambda x: x["gain_pct"],
                reverse=True
            ),
        }

    async def analyze_recommendation_quality(
        self,
        recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        分析推荐质量（不需要等待，实时分析）

        Args:
            recommendations: 推荐结果

        Returns:
            质量分析
        """
        if not recommendations:
            return {"quality_score": 0, "analysis": "没有推荐"}

        scores = [rec.get("score", 0) for rec in recommendations.values()]
        confidences = [rec.get("ai_confidence", 0) for rec in recommendations.values()]
        strategy_counts = [rec.get("strategy_count", 0) for rec in recommendations.values()]

        avg_score = sum(scores) / len(scores) if scores else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        multi_strategy_count = sum(1 for c in strategy_counts if c > 1)

        # 质量评分
        quality_score = 0
        quality_reasons = []

        # 推荐分数高
        if avg_score >= 70:
            quality_score += 30
            quality_reasons.append(f"平均推荐分数 {avg_score:.1f} 较高")
        elif avg_score >= 60:
            quality_score += 20
            quality_reasons.append(f"平均推荐分数 {avg_score:.1f} 中等")
        else:
            quality_reasons.append(f"平均推荐分数 {avg_score:.1f} 偏低")

        # AI 置信度高
        if avg_confidence >= 0.7:
            quality_score += 30
            quality_reasons.append(f"AI 置信度 {avg_confidence:.2f} 较高")
        elif avg_confidence >= 0.6:
            quality_score += 20
            quality_reasons.append(f"AI 置信度 {avg_confidence:.2f} 中等")
        else:
            quality_reasons.append(f"AI 置信度 {avg_confidence:.2f} 偏低")

        # 多策略共振
        if multi_strategy_count >= len(recommendations) * 0.5:
            quality_score += 40
            quality_reasons.append(f"多策略共振率 {multi_strategy_count}/{len(recommendations)}")
        else:
            quality_reasons.append(f"多策略共振率较低 {multi_strategy_count}/{len(recommendations)}")

        return {
            "quality_score": min(quality_score, 100),
            "total_recommendations": len(recommendations),
            "avg_score": round(avg_score, 2),
            "avg_confidence": round(avg_confidence, 4),
            "multi_strategy_count": multi_strategy_count,
            "analysis": "；".join(quality_reasons),
        }

    def check_logic_consistency(
        self,
        recommendations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        检查推荐逻辑是否一致

        Args:
            recommendations: 推荐结果

        Returns:
            逻辑一致性检查结果
        """
        issues = []

        for code, rec_info in recommendations.items():
            # 检查 1：高分但低置信度
            score = rec_info.get("score", 0)
            confidence = rec_info.get("ai_confidence", 0)
            if score >= 70 and confidence < 0.5:
                issues.append(f"{code}: 分数高({score:.1f})但置信度低({confidence:.2f})")

            # 检查 2：技术面和基本面严重不一致
            tech_score = rec_info.get("technical_score")
            fund_score = rec_info.get("fundamental_score")
            if tech_score is None or fund_score is None:
                missing_fields = []
                if tech_score is None:
                    missing_fields.append("technical_score")
                if fund_score is None:
                    missing_fields.append("fundamental_score")
                issues.append(f"{code}: 缺少评分字段({', '.join(missing_fields)})")
            elif abs(tech_score - fund_score) > 30:
                issues.append(
                    f"{code}: 技术面({tech_score:.1f})和基本面({fund_score:.1f})差异大"
                )

            # 检查 3：有冲突但还是推荐
            has_conflict = rec_info.get("has_conflict", False)
            if has_conflict and score >= 70:
                issues.append(f"{code}: 存在维度冲突但仍高分推荐")

        return {
            "total_recommendations": len(recommendations),
            "issue_count": len(issues),
            "issues": issues,
            "consistency_ok": len(issues) == 0,
        }

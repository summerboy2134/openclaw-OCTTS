"""Technical indicators calculation."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class TechnicalSnapshot:
    close: float
    ma5: Optional[float]
    ma10: Optional[float]
    ma20: Optional[float]
    ma60: Optional[float]
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    volume_ratio: float
    price_position_20d: Optional[float]
    breakout: Optional[str]
    trend_status: str
    momentum_status: str
    technical_score: float
    trend_score: float
    momentum_score: float
    volume_score: float
    breakout_score: float
    risk_score: float
    setup_quality_score: float
    recommendation_score: float
    recommendation: str
    setup_type: str
    risk_level: str
    entry_style: str
    confidence: str
    risk_flags: List[str]
    setup_notes: List[str]
    distance_to_ma20_pct: Optional[float]
    distance_to_ma60_pct: Optional[float]
    breakout_strength: float


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均线"""
    return prices.rolling(window=period, min_periods=1).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线"""
    return prices.ewm(span=period, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """计算相对强弱指标(RSI)"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(prices: pd.Series,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> Dict[str, pd.Series]:
    """计算MACD指标"""
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def calculate_volume_ratio(volumes: pd.Series, period: int = 5) -> float:
    """计算量比"""
    if len(volumes) < 2:
        return 1.0

    current_volume = volumes.iloc[-1]
    avg_volume = volumes.iloc[:-1].tail(period).mean()

    if avg_volume > 0:
        return current_volume / avg_volume
    return 1.0


def calculate_bollinger_bands(prices: pd.Series,
                              period: int = 20,
                              std_dev: float = 2) -> Dict[str, pd.Series]:
    """计算布林带"""
    middle = calculate_sma(prices, period)
    std = prices.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def is_golden_cross(ma_short: pd.Series, ma_long: pd.Series) -> bool:
    """判断是否出现金叉"""
    if len(ma_short) < 2 or len(ma_long) < 2:
        return False
    return ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]


def is_death_cross(ma_short: pd.Series, ma_long: pd.Series) -> bool:
    """判断是否出现死叉"""
    if len(ma_short) < 2 or len(ma_long) < 2:
        return False
    return ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-2] >= ma_long.iloc[-2]


def calculate_price_breakout(prices: pd.Series,
                             period: int = 20,
                             threshold: float = 0.02) -> Dict[str, Any]:
    """计算价格突破信号"""
    if len(prices) < period + 1:
        return {
            "is_breakout": False,
            "breakout_type": None,
            "resistance": None,
            "support": None,
        }

    recent_prices = prices.tail(period)
    current_price = prices.iloc[-1]
    resistance = recent_prices.max()
    support = recent_prices.min()

    if current_price > resistance * (1 + threshold):
        return {
            "is_breakout": True,
            "breakout_type": "upward",
            "resistance": resistance,
            "support": support,
            "breakout_strength": (current_price - resistance) / resistance,
        }

    if current_price < support * (1 - threshold):
        return {
            "is_breakout": True,
            "breakout_type": "downward",
            "resistance": resistance,
            "support": support,
            "breakout_strength": (support - current_price) / support,
        }

    return {
        "is_breakout": False,
        "breakout_type": None,
        "resistance": resistance,
        "support": support,
    }


def calculate_kdj(high: pd.Series,
                  low: pd.Series,
                  close: pd.Series,
                  period: int = 9) -> Dict[str, pd.Series]:
    """计算KDJ指标"""
    low_min = low.rolling(window=period, min_periods=1).min()
    high_max = high.rolling(window=period, min_periods=1).max()
    rsv = 100 * (close - low_min) / (high_max - low_min)

    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    return {
        "k": k,
        "d": d,
        "j": j,
    }


def calculate_price_position(close: pd.Series,
                             high: pd.Series,
                             low: pd.Series,
                             period: int = 20) -> pd.Series:
    rolling_low = low.rolling(window=period, min_periods=period).min()
    rolling_high = high.rolling(window=period, min_periods=period).max()
    spread = (rolling_high - rolling_low).replace(0, np.nan)
    return ((close - rolling_low) / spread).clip(lower=0, upper=1)


def get_ma_trend_status(ma5: Optional[float],
                        ma10: Optional[float],
                        ma20: Optional[float],
                        ma60: Optional[float],
                        close: float) -> str:
    if None not in (ma5, ma10, ma20) and ma5 > ma10 > ma20:
        if ma60 is None or ma20 >= ma60:
            return "bullish"
        return "improving"
    if None not in (ma5, ma10, ma20) and ma5 < ma10 < ma20:
        return "bearish"
    if ma20 is not None and close >= ma20:
        return "improving"
    return "mixed"


def get_macd_status(macd: Optional[float],
                    signal: Optional[float],
                    histogram: Optional[float],
                    prev_histogram: Optional[float] = None) -> str:
    if macd is None or signal is None or histogram is None:
        return "neutral"
    if macd >= signal and histogram >= 0:
        if prev_histogram is not None and histogram > prev_histogram:
            return "bullish_rising"
        return "bullish"
    if macd < signal and histogram < 0:
        if prev_histogram is not None and histogram > prev_histogram:
            return "bearish_improving"
        return "bearish"
    return "neutral"


def get_rsi_status(rsi: Optional[float]) -> str:
    if rsi is None:
        return "neutral"
    if rsi <= 30:
        return "oversold"
    if rsi <= 45:
        return "weak"
    if rsi < 65:
        return "neutral"
    if rsi < 80:
        return "strong"
    return "overbought"


def calculate_technical_score(close: float,
                              ma5: Optional[float],
                              ma10: Optional[float],
                              ma20: Optional[float],
                              ma60: Optional[float],
                              rsi: Optional[float],
                              macd_status: str,
                              volume_ratio: float,
                              price_position_20d: Optional[float],
                              breakout_type: Optional[str]) -> float:
    score = 0.0

    trend_status = get_ma_trend_status(ma5, ma10, ma20, ma60, close)
    if trend_status == "bullish":
        score += 35
    elif trend_status == "improving":
        score += 24
    elif trend_status == "mixed":
        score += 12

    if macd_status == "bullish_rising":
        score += 25
    elif macd_status == "bullish":
        score += 20
    elif macd_status == "bearish_improving":
        score += 10

    if rsi is not None:
        if 45 <= rsi <= 65:
            score += 20
        elif 30 <= rsi < 45 or 65 < rsi <= 75:
            score += 12
        elif rsi < 30:
            score += 6

    if volume_ratio >= 2:
        score += 10
    elif volume_ratio >= 1.2:
        score += 6

    if breakout_type == "upward":
        score += 10
    elif price_position_20d is not None:
        if price_position_20d >= 0.7:
            score += 8
        elif price_position_20d >= 0.5:
            score += 5
        elif price_position_20d >= 0.3:
            score += 2

    return round(min(score, 100.0), 2)


def _clamp_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _calculate_distance_pct(close: float, moving_average: Optional[float]) -> Optional[float]:
    if moving_average is None or moving_average == 0:
        return None
    return round((close - moving_average) / moving_average * 100, 2)


def _calculate_trend_score(close: float,
                           ma20: Optional[float],
                           ma60: Optional[float],
                           trend_status: str) -> float:
    score_map = {
        "bullish": 85.0,
        "improving": 68.0,
        "mixed": 50.0,
        "bearish": 22.0,
    }
    score = score_map.get(trend_status, 45.0)
    if ma20 is not None and close >= ma20:
        score += 8
    else:
        score -= 10
    if ma60 is not None and close >= ma60:
        score += 7
    else:
        score -= 8
    if None not in (ma20, ma60) and ma20 >= ma60:
        score += 6
    elif None not in (ma20, ma60):
        score -= 6
    return _clamp_score(score)


def _calculate_momentum_score(momentum_status: str, rsi: Optional[float]) -> float:
    score_map = {
        "bullish_rising": 88.0,
        "bullish": 76.0,
        "neutral": 52.0,
        "bearish_improving": 38.0,
        "bearish": 18.0,
    }
    score = score_map.get(momentum_status, 50.0)
    if rsi is not None:
        if 50 <= rsi <= 68:
            score += 8
        elif 40 <= rsi < 50 or 68 < rsi <= 75:
            score += 2
        elif 30 <= rsi < 40:
            score -= 6
        elif rsi < 30:
            score -= 2
        elif rsi > 80:
            score -= 12
    return _clamp_score(score)


def _calculate_volume_score(volume_ratio: float) -> float:
    if volume_ratio >= 2.5:
        return 88.0
    if volume_ratio >= 1.8:
        return 76.0
    if volume_ratio >= 1.2:
        return 62.0
    if volume_ratio >= 0.9:
        return 48.0
    if volume_ratio >= 0.7:
        return 34.0
    return 20.0


def _calculate_breakout_score(breakout_type: Optional[str],
                              breakout_strength: float,
                              price_position_20d: Optional[float],
                              close: float,
                              ma20: Optional[float],
                              ma60: Optional[float]) -> float:
    score = 42.0
    if breakout_type == "upward":
        score = 82.0 + min(breakout_strength * 600, 12.0)
    elif breakout_type == "downward":
        score = 10.0
    elif price_position_20d is not None:
        if price_position_20d >= 0.85:
            score = 72.0
        elif price_position_20d >= 0.65:
            score = 60.0
        elif price_position_20d >= 0.35:
            score = 48.0
        elif price_position_20d >= 0.15:
            score = 34.0
        else:
            score = 26.0
    if ma20 is not None and close >= ma20:
        score += 4
    if ma60 is not None and close >= ma60:
        score += 4
    return _clamp_score(score)


def _calculate_risk_score(close: float,
                          ma20: Optional[float],
                          ma60: Optional[float],
                          rsi: Optional[float],
                          volume_ratio: float,
                          trend_status: str,
                          breakout_type: Optional[str],
                          distance_to_ma20_pct: Optional[float],
                          distance_to_ma60_pct: Optional[float],
                          history_length: int) -> float:
    score = 18.0
    if history_length < 60:
        score += 16
    if trend_status == "bearish":
        score += 28
    elif trend_status == "mixed":
        score += 12
    if breakout_type == "downward":
        score += 28
    if ma20 is not None and close < ma20:
        score += 12
    if ma60 is not None and close < ma60:
        score += 14
    if rsi is not None:
        if rsi >= 78:
            score += 16
        elif rsi >= 72:
            score += 10
        elif rsi <= 30:
            score += 10
    if volume_ratio < 0.8:
        score += 10
    elif volume_ratio < 1.0:
        score += 6
    if distance_to_ma20_pct is not None and distance_to_ma20_pct >= 10:
        score += 12
    elif distance_to_ma20_pct is not None and distance_to_ma20_pct >= 6:
        score += 7
    if distance_to_ma60_pct is not None and distance_to_ma60_pct >= 18:
        score += 8
    return _clamp_score(score)


def _classify_setup_type(trend_status: str,
                         breakout_type: Optional[str],
                         price_position_20d: Optional[float],
                         close: float,
                         ma20: Optional[float],
                         ma60: Optional[float],
                         rsi: Optional[float]) -> str:
    above_ma20 = ma20 is not None and close >= ma20
    above_ma60 = ma60 is not None and close >= ma60
    if breakout_type == "upward":
        return "breakout_candidate"
    if trend_status in {"bullish", "improving"} and above_ma20 and above_ma60:
        if price_position_20d is not None and 0.35 <= price_position_20d <= 0.65:
            return "pullback_in_uptrend"
        if price_position_20d is not None and price_position_20d >= 0.65:
            return "trend_continuation"
    if rsi is not None and rsi <= 35 and price_position_20d is not None and price_position_20d <= 0.3:
        return "oversold_rebound"
    if trend_status == "bearish" or breakout_type == "downward":
        return "weak_bearish"
    return "mixed_range"


def _classify_recommendation(recommendation_score: float) -> str:
    if recommendation_score >= 80:
        return "strong_buy_watchlist"
    if recommendation_score >= 68:
        return "buy_watchlist"
    if recommendation_score >= 55:
        return "monitor"
    if recommendation_score >= 40:
        return "weak_or_early"
    return "avoid"


def _classify_risk_level(risk_score: float) -> str:
    if risk_score >= 60:
        return "high"
    if risk_score >= 35:
        return "medium"
    return "low"


def _classify_entry_style(setup_type: str,
                          breakout_type: Optional[str],
                          risk_score: float,
                          distance_to_ma20_pct: Optional[float]) -> str:
    if breakout_type == "upward":
        if distance_to_ma20_pct is not None and distance_to_ma20_pct >= 8:
            return "avoid_chasing"
        return "breakout_follow"
    if setup_type == "pullback_in_uptrend":
        return "pullback_preferred"
    if setup_type == "oversold_rebound":
        return "rebound_probe"
    if risk_score >= 55 or setup_type == "weak_bearish":
        return "wait_confirmation"
    return "wait_confirmation"


def _classify_confidence(history_length: int,
                         trend_status: str,
                         momentum_status: str,
                         recommendation_score: float) -> str:
    if history_length < 30:
        return "low"
    if recommendation_score >= 75 and trend_status in {"bullish", "improving"} and momentum_status in {"bullish_rising", "bullish"}:
        return "high"
    if recommendation_score >= 55:
        return "medium"
    return "low"


def _build_risk_flags(history_length: int,
                      rsi: Optional[float],
                      volume_ratio: float,
                      trend_status: str,
                      breakout_type: Optional[str],
                      close: float,
                      ma20: Optional[float],
                      ma60: Optional[float],
                      distance_to_ma20_pct: Optional[float]) -> List[str]:
    flags: List[str] = []
    if history_length < 60:
        flags.append("insufficient_history")
    if rsi is not None and rsi >= 78:
        flags.append("overbought")
    elif rsi is not None and rsi <= 30:
        flags.append("oversold")
    if trend_status == "bearish":
        flags.append("bearish_trend")
    if breakout_type == "downward":
        flags.append("downward_breakout")
    if ma20 is not None and close < ma20:
        flags.append("below_ma20")
    if ma60 is not None and close < ma60:
        flags.append("below_ma60")
    if volume_ratio < 0.8:
        flags.append("weak_volume")
    if distance_to_ma20_pct is not None and distance_to_ma20_pct >= 8:
        flags.append("extended_from_ma20")
    return flags


def _build_setup_notes(setup_type: str,
                       trend_status: str,
                       momentum_status: str,
                       recommendation: str,
                       breakout_type: Optional[str],
                       price_position_20d: Optional[float],
                       distance_to_ma20_pct: Optional[float]) -> List[str]:
    notes: List[str] = []
    notes.append(f"setup={setup_type}")
    notes.append(f"trend={trend_status}")
    notes.append(f"momentum={momentum_status}")
    notes.append(f"recommendation={recommendation}")
    if breakout_type == "upward":
        notes.append("出现向上突破信号")
    elif breakout_type == "downward":
        notes.append("出现向下破位信号")
    if price_position_20d is not None:
        if price_position_20d >= 0.7:
            notes.append("价格位于20日区间高位")
        elif price_position_20d <= 0.3:
            notes.append("价格位于20日区间低位")
    if distance_to_ma20_pct is not None and distance_to_ma20_pct >= 8:
        notes.append("价格偏离MA20较大，注意追高风险")
    return notes


def build_technical_snapshot(close: pd.Series,
                             high: pd.Series,
                             low: pd.Series,
                             volume: pd.Series) -> TechnicalSnapshot:
    ma5_series = calculate_sma(close, 5)
    ma10_series = calculate_sma(close, 10)
    ma20_series = calculate_sma(close, 20)
    ma60_series = calculate_sma(close, 60)
    rsi_series = calculate_rsi(close)
    macd_data = calculate_macd(close)
    price_position = calculate_price_position(close, high, low, 20)
    breakout = calculate_price_breakout(close, 20)
    volume_ratio = float(calculate_volume_ratio(volume)) if not volume.empty else 1.0

    latest_close = float(close.iloc[-1]) if not close.empty else 0.0
    ma5 = float(ma5_series.iloc[-1]) if len(close) >= 5 else None
    ma10 = float(ma10_series.iloc[-1]) if len(close) >= 10 else None
    ma20 = float(ma20_series.iloc[-1]) if len(close) >= 20 else None
    ma60 = float(ma60_series.iloc[-1]) if len(close) >= 60 else None
    rsi = float(rsi_series.iloc[-1]) if len(close) >= 14 else None
    macd = float(macd_data["macd"].iloc[-1]) if len(close) >= 26 else None
    macd_signal = float(macd_data["signal"].iloc[-1]) if len(close) >= 26 else None
    macd_histogram = float(macd_data["histogram"].iloc[-1]) if len(close) >= 26 else None
    prev_histogram = float(macd_data["histogram"].iloc[-2]) if len(close) >= 27 else None
    price_position_20d = float(price_position.iloc[-1]) if len(close) >= 20 and pd.notna(price_position.iloc[-1]) else None
    breakout_type = breakout.get("breakout_type") if breakout.get("is_breakout") else None
    breakout_strength = float(breakout.get("breakout_strength", 0.0) or 0.0)
    trend_status = get_ma_trend_status(ma5, ma10, ma20, ma60, latest_close)
    momentum_status = get_macd_status(macd, macd_signal, macd_histogram, prev_histogram)
    technical_score = calculate_technical_score(
        latest_close,
        ma5,
        ma10,
        ma20,
        ma60,
        rsi,
        momentum_status,
        volume_ratio,
        price_position_20d,
        breakout_type,
    )
    distance_to_ma20_pct = _calculate_distance_pct(latest_close, ma20)
    distance_to_ma60_pct = _calculate_distance_pct(latest_close, ma60)
    trend_score = _calculate_trend_score(latest_close, ma20, ma60, trend_status)
    momentum_score = _calculate_momentum_score(momentum_status, rsi)
    volume_score = _calculate_volume_score(volume_ratio)
    breakout_score = _calculate_breakout_score(
        breakout_type,
        breakout_strength,
        price_position_20d,
        latest_close,
        ma20,
        ma60,
    )
    risk_score = _calculate_risk_score(
        latest_close,
        ma20,
        ma60,
        rsi,
        volume_ratio,
        trend_status,
        breakout_type,
        distance_to_ma20_pct,
        distance_to_ma60_pct,
        len(close),
    )
    setup_quality_score = _clamp_score(
        trend_score * 0.35
        + momentum_score * 0.25
        + volume_score * 0.15
        + breakout_score * 0.25
    )
    recommendation_score = _clamp_score(
        trend_score * 0.28
        + momentum_score * 0.22
        + volume_score * 0.12
        + breakout_score * 0.22
        + (100 - risk_score) * 0.16
    )
    setup_type = _classify_setup_type(
        trend_status,
        breakout_type,
        price_position_20d,
        latest_close,
        ma20,
        ma60,
        rsi,
    )
    recommendation = _classify_recommendation(recommendation_score)
    risk_level = _classify_risk_level(risk_score)
    entry_style = _classify_entry_style(
        setup_type,
        breakout_type,
        risk_score,
        distance_to_ma20_pct,
    )
    confidence = _classify_confidence(
        len(close),
        trend_status,
        momentum_status,
        recommendation_score,
    )
    risk_flags = _build_risk_flags(
        len(close),
        rsi,
        volume_ratio,
        trend_status,
        breakout_type,
        latest_close,
        ma20,
        ma60,
        distance_to_ma20_pct,
    )
    setup_notes = _build_setup_notes(
        setup_type,
        trend_status,
        momentum_status,
        recommendation,
        breakout_type,
        price_position_20d,
        distance_to_ma20_pct,
    )

    return TechnicalSnapshot(
        close=latest_close,
        ma5=ma5,
        ma10=ma10,
        ma20=ma20,
        ma60=ma60,
        rsi=rsi,
        macd=macd,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        volume_ratio=volume_ratio,
        price_position_20d=price_position_20d,
        breakout=breakout_type,
        trend_status=trend_status,
        momentum_status=momentum_status,
        technical_score=technical_score,
        trend_score=trend_score,
        momentum_score=momentum_score,
        volume_score=volume_score,
        breakout_score=breakout_score,
        risk_score=risk_score,
        setup_quality_score=setup_quality_score,
        recommendation_score=recommendation_score,
        recommendation=recommendation,
        setup_type=setup_type,
        risk_level=risk_level,
        entry_style=entry_style,
        confidence=confidence,
        risk_flags=risk_flags,
        setup_notes=setup_notes,
        distance_to_ma20_pct=distance_to_ma20_pct,
        distance_to_ma60_pct=distance_to_ma60_pct,
        breakout_strength=round(breakout_strength, 4),
    )

from datetime import date
from unittest.mock import Mock

from octts.config import Settings
from octts.services.short_term_training_data import ShortTermTrainingDataBuilder


def test_build_samples_for_trade_date_maps_pool_state_and_labels() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    store = Mock()
    store.load_recommendation_pool_state.return_value = [
        {
            "ts_code": "688010.SH",
            "name": "福光股份",
            "source_tag": "今日Top3",
            "in_frontlist": True,
            "recommend_rank": 1,
            "strategy_count": 2,
            "is_repeat_pick": True,
            "news_mentioned": False,
            "technical_signal": "趋势改善",
            "close": 30.9,
            "pct_change": 6.92,
            "volume_ratio": 2.4,
            "turnover_rate": 3.38,
            "recommendation_score": 59.56,
            "overall_score": 66.98,
            "technical_score": 87.0,
            "fundamental_score": 27.0,
            "industry_heat_score": -1.02,
            "industry_flow_bias": "偏弱",
            "distribution_risk_score": 0.0,
            "distribution_risk_flags": ["近3日资金承接偏弱"],
            "moneyflow_3d_value": 5506.3,
            "continuation_bias_score": 3.6,
            "continuation_positive_flags": ["多策略共振"],
            "continuation_negative_flags": ["板块热度偏弱"],
            "score_change": 10.3,
            "action_plan": {"action_bias": "回避"},
        }
    ]
    store.list_recommendation_run_items.return_value = [
        {
            "ts_code": "688010.SH",
            "entry_price": 30.9,
            "return_1d": 0.021,
            "return_3d": 0.03,
            "return_5d": None,
            "return_10d": None,
            "max_drawdown_10d": -0.015,
            "benchmark_return_5d": None,
            "vs_benchmark_5d": None,
        }
    ]

    builder = ShortTermTrainingDataBuilder(settings, store=store)
    samples = builder.build_samples_for_trade_date(date(2026, 4, 7))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.ts_code == "688010.SH"
    assert sample.label_up_1d is True
    assert sample.return_1d == 0.021
    assert sample.continuation_bias_score == 3.6
    assert sample.distribution_risk_flags == ["近3日资金承接偏弱"]


def test_build_samples_for_trade_date_falls_back_to_future_snapshots_for_labels() -> None:
    settings = Settings(OCTTS_MEMORY_BACKEND="file", OCTTS_MEMORY_FILE_PATH="memory.json")
    store = Mock()
    store.load_recommendation_pool_state.return_value = [
        {
            "ts_code": "688010.SH",
            "name": "福光股份",
            "source_tag": "今日Top3",
            "in_frontlist": True,
            "recommend_rank": 1,
            "close": 30.9,
            "entry_price": 30.9,
        }
    ]
    store.list_recommendation_run_items.return_value = []

    tushare_client = Mock()
    tushare_client.fetch_trading_dates.return_value = ["20260403", "20260407", "20260408", "20260409"]
    tushare_client.get_or_build_screening_snapshot.side_effect = [
        {
            "daily_basic": {
                "000300.SH": {"close": 3982.0},
            }
        },
        {
            "daily_basic": {
                "688010.SH": {"close": 30.05},
            }
        },
        {
            "daily_basic": {
                "688010.SH": {"close": 31.00},
            }
        },
        {
            "daily_basic": {
                "688010.SH": {"close": 31.50},
                "000300.SH": {"close": 4000.0},
            }
        },
    ]

    builder = ShortTermTrainingDataBuilder(settings, store=store, tushare_client=tushare_client)
    samples = builder.build_samples_for_trade_date(date(2026, 4, 3))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.return_1d == (30.05 - 30.9) / 30.9
    assert sample.return_3d == (31.50 - 30.9) / 30.9
    assert sample.label_up_1d is False

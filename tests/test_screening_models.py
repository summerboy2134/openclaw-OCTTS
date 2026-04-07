from datetime import date

from octts.models.screening_models import DatabaseManager
from octts.schemas.screener import TrackedRecommendationState


def test_upsert_recommendation_pool_states_persists_continuation_fields(tmp_path) -> None:
    db_path = tmp_path / "screening.db"
    manager = DatabaseManager(f"sqlite:///{db_path}")

    state = TrackedRecommendationState(
        ts_code="000001.SZ",
        trade_date=date(2026, 4, 1),
        continuation_bias_score=2.4,
        continuation_positive_flags=["3日资金承接偏强"],
        continuation_negative_flags=["近5日涨幅偏大"],
        top3_risk_penalty=8.4,
        short_term_contradiction_penalty=3.0,
        final_display_recommendation_score=45.61,
    )

    persisted = manager.upsert_recommendation_pool_states([state])

    assert persisted[0]["continuation_bias_score"] == 2.4
    assert persisted[0]["continuation_positive_flags"] == ["3日资金承接偏强"]
    assert persisted[0]["continuation_negative_flags"] == ["近5日涨幅偏大"]
    assert persisted[0]["top3_risk_penalty"] == 8.4
    assert persisted[0]["short_term_contradiction_penalty"] == 3.0
    assert persisted[0]["final_display_recommendation_score"] == 45.61

    loaded = manager.load_recommendation_pool_state(trade_date=date(2026, 4, 1))
    assert loaded[0]["continuation_bias_score"] == 2.4
    assert loaded[0]["continuation_positive_flags"] == ["3日资金承接偏强"]
    assert loaded[0]["continuation_negative_flags"] == ["近5日涨幅偏大"]
    assert loaded[0]["top3_risk_penalty"] == 8.4
    assert loaded[0]["short_term_contradiction_penalty"] == 3.0
    assert loaded[0]["final_display_recommendation_score"] == 45.61

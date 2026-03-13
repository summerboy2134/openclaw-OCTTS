from octts.schemas.report import MemorySummary
from octts.services.memory_store import FileMemoryStore


def test_file_memory_store_round_trip(tmp_path) -> None:
    store = FileMemoryStore(str(tmp_path / "memory.json"))
    summary = MemorySummary(
        ts_code="600000.SH",
        phase="review",
        trend_bias="neutral",
        capital_flow_view="capital waiting for breakout",
        confidence_score=0.6,
        summary="sideways consolidation",
        key_risks=["break below support"],
    )

    store.set(summary)
    loaded = store.get("600000.SH")

    assert loaded is not None
    assert loaded.ts_code == "600000.SH"
    assert loaded.key_risks == ["break below support"]


def test_file_memory_store_delete_and_clear(tmp_path) -> None:
    store = FileMemoryStore(str(tmp_path / "memory.json"))
    first = MemorySummary(
        ts_code="600000.SH",
        phase="review",
        trend_bias="neutral",
        capital_flow_view="capital waiting for breakout",
        confidence_score=0.6,
        summary="sideways consolidation",
    )
    second = MemorySummary(
        ts_code="000001.SZ",
        phase="review",
        trend_bias="bullish",
        capital_flow_view="capital inflow improving",
        confidence_score=0.7,
        summary="trend constructive",
    )

    store.set(first)
    store.set(second)
    store.delete("600000.SH")

    assert store.get("600000.SH") is None
    assert store.get("000001.SZ") is not None

    store.clear()
    assert store.get("000001.SZ") is None

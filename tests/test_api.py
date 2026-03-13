from fastapi.testclient import TestClient

from octts.api import app
from octts.config import Settings
from octts.schemas.backtest import BacktestMetrics, BacktestResult
from octts.schemas.report import (
    DecisionValidation,
    HistoricalAnalysisRecord,
    MemorySummary,
    PriceSnapshot,
    PriceZone,
    StructuredAnalysis,
    TradingDecision,
)
from octts.services.history_store import FileHistoryStore
from octts.services.memory_store import FileMemoryStore


def _build_record() -> HistoricalAnalysisRecord:
    snapshot = PriceSnapshot(
        ts_code="600000.SH",
        trade_date="20260309",
        close=10.2,
        high=10.3,
        low=10.0,
    )
    report = StructuredAnalysis(
        ts_code="600000.SH",
        phase="review",
        trend_judgement="等待向上突破",
        previous_view_status="initial",
        operation_advice="靠近支撑时分批关注",
        risk_warning=["若跌破 9.8 需止损"],
        observation_points=["关注 10.5 压力位"],
        summary_markdown="等待突破。",
        decision=TradingDecision(
            signal="buy",
            rationale="支撑有效且量能温和修复。",
            entry_zone=PriceZone(low=10.0, high=10.2),
            stop_loss=9.8,
            take_profit=[10.5],
            invalidation_condition="放量跌破 9.8",
            holding_horizon="swing",
            confidence_score=0.72,
            risk_reward_ratio=1.8,
            evidence=["支撑位有效"],
        ),
        memory=MemorySummary(
            ts_code="600000.SH",
            phase="review",
            trend_bias="bullish",
            capital_flow_view="主力资金小幅回流",
            confidence_score=0.72,
            summary="等待突破确认",
        ),
    )
    return HistoricalAnalysisRecord(
        record_id="r1",
        request_id="req1",
        generated_at=report.memory.generated_at,
        snapshot=snapshot,
        report=report,
        validation=DecisionValidation(status="entered", note="已进入建议区间。", entry_triggered=True),
    )


def test_dashboard_data_returns_latest_cards(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="",
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.get("/dashboard/data")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["cards"]) == 1
    assert payload["cards"][0]["ts_code"] == "600000.SH"
    assert payload["default_stock_pool"] == []
    assert "openclaw_status" in payload


def test_dashboard_route_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "OCTTS Dashboard" in response.text
    assert 'id="backtestForm"' in response.text
    assert 'id="backtestTemplateSelect"' in response.text
    assert "回撤曲线" in response.text


def test_root_redirects_to_dashboard() -> None:
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/dashboard"


def test_favicon_returns_empty_response() -> None:
    client = TestClient(app)
    response = client.get("/favicon.ico")

    assert response.status_code == 204
    assert response.text == ""


def test_stock_detail_data_returns_symbol_payload(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_FILE_PATH=str(history_path),
            OCTTS_HISTORY_LIMIT_PER_SYMBOL=30,
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
            OPENCLAW_GATEWAY_URL="http://127.0.0.1:18789",
            OPENCLAW_HOOKS_ENABLED=True,
        ),
    )

    client = TestClient(app)
    response = client.get("/stocks/600000.SH/data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"]["ts_code"] == "600000.SH"
    assert payload["openclaw_status"]["connected"] is True


def test_stock_detail_page_returns_html() -> None:
    client = TestClient(app)
    response = client.get("/stocks/600000.SH")

    assert response.status_code == 200
    assert "600000.SH 单股详情" in response.text


def test_openclaw_status_endpoint_uses_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_AUTOMATION_ENABLED=True,
            OCTTS_AUTOMATION_TIMEZONE="Asia/Shanghai",
            OPENCLAW_AGENT_ID="octts",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH="memory/latest_memory.json",
        ),
    )

    client = TestClient(app)
    response = client.get("/openclaw/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["automation_enabled"] is True
    assert payload["automation_timezone"] == "Asia/Shanghai"
    assert len(payload["automation_slots"]) == 1
    assert payload["automation_slots"][0]["phase"] == "review"


def test_backtest_endpoint_returns_result(monkeypatch) -> None:
    class DummyBacktestEngine:
        def run(self, request):
            assert request.phase == "review"
            return BacktestResult(
                phase="review",
                stock_pool=["600000.SH"],
                start_date="20260101",
                end_date="20260110",
                initial_cash=100000,
                ending_cash=101000,
                metrics=BacktestMetrics(trade_count=1, total_return=0.01),
            )

    monkeypatch.setattr("octts.api._build_backtest_engine", lambda: DummyBacktestEngine())

    client = TestClient(app)
    response = client.post(
        "/backtest",
        json={
            "stock_pool": ["600000.SH"],
            "start_date": "20260101",
            "end_date": "20260110",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["phase"] == "review"
    assert payload["metrics"]["trade_count"] == 1


def test_add_stock_pool_item_persists_to_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OCTTS_STOCK_POOL=600000.SH\n", encoding="utf-8")

    monkeypatch.setattr("octts.api._env_file_path", lambda: env_path)
    monkeypatch.setattr("octts.api._clear_settings_cache", lambda: None)
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="600000.SH",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.post("/stock-pool", json={"ts_code": "000001.sz"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["stock_pool"] == ["600000.SH", "000001.SZ"]
    assert "OCTTS_STOCK_POOL=600000.SH,000001.SZ" in env_path.read_text(encoding="utf-8")


def test_remove_stock_pool_item_persists_to_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OCTTS_STOCK_POOL=600000.SH,000001.SZ\n", encoding="utf-8")

    monkeypatch.setattr("octts.api._env_file_path", lambda: env_path)
    monkeypatch.setattr("octts.api._clear_settings_cache", lambda: None)
    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_STOCK_POOL="600000.SH,000001.SZ",
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(tmp_path / "memory.json"),
        ),
    )

    client = TestClient(app)
    response = client.delete("/stock-pool/000001.SZ")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stock_pool"] == ["600000.SH"]
    assert "OCTTS_STOCK_POOL=600000.SH" in env_path.read_text(encoding="utf-8")


def test_clear_symbol_analysis_data_removes_history_and_memory(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history"
    memory_path = tmp_path / "memory.json"
    store = FileHistoryStore(str(history_path))
    store.append(_build_record())
    memory_store = FileMemoryStore(str(memory_path))
    memory_store.set(_build_record().report.memory)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(history_path),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(memory_path),
        ),
    )

    client = TestClient(app)
    response = client.delete("/analysis-data/600000.SH")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cleared_symbols"] == ["600000.SH"]
    assert payload["removed_records"] == 1
    assert FileHistoryStore(str(history_path)).list_records("600000.SH") == []
    assert FileMemoryStore(str(memory_path)).get("600000.SH") is None


def test_clear_all_analysis_data_removes_everything(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "history"
    memory_path = tmp_path / "memory.json"
    store = FileHistoryStore(str(history_path))
    first = _build_record()
    second = _build_record()
    second.report.ts_code = "000001.SZ"
    second.snapshot.ts_code = "000001.SZ"
    second.report.memory.ts_code = "000001.SZ"
    store.append(first)
    store.append(second)

    memory_store = FileMemoryStore(str(memory_path))
    memory_store.set(first.report.memory)
    memory_store.set(second.report.memory)

    monkeypatch.setattr(
        "octts.api.get_settings",
        lambda: Settings(
            OCTTS_HISTORY_DIR_PATH=str(history_path),
            OCTTS_MEMORY_BACKEND="file",
            OCTTS_MEMORY_FILE_PATH=str(memory_path),
        ),
    )

    client = TestClient(app)
    response = client.delete("/analysis-data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cleared_all"] is True
    assert payload["removed_records"] == 2
    assert payload["removed_memory_items"] == 2
    assert FileHistoryStore(str(history_path)).list_latest() == []
    assert FileMemoryStore(str(memory_path)).get("600000.SH") is None
    assert FileMemoryStore(str(memory_path)).get("000001.SZ") is None

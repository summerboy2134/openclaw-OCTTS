from __future__ import annotations

from fastapi import FastAPI, HTTPException

from octts.api_legacy import _build_backtest_engine, _build_pipeline
from octts.schemas.backtest import BacktestRequest, BacktestResult
from octts.schemas.report import AnalysisRequest, AnalysisResult


def register_analysis_routes(app: FastAPI) -> None:
    @app.post("/analyze", response_model=AnalysisResult)
    def analyze(request: AnalysisRequest) -> AnalysisResult:
        try:
            pipeline = _build_pipeline()
            return pipeline.run(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}") from exc

    @app.post("/backtest", response_model=BacktestResult)
    def backtest(request: BacktestRequest) -> BacktestResult:
        try:
            engine = _build_backtest_engine()
            return engine.run(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Backtest failed: {exc}") from exc

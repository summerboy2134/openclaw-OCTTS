from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from octts.config import get_settings
from octts.schemas.api import LightweightBacktestRequest
from octts.services.lightweight_backtester import LightweightBacktester
from octts.services.stock_screener import StockScreener


def register_backtest_routes(app: FastAPI) -> None:
    @app.get("/backtest")
    def backtest_page() -> HTMLResponse:
        from octts.ui.backtest_page import render_backtest_page

        return HTMLResponse(content=render_backtest_page())

    @app.post("/api/backtest")
    async def run_backtest(request: LightweightBacktestRequest) -> Dict[str, Any]:
        try:
            backtester = LightweightBacktester(get_settings())
            all_presets = StockScreener.get_presets()
            selected_strategies = [preset for preset in all_presets if preset.id in request.strategies]
            if not selected_strategies:
                raise HTTPException(status_code=400, detail="No valid strategies selected")

            results = backtester.compare_strategies(
                strategies=selected_strategies,
                start_date=request.start_date,
                end_date=request.end_date,
                holding_days=request.holding_days,
                top_n=request.top_n,
                commission_rate=request.commission_rate,
                slippage_rate=request.slippage_rate,
                stock_pool=request.stock_pool,
            )

            best_strategy = max(results.items(), key=lambda x: x[1].total_return)
            stock_scope = (
                f"限定股票池：{', '.join(request.stock_pool)}"
                if request.stock_pool
                else "全市场中命中当前策略条件的股票"
            )
            summary = {
                "period": f"{request.start_date} - {request.end_date}",
                "best_strategy": best_strategy[0],
                "best_total_return": best_strategy[1].total_return,
                "best_max_drawdown": best_strategy[1].max_drawdown,
                "best_sharpe_ratio": best_strategy[1].sharpe_ratio,
                "commission_rate": request.commission_rate,
                "slippage_rate": request.slippage_rate,
                "stock_scope": stock_scope,
                "selected_stock_pool": request.stock_pool,
                "recommendation": (
                    f"{stock_scope}下，建议关注{best_strategy[0]}策略，"
                    f"总收益{best_strategy[1].total_return:.1f}%，"
                    f"胜率{best_strategy[1].win_rate:.1%}"
                ),
            }

            formatted_results = {}
            for name, result in results.items():
                equity = 1.0
                peak = 1.0
                equity_curve = []
                sorted_records = sorted(result.detail_records, key=lambda item: item.get("entry_date", ""))
                for record in sorted_records:
                    record_return = float(record.get("return_pct", 0.0)) / 100.0
                    equity *= 1 + record_return
                    peak = max(peak, equity)
                    drawdown = 0.0 if peak <= 0 else (equity - peak) / peak
                    equity_curve.append(
                        {
                            "trade_date": record.get("entry_date") or record.get("signal_date") or "",
                            "value": round(equity, 4),
                            "drawdown": round(drawdown * 100, 2),
                        }
                    )

                formatted_results[name] = {
                    "total_trades": result.total_trades,
                    "winning_trades": result.winning_trades,
                    "losing_trades": result.losing_trades,
                    "win_rate": result.win_rate,
                    "avg_return": result.avg_return,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "sharpe_ratio": result.sharpe_ratio,
                    "detail_records": sorted_records,
                    "equity_curve": equity_curve,
                }

            return {
                "results": formatted_results,
                "summary": summary,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc

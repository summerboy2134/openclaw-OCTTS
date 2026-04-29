from .analysis_routes import register_analysis_routes
from .backtest_routes import register_backtest_routes
from .base_routes import register_base_routes
from .dashboard_routes import register_dashboard_routes
from .portfolio_routes import register_portfolio_routes
from .screening_routes import register_screening_routes

__all__ = [
    "register_analysis_routes",
    "register_backtest_routes",
    "register_base_routes",
    "register_dashboard_routes",
    "register_portfolio_routes",
    "register_screening_routes",
]

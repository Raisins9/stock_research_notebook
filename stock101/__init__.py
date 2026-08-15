"""stock101: QQQ vs SPY research analysis toolkit.

This package provides pure, unit-tested functions for the standard
performance and risk metrics used in the companion research notebook.
"""

from .metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    beta,
    capm_alpha,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
    total_return,
)

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "annualized_volatility",
    "beta",
    "capm_alpha",
    "daily_returns",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
]

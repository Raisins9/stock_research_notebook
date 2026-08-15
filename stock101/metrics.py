"""Core performance and risk metrics.

All functions in this module are *pure*: they take numeric inputs
(pandas Series / numpy arrays) and return Python floats, so they are
straightforward to unit test without any network or disk I/O.

Conventions and assumptions (see README.md for the full rationale):

* Returns are *simple* (arithmetic) daily returns, ``r_t = p_t / p_{t-1} - 1``.
* Volatility, Sharpe ratio and alpha are annualized with
  :data:`TRADING_DAYS_PER_YEAR` = 252 trading days.
* The risk-free rate is a *annualized* simple rate; it is divided by 252 to
  obtain the daily rate. It defaults to 0.0.
* ``max_drawdown`` is reported as a *negative* fraction, e.g. ``-0.35``
  means a 35% peak-to-trough decline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Number of trading days used to annualize daily statistics.
TRADING_DAYS_PER_YEAR: int = 252


def _as_series(values: pd.Series | np.ndarray, name: str) -> pd.Series:
    """Coerce ``values`` to a float pandas Series and validate it."""
    series = pd.Series(values, dtype="float64")
    if series.isna().any():
        raise ValueError(f"{name} must not contain NaN values.")
    if len(series) == 0:
        raise ValueError(f"{name} must not be empty.")
    return series


def daily_returns(prices: pd.Series | np.ndarray) -> pd.Series:
    """Compute simple daily returns from a price series.

    The result has one fewer observation than ``prices`` and is aligned to
    the *return* date (i.e. the date on which the return was realized).
    """
    prices = _as_series(prices, "prices")
    if len(prices) < 2:
        raise ValueError("prices must contain at least two observations.")
    return prices.pct_change().dropna()


def total_return(prices: pd.Series | np.ndarray) -> float:
    """Buy-and-hold total return over the full price history.

    ``(P_end / P_start) - 1``. When ``prices`` are adjusted closes, dividends
    and splits are already reflected, so this is a *total* (not price) return.
    """
    prices = _as_series(prices, "prices")
    if len(prices) < 2:
        raise ValueError("prices must contain at least two observations.")
    if prices.iloc[0] == 0:
        raise ValueError("The first price is zero; total return is undefined.")
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def annualized_volatility(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized volatility (standard deviation of returns).

    Uses the *sample* standard deviation (ddof=1) of daily returns scaled by
    ``sqrt(periods_per_year)``, under the standard i.i.d.-returns assumption.
    """
    returns = _as_series(returns, "returns")
    if len(returns) < 2:
        return float("nan")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(prices: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a price series.

    Returns the most negative drawdown as a *negative* fraction. For example,
    ``-0.35`` means the strategy lost 35% from a prior peak to a subsequent
    trough before recovering.
    """
    prices = _as_series(prices, "prices")
    if len(prices) < 2:
        return float("nan")
    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1.0
    return float(drawdown.min())


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio of a return series.

    ``(mean daily excess return / std daily excess return) * sqrt(n)`` where
    ``n = periods_per_year``. The daily risk-free rate is
    ``risk_free_rate / periods_per_year``.

    Returns ``nan`` when volatility is zero (the ratio is undefined) rather
    than raising, so a flat asset does not crash an analysis loop.
    """
    returns = _as_series(returns, "returns")
    if len(returns) < 2 or periods_per_year <= 0:
        return float("nan")
    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def beta(
    asset_returns: pd.Series | np.ndarray,
    market_returns: pd.Series | np.ndarray,
) -> float:
    """Beta of an asset relative to a market index.

    ``cov(asset, market) / var(market)`` estimated from aligned daily returns
    with ddof=1. ``market_returns`` should be the market proxy (SPY here).
    """
    asset = _as_series(asset_returns, "asset_returns")
    market = _as_series(market_returns, "market_returns")
    if len(asset) != len(market):
        raise ValueError(
            "asset_returns and market_returns must be aligned and equal length."
        )
    if len(asset) < 2:
        return float("nan")
    cov = np.cov(asset.to_numpy(), market.to_numpy(), ddof=1)
    market_var = cov[1, 1]
    if market_var == 0 or np.isnan(market_var):
        return float("nan")
    return float(cov[0, 1] / market_var)


def capm_alpha(
    asset_returns: pd.Series | np.ndarray,
    market_returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized CAPM (Jensen's) alpha.

    ``alpha = (mean excess asset return - beta * mean excess market return)
    * periods_per_year``. A positive alpha means the asset earned more than
    CAPM predicts for its beta, on an annualized basis.
    """
    asset = _as_series(asset_returns, "asset_returns")
    market = _as_series(market_returns, "market_returns")
    if len(asset) != len(market):
        raise ValueError(
            "asset_returns and market_returns must be aligned and equal length."
        )
    if len(asset) < 2 or periods_per_year <= 0:
        return float("nan")

    daily_rf = risk_free_rate / periods_per_year
    excess_asset = asset - daily_rf
    excess_market = market - daily_rf
    b = beta(asset, market)
    daily_alpha = excess_asset.mean() - b * excess_market.mean()
    return float(daily_alpha * periods_per_year)

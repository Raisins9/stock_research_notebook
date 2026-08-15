"""Unit tests for stock101.metrics.

These tests are fully offline: they use small, hand-constructed series whose
expected metric values are computable by inspection or by independent formulas.
"""

import numpy as np
import pandas as pd
import pytest

from stock101.metrics import (
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    beta,
    capm_alpha,
    daily_returns,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


# --------------------------------------------------------------------------- #
# daily_returns
# --------------------------------------------------------------------------- #
def test_daily_returns_simple_arithmetic():
    prices = pd.Series([100.0, 110.0, 99.0])
    expected = pd.Series(
        [0.10, -0.10],
        index=pd.Index([1, 2]),
    )
    result = daily_returns(prices)
    pd.testing.assert_series_equal(result, expected)


def test_daily_returns_drops_first_observation():
    prices = pd.Series([1.0, 2.0, 3.0])
    assert len(daily_returns(prices)) == len(prices) - 1


def test_daily_returns_rejects_short_series():
    with pytest.raises(ValueError):
        daily_returns(pd.Series([1.0]))


# --------------------------------------------------------------------------- #
# total_return
# --------------------------------------------------------------------------- #
def test_total_return_gain():
    assert total_return(pd.Series([100.0, 150.0])) == pytest.approx(0.50)


def test_total_return_loss():
    assert total_return(pd.Series([100.0, 40.0])) == pytest.approx(-0.60)


def test_total_return_ignores_path():
    # Only endpoints matter: (end/start) - 1.
    up_down = pd.Series([100.0, 200.0, 100.0])
    flat = pd.Series([100.0, 100.0])
    assert total_return(up_down) == pytest.approx(0.0)
    assert total_return(flat) == pytest.approx(0.0)


def test_total_return_zero_start_raises():
    with pytest.raises(ValueError):
        total_return(pd.Series([0.0, 10.0]))


# --------------------------------------------------------------------------- #
# annualized_volatility
# --------------------------------------------------------------------------- #
def test_volatility_zero_for_flat_series():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert annualized_volatility(returns) == pytest.approx(0.0)


def test_volatility_scales_by_sqrt_of_periods():
    # Two return observations with a known sample std.
    returns = pd.Series([0.01, -0.01])
    daily_std = returns.std(ddof=1)
    assert annualized_volatility(returns, periods_per_year=4) == pytest.approx(
        daily_std * 2.0
    )
    assert annualized_volatility(returns) == pytest.approx(
        daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def test_volatility_single_observation_is_nan():
    assert np.isnan(annualized_volatility(pd.Series([0.05])))


# --------------------------------------------------------------------------- #
# max_drawdown
# --------------------------------------------------------------------------- #
def test_max_drawdown_known_sequence():
    prices = pd.Series([100.0, 120.0, 90.0, 95.0, 80.0])
    # Peak 120 -> trough 80 => -33.33%.
    assert max_drawdown(prices) == pytest.approx(80.0 / 120.0 - 1.0)


def test_max_drawdown_flat_is_zero():
    assert max_drawdown(pd.Series([10.0, 10.0, 10.0])) == pytest.approx(0.0)


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(pd.Series([10.0, 20.0, 30.0])) == pytest.approx(0.0)


def test_max_drawdown_is_negative_fraction():
    prices = pd.Series([100.0, 50.0])
    assert max_drawdown(prices) == pytest.approx(-0.50)


# --------------------------------------------------------------------------- #
# sharpe_ratio
# --------------------------------------------------------------------------- #
def test_sharpe_ratio_positive_excess():
    # Positive mean and non-zero variance -> positive Sharpe.
    returns = pd.Series([0.01, 0.02, 0.015, 0.03, 0.025])
    assert sharpe_ratio(returns, risk_free_rate=0.0) > 0


def test_sharpe_ratio_zero_volatility_is_nan():
    returns = pd.Series([0.02, 0.02, 0.02])
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_respects_risk_free_rate():
    # Returns with a mean of exactly 0.1%/day and non-zero variance, compared
    # against a 0.1%/day risk-free rate -> zero excess -> zero Sharpe.
    returns = pd.Series([0.0, 0.002, 0.001, 0.001])
    assert returns.mean() == pytest.approx(0.001)
    sharpe = sharpe_ratio(returns, risk_free_rate=0.252)
    assert sharpe == pytest.approx(0.0, abs=1e-6)


def test_sharpe_ratio_annualization_factor():
    returns = pd.Series([0.02, -0.01, 0.03, 0.0])
    excess = returns - 0.0
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# beta
# --------------------------------------------------------------------------- #
def test_beta_identical_series_is_one():
    market = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    assert beta(market, market) == pytest.approx(1.0)


def test_beta_scaled_series_is_scale():
    # An asset that is exactly 2x the market every day has beta == 2.
    market = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    asset = market * 2.0
    assert beta(asset, market) == pytest.approx(2.0)


def test_beta_mismatched_length_raises():
    with pytest.raises(ValueError):
        beta(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# capm_alpha
# --------------------------------------------------------------------------- #
def test_alpha_zero_when_capm_holds_exactly():
    # Construct an asset that is CAPM-consistent: r_a = beta * r_m.
    market = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    b = 1.5
    asset = b * market
    assert capm_alpha(asset, market, risk_free_rate=0.0) == pytest.approx(0.0, abs=1e-12)


def test_alpha_is_annualized():
    # Asset = market + 0.1%/day constant excess -> beta 1, alpha 25.2% annual.
    market = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    asset = market + 0.001
    assert capm_alpha(asset, market, risk_free_rate=0.0) == pytest.approx(
        0.001 * 252
    )


def test_alpha_mismatched_length_raises():
    with pytest.raises(ValueError):
        capm_alpha(pd.Series([1.0]), pd.Series([1.0, 2.0]))


# --------------------------------------------------------------------------- #
# input validation
# --------------------------------------------------------------------------- #
def test_nan_inputs_rejected():
    with pytest.raises(ValueError):
        total_return(pd.Series([1.0, np.nan]))
    with pytest.raises(ValueError):
        beta(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))

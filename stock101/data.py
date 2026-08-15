"""Data access: download adjusted prices and cache them locally.

The download uses ``yfinance`` and requests *adjusted* close prices. Local
CSV caching keeps runs reproducible and lets the notebook (and tests) run
offline after the first successful download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

#: Column order written to / read from the cache.
_PRICE_COLUMNS = ["QQQ", "SPY"]


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download daily *adjusted* close prices from Yahoo Finance.

    ``auto_adjust=True`` makes the ``Close`` column already adjusted for
    dividends and splits (Yahoo's "Adj Close" semantics). Returns a DataFrame
    of one price column per ticker, indexed by date (timezone-naive).
    """
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data is None or data.empty:
        raise RuntimeError(f"No data returned for {tickers} ({start}..{end}).")

    # With a single ticker the frame is 2-D; with multiple tickers Close is
    # itself a DataFrame. Normalize both shapes to a DataFrame.
    if "Close" in data.columns:
        close = data["Close"]
    else:
        close = data

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    # Drop any rows where every ticker is missing, keep column order stable.
    close = close[tickers].dropna(how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    return close


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data",
    fetcher: Callable[[list[str], str, str], pd.DataFrame] = fetch_prices,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, using a CSV cache.

    The cache file name is derived from the tickers and the date range, so
    changing the window or universe yields a fresh file. Pass ``refresh=True``
    to force a re-download.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_filename(tickers, start, end)

    if cache_path.exists() and not refresh:
        return _read_cache(cache_path, tickers)

    prices = fetcher(tickers, start, end)
    prices = prices[tickers].dropna(how="all")
    prices.to_csv(cache_path)
    return prices


def _cache_filename(tickers: list[str], start: str, end: str) -> str:
    """Deterministic cache file name for a (tickers, window) request."""
    joined = "-".join(tickers)
    return f"{joined}_{start}_{end}.csv"


def _read_cache(path: Path, tickers: list[str]) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices[tickers]
    prices.index.name = "Date"
    return prices

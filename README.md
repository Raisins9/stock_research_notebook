# stock101 — QQQ vs SPY Research Notebook

A small, reproducible Python project that downloads **daily adjusted close
prices** for QQQ and SPY, computes standard performance/risk metrics, and ships
with unit tests.

The metric math lives in a pure, testable module (`stock101/metrics.py`); the
Jupyter notebook (`stock_research.ipynb`) imports and displays those same
functions, so the notebook numbers and the test suite exercise identical code.

## Metrics computed

| Metric | Definition |
| --- | --- |
| **Total return** | Buy-and-hold `(P_end / P_start) - 1` over the full window. |
| **Annualized volatility** | Sample std (ddof=1) of daily returns × `√252`. |
| **Max drawdown** | Worst peak-to-trough decline, reported as a negative fraction. |
| **Sharpe ratio** | Annualized mean excess return ÷ std of excess return. |
| **Beta** | `Cov(r_a, r_m) / Var(r_m)` vs SPY (the market proxy). |
| **CAPM alpha** | Jensen's alpha, annualized: `(r̄_a − r_f − β(r̄_m − r_f)) × 252`. |

## Project structure

```
stock101/
├── stock101/
│   ├── __init__.py      # public exports
│   ├── metrics.py       # pure metric functions (unit-tested)
│   └── data.py          # yfinance download + local CSV cache
├── tests/
│   └── test_metrics.py  # 25 offline unit tests
├── stock_research.ipynb # the research notebook
├── build_notebook.py    # regenerates the .ipynb from source
├── pyproject.toml       # project + pytest config
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Create the environment and install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Run the unit tests (offline, no network needed)
.venv/bin/python -m pytest

# 3. Open the notebook
.venv/bin/python -m jupyter notebook stock_research.ipynb
```

The notebook downloads data from Yahoo Finance on first run and caches it to
`data/` (git-ignored), so later runs work offline. Set `refresh=True` (or delete
the cache) to force a re-download.

To regenerate the notebook after editing `build_notebook.py`:

```bash
.venv/bin/python build_notebook.py
```

## Assumptions & conventions

1. **Data source** — Yahoo Finance daily *adjusted* close
   (`yfinance` with `auto_adjust=True`). Adjusted close incorporates splits and
   reinvested dividends, so buy-and-hold total return is measured correctly.
2. **Return type** — *simple* (arithmetic) daily returns
   `r_t = p_t / p_{t-1} - 1`, not log returns.
3. **Annualization** — 252 trading days per year
   (`stock101.metrics.TRADING_DAYS_PER_YEAR`). Volatility and Sharpe are scaled
   by `√252`; CAPM alpha is scaled by `252`.
4. **Volatility** — *sample* standard deviation of daily returns (ddof=1).
5. **Risk-free rate** — 0.0% by default (an annualized simple rate, divided by
   252 to obtain the daily rate). Override via `RISK_FREE_RATE` in the notebook
   or the `risk_free_rate=` argument.
6. **Market proxy** — SPY is used as the market portfolio for beta and CAPM
   alpha.
7. **Beta estimation** — slope `Cov(r_a, r_m)/Var(r_m)` on *daily* returns over
   the tickers' common trading days. A daily-frequency beta can differ from a
   monthly-frequency beta.
8. **CAPM alpha** — Jensen's alpha, annualized (see table above). It is only
   meaningful conditional on CAPM holding and is noisy at daily frequency.
9. **Max drawdown sign** — reported as a *negative* fraction; e.g. `-0.35`
   means a 35% peak-to-trough decline.
10. **No frictions** — no transaction costs, taxes, or fees beyond what the ETF
    NAV already reflects; no survivorship adjustment.
11. **Missing data** — rows where every ticker is missing are dropped; NaN
    values inside a series are rejected with a clear error by the metric
    functions.

## Testing

```bash
.venv/bin/python -m pytest
```

Tests are fully offline: they feed small, hand-constructed series whose metric
values are known by inspection, and cover edge cases (zero volatility →
`nan` Sharpe, mismatched alignment, NaN inputs, single-observation inputs,
zero initial price).

## Notes

- Daily beta/alpha are sensitive to the sample window; treat them as
  descriptive, not predictive. Past performance is not indicative of future
  results.
- The default window is 2020-01-01 → 2024-12-31. Change `START`/`END` in the
  notebook or pass new values to `load_prices(...)`.

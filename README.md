# Which EMAs does price actually respect?

This notebook allows to test a full range of EMA periods against real market data from Bybit perpetual futures and tells you which ones the market actually treats as support or resistance on your chosen symbol and timeframe.

Use it to find the most relevant EMAs for the market you trade.

## What it does

For a user-supplied range of EMA periods, a symbol, an interval, and a date range, it counts per EMA:

- **low_touch** — candles whose Low came within delta of the EMA
- **high_touch** — candles whose High came within delta of the EMA
- **cross** — candles whose range straddled the EMA (`Low ≤ EMA ≤ High`)
- **above / below** — candles trading entirely above / below the EMA

…then resolves each cross's direction by the bar's open and computes per-direction quality counters:
support_test, support_held, resistance_test, resistance_held.

From these, it derives **13 interpretive ratios** in three families:

- Support quality / strict rejection / frequency / bullishness / regime (5 ratios)
- Resistance quality / strict rejection / frequency / bearishness / regime (5 ratios)
- Universal hold rate / bounce rate / tradability (3 ratios)

High ratios — especially when clustered across neighbouring EMA periods — suggest a price level that the market respects.

Two filters (cross-saturation and sample-size) suppress structurally degenerate or under-sampled EMAs.

The backtesting notebook then takes the picked EMAs and runs a full strategy with stops, take-profits, position sizing, fees, slippage, and parameter sweeps. Two routes are available: **Route A** backtests EMAs you choose by hand from notebook 2's rankings (head-to-head leaderboard); **Route B** runs walk-forward validation with an automated EMA picker on the train slice.

## Project layout

```
ema/
├── ema_core.py # shared fetch / analyze / filter / pick helpers
├── backtest.py # strategy engine (configs, backtest loop, metrics, sweeps)
├── 1_core_pipeline.ipynb # fetch + analyze
├── 2_ema_analysis.ipynb # rank EMAs as support / resistance / universal S/R
├── 3_ema_backtesting.ipynb # strategy backtest (Route A: manual EMAs / Route B: walk-forward + auto picker) + sweeps
├── ema_analysis.ipynb # original single-notebook version (legacy reference)
├── data/ # OHLC parquet cache (git-ignored)
│ └── klines_*.parquet # raw OHLC, one file per symbol/interval/date-range/category
├── README.md
├── LICENSE
└── .gitignore
```

## How to use

The three notebooks are designed to run **in order**. Notebook 1 caches the OHLC fetch; notebooks 2 and 3 reuse that cache and recompute the analysis directly (recompute is cheap — a few seconds for typical configs).

### 1. Setup (one-time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests pandas numpy pyarrow plotly jupyter
```

pyarrow is needed for the parquet cache.

### 2. Configure

The configuration block lives **only in `1_core_pipeline.ipynb`** and is the single source of truth. Edit values there, run the cell, and notebook 1 calls `save_config(...)` to persist them to `data/config.json`. Notebooks 2 and 3 read the same config back via `load_config()` — they have no editable config block of their own.

```python
symbol     = "BTCUSDT"
interval   = "15"               # "1","3","5","15","30","60","240","D","W","M"
start      = "2026-01-01"
end        = "2026-04-01"
ema_range  = range(1, 200, 1)
delta      = 20
delta_mode = "absolute"          # or "percent"
category   = "linear"            # "linear" (USDT perps) or "inverse" (coin-margined)

# Persist to data/config.json so notebooks 2 & 3 pick it up.
save_config(symbol, interval, start, end, ema_range, delta, delta_mode, category)
```

### 3. Run the notebooks

```bash
jupyter lab
```

Then in order:

| notebook | what it does | output |
|---|---|---|
| **`1_core_pipeline.ipynb`** | Defines and persists the config to `data/config.json`. Fetches OHLC from Bybit, runs analyze_ema_touches, caches the OHLC. | `data/klines_*.parquet`, `data/config.json` |
| **`2_ema_analysis.ipynb`** | Loads config via `load_config()`. Loads OHLC from cache, recomputes the analysis. Ranks EMAs as support / resistance / universal across 13 ratios. Visualises the picks. | rankings + plots |
| **`3_ema_backtesting.ipynb`** | Loads config via `load_config()`. Loads OHLC from cache, recomputes the analysis. Two routes: **A** — backtest a hand-picked list of EMAs from notebook 2 head-to-head; **B** — walk-forward split, picks the best EMA on the train slice (direction-aware), runs the strategy on train + test. Parameter sweeps work on either route's cfg. | trade results, equity curve, metrics |

Re-running notebook 1 with the **same** configuration loads OHLC from cache instantly. Changing symbol / interval / start / end / category produces a fresh OHLC cache entry alongside the old one. Changing only analysis parameters (ema_range, delta, delta_mode) skips the fetch and just re-runs analyze_ema_touches.

### 4. Force a refresh

If you want to re-fetch OHLC from Bybit (e.g. Bybit corrected historical data):

```python
df = load_or_fetch_klines(symbol, interval, start, end, category, force_refetch=True)
```

The analysis (analyze_ema_touches) is recomputed every run — no force-recompute flag needed.

## How the modules fit together

- **`ema_core.py`** is the shared module imported by all three notebooks. It owns the data pipeline (fetch + cache + analyze) plus the filter helper (filter_by_cross_rate for the cross-saturation guard) and the EMA pickers (pick_best_ema_support / _resistance / _universal). Only the OHLC fetch is cached on disk; analyze_ema_touches is recomputed each run.
- **`backtest.py`** is the strategy engine. It defines the config dataclasses (StopLossConfig, TakeProfitConfig, PositionSizingConfig, StrategyConfig), the `backtest()` function, metrics, plots, walk-forward split, and the parameter sweeps (sweep_configs, sweep_grid).

Both are pure modules — import what you need, call from any notebook or script.

## Delta modes

- `"percent"` — tolerance scales with price (e.g. `delta=0.5` means 0.5% of EMA value). 
- `"absolute"` — fixed distance in quote currency (e.g. `delta=40` means $40 either side).

For BTC at $50k: `delta=0.5%` ≈ $250; `delta=40` (absolute) is much tighter.

## Filters

Two filters are applied to the hold-rate and rejection-rate ratios (ratios 1–2 in each section) and to the EMA pickers. The frequency / regime ratios (3, 4, 5) are reported unfiltered.

- **Cross-saturation guard** (`MAX_CROSS_RATE = 0.3`): drops EMAs that hug price too tightly to act as S/R (a fast EMA's range crosses on most bars, making the close-direction inequalities trivial).
- **Sample-size guard** (MIN_TOUCHES_*): require enough tests for the ratio to be statistically meaningful. Can be computed from the median of the relevant test column in result, so the threshold auto-scales with the dataset.

Both can be tightened or relaxed per-call by passing `max_cross_rate=...` and `min_touches=...` to the helpers.

## Conventions worth knowing

- EMA computation uses `ewm(span=period, adjust=False)` (Wilder / TradingView style). Switching to SMA or changing adjust would silently alter every downstream metric.
- All inequalities in the held / direction logic are **strict**. A tie (`open == EMA` or `close == EMA`) counts as neither — this avoids degeneracies (especially EMA1, where `close == EMA` always).
- Each cross is counted in exactly one direction (resolved by the bar's open), not double-counted as both a support and resistance test.
- A 0.1s sleep between paginated Bybit requests is intentional rate-limiting.

## Legacy

`ema_analysis.ipynb` is the original single-notebook version of the project. It's kept for reference but new work happens in the three split notebooks plus `ema_core.py`. The split version is faster to iterate on (no re-fetching for analysis tweaks) and easier to follow (each notebook has one job).

## License

MIT


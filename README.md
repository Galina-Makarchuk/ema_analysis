# Which EMAs does price actually respect?

This notebook allows to test a full range of EMA periods against real market data from ByBit perpetual futures and tells you which ones the market actually treats as support or resistance on your chosen symbol and timeframe.

Use it to find the most relevant EMAs for the market you trade.

## What it does

For a user-supplied range of EMA periods, a symbol, an interval, and a date range, it counts per EMA:

- **low_touch** — candles whose Low came within `delta` of the EMA
- **high_touch** — candles whose High came within `delta` of the EMA
- **cross** — candles whose range straddled the EMA (`Low ≤ EMA ≤ High`)
- **above / below** — candles trading entirely above / below the EMA

…then resolves each cross's direction by the bar's `open` and computes per-direction quality counters:
`support_test`, `support_held`, `resistance_test`, `resistance_held`.

From these, it derives **13 interpretive ratios** in three families:

- Support quality / strict rejection / frequency / bullishness / regime (5 ratios)
- Resistance quality / strict rejection / frequency / bearishness / regime (5 ratios)
- Universal hold rate / bounce rate / tradability (3 ratios)

High ratios — especially when clustered across neighbouring EMA periods — suggest a price level that the market respects.

Two filters (cross-saturation and sample-size) suppress structurally degenerate or under-sampled EMAs.

The backtesting notebook then takes the picked EMAs and runs a full strategy with stops, take-profits, position sizing, fees, slippage, walk-forward validation, and parameter sweeps.

## Project layout

```
ema/
├── ema_core.py # shared fetch / analyze / cache / filter / pick helpers
├── backtest.py # strategy engine (configs, backtest loop, metrics, sweeps)
├── 1_core_pipeline.ipynb # fetch + analyze + cache
├── 2_ema_analysis.ipynb # rank EMAs as support / resistance / universal S/R
├── 3_ema_backtesting.ipynb # walk-forward + strategy backtest + sweeps
├── ema_analysis.ipynb # original single-notebook version (legacy reference)
├── data/ # parquet caches (git-ignored)
│ ├── klines_*.parquet # raw OHLC, one per symbol/interval/range
│ └── result_*.parquet # analyze_ema_touches output, keyed by analysis config
├── README.md
├── LICENSE
└── .gitignore
```

## How to use

The three notebooks are designed to run **in order**. Each builds on the cached output of the previous one.

### 1. Setup (one-time)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests pandas numpy pyarrow plotly jupyter
```

`pyarrow` is needed for the parquet cache.

### 2. Configure

All arguments are configurable — replace them with values for your own market.
The same configuration block lives at the top of all three notebooks. 
It must match across them so the cache lookups succeed:

```python
symbol = "BTCUSDT"
interval = "15" # "1","3","5","15","30","60","240","D","W","M"
start = "2026-01-01"
end = "2026-04-01"
ema_range = range(1, 200, 1)
delta = 20
delta_mode = "absolute" # or "percent"
```

### 3. Run the notebooks

```bash
jupyter lab
```

Then in order:

| notebook | what it does | output |
|---|---|---|
| **`1_core_pipeline.ipynb`** | Fetches OHLC from ByBit, runs `analyze_ema_touches`, caches both. | `data/klines_*.parquet`, `data/result_*.parquet` |
| **`2_ema_analysis.ipynb`** | Loads caches. Ranks EMAs as support / resistance / universal across 13 ratios. Visualises the picks. | rankings + plots |
| **`3_ema_backtesting.ipynb`** | Loads caches. Walk-forward split, picks the best EMA on the train slice (direction-aware), runs the strategy on train + test, sweeps parameters. | trade results, equity curve, metrics |

Re-running notebook 1 with the **same** configuration is instant — it loads from cache. Changing any configuration field (especially `ema_range`, `delta`, `delta_mode`) creates a fresh cache entry alongside the old one (the result-cache filename includes a config hash).

### 4. Force a refresh

If you want to re-fetch from ByBit or recompute the analysis:

```python
df = load_or_fetch_klines(symbol, interval, start, end, force_refetch=True)
result = load_or_compute_result(df, symbol, interval, start, end,
ema_range, delta, delta_mode,
force_recompute=True)
```

## How the modules fit together

- **`ema_core.py`** is the shared module imported by all three notebooks. It owns the data pipeline (fetch + analyze + cache) plus the filter helper (`filter_by_cross_rate` for the cross-saturation guard) and the EMA pickers (`pick_best_ema_support / _resistance / _universal`).
- **`backtest.py`** is the strategy engine. It defines the config dataclasses (`StopLossConfig`, `TakeProfitConfig`, `PositionSizingConfig`, `StrategyConfig`), the `backtest()` function, metrics, plots, walk-forward split, and the parameter sweeps (`sweep_configs`, `sweep_grid`).

Both are pure modules — import what you need, call from any notebook or script.

## Delta modes

- `"percent"` — tolerance scales with price (e.g. `delta=0.5` means 0.5% of EMA value). 
- `"absolute"` — fixed distance in quote currency (e.g. `delta=40` means $40 either side).

For BTC at $50k: `delta=0.5%` ≈ $250; `delta=40` (absolute) is much tighter.

## Filters

Two filters are applied across every ratio table and the EMA pickers:

- **Cross-saturation guard** (`MAX_CROSS_RATE = 0.3`): drops EMAs that hug price too tightly to act as S/R (a fast EMA's range crosses on most bars, making the close-direction inequalities trivial).
- **Sample-size guard** (`MIN_TOUCHES_*`): require enough tests for the ratio to be statistically meaningful. Can be computed from the median of the relevant test column in `result`, so the threshold auto-scales with the dataset.

Both can be tightened or relaxed per-call by passing `max_cross_rate=...` and `min_touches=...` to the helpers.

## Conventions worth knowing

- EMA computation uses `ewm(span=period, adjust=False)` (Wilder / TradingView style). Switching to SMA or changing `adjust` would silently alter every downstream metric.
- All inequalities in the held / direction logic are **strict**. A tie (`open == EMA` or `close == EMA`) counts as neither — this avoids degeneracies (especially EMA1, where `close == EMA` always).
- Each `cross` is counted in exactly one direction (resolved by the bar's `open`), not double-counted as both a support and resistance test.
- A 0.1s sleep between paginated ByBit requests is intentional rate-limiting.

## Legacy

`ema_analysis.ipynb` is the original single-notebook version of the project. It's kept for reference but new work happens in the three split notebooks plus `ema_core.py`. The split version is faster to iterate on (no re-fetching for analysis tweaks) and easier to follow (each notebook has one job).

## License

MIT


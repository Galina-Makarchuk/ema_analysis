# Which EMA does price actually respect?

This notebook allows to test a full range of EMA periods against real market data from ByBit perpetual futures and tells you which ones the market actually treats as support or resistance on your chosen symbol and timeframe.

Use it to find the most relevant EMAs for the market you trade.

## What it does

For a user-supplied range of EMA periods, a symbol, an interval, and a date range, it counts per EMA:

- **low_touches** — candles whose Low came within `delta` of the EMA
- **high_touches** — candles whose High came within `delta` of the EMA
- **crosses** — candles whose range straddled the EMA (`Low ≤ EMA ≤ High`)
- **above / below** — candles trading entirely above / below the EMA

It then derives three interpretive ratios:

- **Support ratio** = `low_touches / crosses`
- **Resistance ratio** = `high_touches / crosses`
- **Universal ratio** = `(low_touches + high_touches) / crosses`

High ratios — especially when clustered across neighbouring EMA periods — suggest a price level that the market respects.

## Pipeline

Three stages, all implemented in `ema_analysis.ipynb`:

1. **Fetch** — `fetch_bybit_klines()` pulls candles from the ByBit v5 `/market/kline` endpoint, paginating backwards in 1000-candle chunks and returning a typed DataFrame `[timestamp, open, high, low, close, volume, turnover]`.
2. **Analyze** — `analyze_ema_touches()` computes `ewm(span=period, adjust=False)` (Wilder / TradingView-style EMA) for each period in the range and counts the metrics above. The first `period` candles per EMA are dropped as warmup by default.
3. **Run** — `run()` is a thin wrapper that chains fetch → analyze and returns the `df` and `result` DataFrames consumed by the downstream analysis sections.

## Delta modes

- `"percent"` — tolerance scales with price (recommended across price regimes)
- `"absolute"` — fixed distance in quote currency, e.g. USDT

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests pandas numpy plotly jupyter
jupyter lab ema_analysis.ipynb
```

Then inside the notebook (all arguments are configurable — replace them with values for your own market):

```python
result = run(
    symbol="BTCUSDT",
    interval="15",
    start="2026-01-01",
    end="2026-04-01",
    ema_range=range(1, 200),
    delta=50,
    delta_mode="absolute",
)
```

## Project layout

- `ema_analysis.ipynb` — the notebook (fetch, analyze, visualize, export)
- `df_*.csv` — cached raw candle data
- `ema_analysis_*.csv` — exported analysis output

## Notes

- Defaults to USDT-margined perps (`category="linear"`); pass `category="inverse"` for coin-margined contracts.
- A 0.1s sleep between paginated requests is intentional rate-limiting.
- `MIN_TOUCHES_*` thresholds in the analysis sections suppress ratios computed over tiny samples; they are deliberately tuned per dataset rather than hard-coded, since they scale with candle count and timeframe.
- Changing `adjust=False` or switching to SMA would silently alter every downstream metric — keep the EMA definition stable or update the interpretation cells alongside it.

## License

MIT


"""Shared functions and constants for the EMA support/resistance pipeline.

This module is the single source of truth for the pieces that are reused across
the three notebooks (1_core_pipeline, 2_ema_analysis, 3_ema_backtesting):

- Data fetch from ByBit
- Touch / cross analysis (the core algorithm)
- Cache helpers (parquet, keyed by symbol/interval/dates and analysis config)
- Filter helpers (cross-saturation guard, MIN_TOUCHES sample-size guard)
- Best-EMA pickers used by the backtesting walk-forward setup

Public API
----------
Constants:    BYBIT_API, DATA_DIR, CONFIG_PATH
Fetch:        fetch_bybit_klines, load_or_fetch_klines
Analyze:      analyze_ema_touches, run
Config:       save_config, load_config (single source of truth, written by NB1, read by NB2/NB3)
Filters:      filter_by_cross_rate
Helpers:      pick_best_ema_support, pick_best_ema_resistance, pick_best_ema_universal

Conventions worth preserving:

- EMA computation uses ``ewm(span=period, adjust=False)`` (Wilder/TradingView style).
  Switching to SMA, or changing ``adjust``, silently changes every downstream metric.
- Strict inequalities throughout the held / direction logic — a tie (``open == EMA``
  or ``close == EMA``) counts as neither held nor broken. This avoids the EMA1
  degeneracy where ``close == EMA`` always.
- Each cross is counted in exactly one direction, resolved by ``open`` vs EMA:
  ``cross_from_above = cross AND open > EMA`` (a support test),
  ``cross_from_below = cross AND open < EMA`` (a resistance test).
- ``series.replace(0, np.nan)`` before dividing (applied to each ratio's
  denominator — ``support_test``, ``resistance_test``, or their sum — in the
  pickers and notebooks) is load-bearing: it converts a 0-count denominator
  into NaN instead of raising or producing infinities.
- Warmup skipping is opt-out, not opt-in.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


# =============================================================================
# Constants
# =============================================================================

BYBIT_API = "https://api.bybit.com/v5/market/kline"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = DATA_DIR / "config.json"



# =============================================================================
# Fetch (raw OHLCV from ByBit)
# =============================================================================

def fetch_bybit_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    category: str,           # "linear" (USDT perps) or "inverse" (coin-margined)
) -> pd.DataFrame:
    """Pull kline (OHLCV) data from ByBit v5, with automatic backward pagination.

    Parameters
    ----------
    symbol     : e.g. ``"BTCUSDT"``.
    interval   : minute bucket as a string. One of
                 ``"1","3","5","15","30","60","120","240","360","720","D","W","M"``.
    start, end : ISO date strings, e.g. ``"2024-01-01"``. Both are interpreted
                 as midnight UTC. The returned DataFrame contains candles with
                 timestamp in ``[start, end]`` inclusive at the midnight boundary —
                 i.e. a candle whose open is exactly at ``end`` 00:00:00 UTC is
                 kept, but nothing after that. To extend through the full
                 end-of-day, see the inline comment on ``end_ts``.
    category   : ``"linear"`` for USDT-margined perps,
                 ``"inverse"`` for coin-margined perps.

    Returns
    -------
    DataFrame with columns ``[timestamp, open, high, low, close, volume, turnover]``,
    sorted ascending by timestamp, deduplicated.

    Notes
    -----
    ByBit caps each request at 1000 candles. We page backwards from ``end``,
    sleeping 0.1s between pages to be polite. ``retCode != 0`` raises ``RuntimeError``;
    a date range yielding zero candles raises ``ValueError``.
    """
    # Convert dates to milliseconds
    # end date is exclusive
    # to extend to end-of-day (so include the end date): end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) + 86_400_000 - 1
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

    rows: list = []
    cursor_end = end_ts

    while True:
        params = {
            "category": category,
            "symbol": symbol.upper(),
            "interval": interval,
            "start": start_ts,
            "end": cursor_end,
            "limit": 1000,          # ByBit max per request
        }
        r = requests.get(BYBIT_API, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        if data.get("retCode") != 0:
            raise RuntimeError(f"ByBit API error: {data}")

        batch = data["result"]["list"]      # newest first
        if not batch:
            break

        rows.extend(batch)

        # Result_list is sorted DESCENDING (newest candle first)
        oldest_ts = int(batch[-1][0])
        if oldest_ts <= start_ts or len(batch) < 1000:
            break
        # Move backward to fetch older candles
        cursor_end = oldest_ts - 1
        time.sleep(0.1)               # polite rate-limiting

    if not rows:
        raise ValueError(
            f"No candles returned for {symbol} {interval} between {start} and {end}"
        )

    # Create DataFrame
    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    # Convert types
    df = df.astype({c: float for c in cols[1:]})
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms", utc=True)

    # Remove any duplicates and sort ascending by time
    df = (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    
    # Filter exact date range (inclusive)
    df = df[
        (df["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (df["timestamp"] <= pd.Timestamp(end, tz="UTC"))
    ].reset_index(drop=True)

    return df


# =============================================================================
# Analyze (touch / cross counts per EMA)
# =============================================================================

# skip_warmup=True: ignore the first N candles (where N = the EMA period)
# Example: For EMA 50, it throws away the first 50 candles and only counts touches/cross starting from candle 51.
# Set it to False if you want to include every candle regardless.

# delta %: the allowed distance changes with price
# delta absolute: the allowed distance is always exactly the chosen 'number' regardless of price

def analyze_ema_touches(
    df: pd.DataFrame,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,                        # "percent" (of EMA) or "absolute"
    *,
    skip_warmup: bool = True,
) -> pd.DataFrame:
    """Count touch / cross / above / below behavior of every EMA in ``ema_range``.

    Per EMA period, every evaluated candle is placed in exactly ONE of these
    mutually exclusive buckets:

        cross      : Low <= EMA <= High           (range straddles the EMA)
        low_touch  : Low > EMA  AND  Low - EMA  <= delta
                     (entire candle above EMA, low close enough to register a touch)
        high_touch : High < EMA  AND  EMA - High <= delta
                     (entire candle below EMA, high close enough to register a touch)
        above      : Low > EMA  AND  Low - EMA  > delta
                     (entire candle above EMA, low far from EMA = clean trend)
        below      : High < EMA  AND  EMA - High > delta
                     (entire candle below EMA, high far from EMA = clean trend)

    Quality counters (used by the rejection-rate ratios downstream).

    A cross's direction is resolved by the bar's ``open`` relative to EMA:
        cross_from_above = cross AND open > EMA   # bar started above; range pierced down
        cross_from_below = cross AND open < EMA   # bar started below; range pierced up
    A tie (``open == EMA`` exactly) counts as neither — vanishingly rare with float prices.

        support_test    = low_touch  + cross_from_above                             # approaches from above
        support_held    = low_touch  + (cross_from_above AND close > EMA)           # ... that ended strictly above
        resistance_test = high_touch + cross_from_below                             # approaches from below
        resistance_held = high_touch + (cross_from_below AND close < EMA)           # ... that ended strictly below

    Invariants that always hold:
        cross + low_touch + high_touch + above + below = evaluated_candles
        any_touch        = low_touch + high_touch       (no double counting)
        cross_above + cross_below + cross_at_open_tie = cross
        support_held    <= support_test
        resistance_held <= resistance_test

    Parameters
    ----------
    df          : OHLC DataFrame as returned by ``fetch_bybit_klines``. Must have
                  ``open``, ``high``, ``low``, ``close`` columns.
    ema_range   : iterable of EMA periods to evaluate.
    delta       : tolerance value (units depend on ``delta_mode``).
    delta_mode  : ``"percent"`` (of EMA value) or ``"absolute"`` (fixed quote-currency).
    skip_warmup : drop the first ``period`` candles per EMA. The notebooks always pass True.

    Returns
    -------
    One row per EMA, with the columns described above plus
    ``cross_above``, ``cross_below``, ``evaluated_candles``.
    """
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]
    results: list[dict] = []

    for period in ema_range:
        # Compute EMA (pandas built-in, adjust=False = classic EMA)
        ema = close.ewm(span=period, adjust=False).mean()

        if delta_mode == "percent":
            tol = ema.abs() * (delta / 100.0)
        elif delta_mode == "absolute":
            tol = pd.Series(delta, index=ema.index)
        else:
            raise ValueError("delta_mode must be 'percent' or 'absolute'")

        # Optionally skip the first `period` candles while the EMA is warming up
        warmup = period if skip_warmup else 0
        mask = pd.Series(False, index=ema.index)
        mask.iloc[warmup:] = True

        # Top-level partition: cross / strictly-above / strictly-below
        crossed = (low <= ema) & (ema <= high) & mask
        strictly_above = (low > ema) & mask
        strictly_below = (high < ema) & mask

        # Touches are *near approaches that did not cross — *
        # split out from strictly_above / strictly_below by the delta criterion.
        low_touch = strictly_above & ((low - ema) <= tol)
        high_touch = strictly_below & ((ema - high) <= tol)
        any_touch = low_touch | high_touch

        # Clean trending candles (strictly past EMA, NOT a touch).
        above = strictly_above & ~low_touch
        below = strictly_below & ~high_touch

        # Resolve cross direction by the bar's OPEN relative to EMA.
        # A cross opened above EMA is a support test (price came down from above).
        # A cross opened below EMA is a resistance test (price came up from below).
        # Strict inequalities — a tie (open == EMA, or close == EMA on the held check)
        # counts as neither, which avoids the EMA1 degeneracy where close == EMA always.
        crossed_from_above = crossed & (open_ > ema)
        crossed_from_below = crossed & (open_ < ema)
        crossed_held_above = crossed_from_above & (close > ema)   # opened above, closed back above → support held
        crossed_held_below = crossed_from_below & (close < ema)   # opened below, closed back below → resistance held

        # Rejection-rate counters. Each cross counted in exactly one direction
        # (or neither, for ties). low_touch is strictly_above by definition, so
        # automatically has open > EMA and close > EMA — always a "support held".
        support_test = low_touch | crossed_from_above
        support_held = low_touch | crossed_held_above
        resistance_test = high_touch | crossed_from_below
        resistance_held = high_touch | crossed_held_below

        results.append(
            {
                "ema": period,
                "low_touch": int(low_touch.sum()),
                "high_touch": int(high_touch.sum()),
                "any_touch": int(any_touch.sum()),
                "above": int(above.sum()),
                "below": int(below.sum()),
                "cross": int(crossed.sum()),
                "cross_above": int(crossed_from_above.sum()),
                "cross_below": int(crossed_from_below.sum()),
                "support_test": int(support_test.sum()),
                "support_held": int(support_held.sum()),
                "resistance_test": int(resistance_test.sum()),
                "resistance_held": int(resistance_held.sum()),
                "evaluated_candles": int(mask.sum()),
            }
        )

    return pd.DataFrame(results)


def run(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    category: str,
    *,
    skip_warmup: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: fetch then analyze. Returns ``(df, result)``.

    Note: this does NOT use the parquet cache. For cached OHLC, use
    ``load_or_fetch_klines`` instead. ``result`` is recomputed each run
    (cheap — typically a few seconds for ~200 EMAs over a year of data).
    """
    print(f"Fetching {symbol} interval={interval} from {start} to {end} ...")
    df = fetch_bybit_klines(symbol, interval, start, end, category)
    print(f"  -> {len(df)} candles downloaded.")
    result = analyze_ema_touches(df, ema_range, delta, delta_mode, skip_warmup=skip_warmup)
    return df, result


# =============================================================================
# Cache helpers (parquet)
# =============================================================================

def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _klines_key(symbol: str, interval: str, start: str, end: str, category: str) -> str:
    return f"{symbol.lower()}_{interval}_{start}_{end}_{category}"


def klines_cache_path(
    symbol: str, interval: str, start: str, end: str, category: str
) -> Path:
    """Deterministic parquet path for cached OHLC."""
    _ensure_data_dir()
    return DATA_DIR / f"klines_{_klines_key(symbol, interval, start, end, category)}.parquet"


def load_or_fetch_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    category: str,
    *,
    force_refetch: bool = False,
) -> pd.DataFrame:
    """Return OHLC DataFrame from cache, fetching from ByBit if missing or forced."""
    path = klines_cache_path(symbol, interval, start, end, category)
    if path.exists() and not force_refetch:
        print(f"[ema_core] Loading klines cache: {path.name}")
        return pd.read_parquet(path)
    print(f"[ema_core] Fetching {symbol} {interval} {start} → {end} from ByBit ...")
    df = fetch_bybit_klines(symbol, interval, start, end, category)
    df.to_parquet(path)
    print(f"[ema_core] Saved klines cache: {path.name} ({len(df)} candles)")
    return df


# =============================================================================
# Config: single source of truth (written by notebook 1, read by 2 and 3)
# =============================================================================

def save_config(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    category: str,
) -> None:
    """Write the notebook-1 config to ``data/config.json`` so notebooks 2 & 3 can load it.

    ``ema_range`` is preserved as a ``range`` object across save/load (start/stop/step
    are stored explicitly). Any other iterable (list, tuple, ndarray) is stored as a
    flat list and loaded back as a list — both are accepted by ``analyze_ema_touches``.
    """
    _ensure_data_dir()
    if isinstance(ema_range, range):
        ema_range_payload = {
            "type": "range",
            "start": ema_range.start,
            "stop": ema_range.stop,
            "step": ema_range.step,
        }
    else:
        ema_range_payload = {"type": "list", "values": list(ema_range)}

    cfg = {
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end": end,
        "ema_range": ema_range_payload,
        "delta": delta,
        "delta_mode": delta_mode,
        "category": category,
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"[ema_core] Saved config: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")


def load_config() -> dict:
    """Load the config saved by notebook 1. Returns a dict ready to unpack.

    Raises ``FileNotFoundError`` with a clear message if notebook 1 hasn't been
    run yet (no ``data/config.json``).
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No config found at {CONFIG_PATH}. "
            f"Run 1_core_pipeline.ipynb first — it writes the config block "
            f"to disk so this notebook can pick it up."
        )
    cfg = json.loads(CONFIG_PATH.read_text())

    er = cfg["ema_range"]
    if isinstance(er, dict) and er.get("type") == "range":
        cfg["ema_range"] = range(er["start"], er["stop"], er["step"])
    elif isinstance(er, dict) and er.get("type") == "list":
        cfg["ema_range"] = er["values"]
    # else: legacy plain list — leave as is

    print(f"[ema_core] Loaded config: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    return cfg


def load_and_analyze(force_refetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read ``data/config.json``, load OHLC from cache (or fetch), run analyze.

    One-line pipeline reload for notebooks 2 and 3: replaces the load_config +
    load_or_fetch_klines + analyze_ema_touches sequence. Returns ``(df, result, cfg)``.

    Parameters
    ----------
    force_refetch : bool, default False
        Pass through to ``load_or_fetch_klines`` to skip the parquet cache and
        re-fetch OHLC from ByBit.
    """
    cfg = load_config()
    df = load_or_fetch_klines(
        cfg["symbol"], cfg["interval"], cfg["start"], cfg["end"],
        cfg["category"], force_refetch=force_refetch,
    )
    result = analyze_ema_touches(df, cfg["ema_range"], cfg["delta"], cfg["delta_mode"])
    return df, result, cfg


# =============================================================================
# Filter helpers
# =============================================================================

def filter_by_cross_rate(
    df: pd.DataFrame, max_cross_rate: float
) -> pd.DataFrame:
    """Add a ``cross_rate`` column and keep only rows below the threshold.

    The cross-saturation guard drops EMAs that hug price too tightly to act as
    S/R (a fast EMA's range crosses on most bars, making the close-direction
    inequalities trivial).

    Parameters
    ----------
    df             : must include ``cross`` and ``evaluated_candles`` columns.
    max_cross_rate : threshold (e.g. 0.3) — required, must come from the notebook config.
    """
    df = df.copy()
    df["cross_rate"] = df["cross"] / df["evaluated_candles"]
    return df[df["cross_rate"] < max_cross_rate]


# =============================================================================
# Best-EMA pickers (used by the backtesting walk-forward setup)
# =============================================================================

def _pick_best_ema(
    train_df: pd.DataFrame,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    *,
    test_col: str,
    held_col: str,
    out_col: str,
    min_touches: int,
    max_cross_rate: float,
) -> int:
    """Internal: shared body for support / resistance pickers.

    Picks the EMA with the highest hold rate (``held_col / test_col``) among rows
    that pass both the cross-saturation and sample-size filters.
    """
    res = analyze_ema_touches(train_df, ema_range, delta, delta_mode, skip_warmup=True)
    res = res[res["cross"] / res["evaluated_candles"] < max_cross_rate]
    res = res[res[test_col] >= min_touches].copy()
    if len(res) == 0:
        raise ValueError(
            f"No EMA met filters (min_touches={min_touches}, "
            f"max_cross_rate={max_cross_rate}) on train slice"
        )
    res[out_col] = res[held_col] / res[test_col].replace(0, np.nan)
    best = res.sort_values(out_col, ascending=False).iloc[0]
    return int(best["ema"])


def pick_best_ema_support(
    train_df: pd.DataFrame,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    min_touches: int,
    max_cross_rate: float,
) -> int:
    """EMA period with the highest support hold rate on the train slice.

    Hold rate = ``support_held / support_test`` (the canonical ``ratio_support_1`` metric).

    Parameters
    ----------
    train_df       : OHLC slice to evaluate on (e.g. the train portion of a walk-forward split).
    ema_range      : iterable of EMA periods to evaluate.
    delta, delta_mode : tolerance for touch detection (matches ``analyze_ema_touches``).
    min_touches    : minimum ``support_test`` count to consider an EMA (statistical guard).
    max_cross_rate : drop EMAs with ``cross / evaluated_candles >= max_cross_rate``.
    """
    return _pick_best_ema(
        train_df, ema_range, delta, delta_mode,
        test_col="support_test",
        held_col="support_held",
        out_col="ratio_support_1",
        min_touches=min_touches,
        max_cross_rate=max_cross_rate,
    )


def pick_best_ema_resistance(
    train_df: pd.DataFrame,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    min_touches: int,
    max_cross_rate: float,
) -> int:
    """EMA period with the highest resistance hold rate on the train slice.

    Hold rate = ``resistance_held / resistance_test`` (the canonical ``ratio_resistance_1`` metric).
    """
    return _pick_best_ema(
        train_df, ema_range, delta, delta_mode,
        test_col="resistance_test",
        held_col="resistance_held",
        out_col="ratio_resistance_1",
        min_touches=min_touches,
        max_cross_rate=max_cross_rate,
    )


def pick_best_ema_universal(
    train_df: pd.DataFrame,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    min_touches: int,
    max_cross_rate: float,
) -> int:
    """EMA period with the highest universal hold rate on the train slice.

    Hold rate = ``(support_held + resistance_held) / (support_test + resistance_test)``
    (the canonical ``ratio_universal_1`` metric — weighted average of both directions).

    ``min_touches`` is applied to ``support_test + resistance_test`` (the actual
    denominator of the universal ratio), not to support / resistance individually.
    """
    res = analyze_ema_touches(train_df, ema_range, delta, delta_mode, skip_warmup=True)
    res = res[res["cross"] / res["evaluated_candles"] < max_cross_rate]
    total_test = res["support_test"] + res["resistance_test"]
    res = res[total_test >= min_touches].copy()
    if len(res) == 0:
        raise ValueError(
            f"No EMA met filters (min_touches={min_touches}, "
            f"max_cross_rate={max_cross_rate}) on train slice"
        )
    res["ratio_universal_1"] = (
        (res["support_held"] + res["resistance_held"])
        / (res["support_test"] + res["resistance_test"]).replace(0, np.nan)
    )
    best = res.sort_values("ratio_universal_1", ascending=False).iloc[0]
    return int(best["ema"])

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
Constants:    BYBIT_API, DATA_DIR
Fetch:        fetch_bybit_klines, load_or_fetch_klines
Analyze:      analyze_ema_touches, load_or_compute_result, run
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
- ``crosses.replace(0, np.nan)`` before division is load-bearing — avoids div-by-zero
  spikes in the ratio columns.
- Warmup skipping is opt-out, not opt-in.
"""
from __future__ import annotations

import hashlib
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
    start, end : ISO date strings, e.g. ``"2024-01-01"``. ``end`` is exclusive at
                 the millisecond boundary; the returned DataFrame is filtered to
                 ``timestamp <= end`` inclusive.
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
    skip_warmup: bool,
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
    skip_warmup: bool,
    category: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: fetch then analyze. Returns ``(df, result)``.

    Note: this does NOT use the parquet cache. For cached, use ``load_or_fetch_klines``
    + ``load_or_compute_result`` instead, which is what the notebooks do.
    """
    print(f"Fetching {symbol} interval={interval} from {start} to {end} ...")
    df = fetch_bybit_klines(symbol, interval, start, end, category)
    print(f"  -> {len(df)} candles downloaded.")
    result = analyze_ema_touches(df, ema_range, delta, delta_mode, skip_warmup)
    return df, result


# =============================================================================
# Cache helpers (parquet)
# =============================================================================

def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _klines_key(symbol: str, interval: str, start: str, end: str, category: str) -> str:
    return f"{symbol.lower()}_{interval}_{start}_{end}_{category}"


def _result_config_hash(
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    skip_warmup: bool,
) -> str:
    payload = f"{list(ema_range)}|{delta}|{delta_mode}|{skip_warmup}"
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def klines_cache_path(
    symbol: str, interval: str, start: str, end: str, category: str
) -> Path:
    """Deterministic parquet path for cached OHLC."""
    _ensure_data_dir()
    return DATA_DIR / f"klines_{_klines_key(symbol, interval, start, end, category)}.parquet"


def result_cache_path(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    skip_warmup: bool,
    category: str,
) -> Path:
    """Deterministic parquet path for cached analyze_ema_touches output.

    Path includes a short hash of (ema_range, delta, delta_mode, skip_warmup) so
    changing any of those produces a fresh cache entry.
    """
    _ensure_data_dir()
    cfg_hash = _result_config_hash(ema_range, delta, delta_mode, skip_warmup)
    return DATA_DIR / (
        f"result_{_klines_key(symbol, interval, start, end, category)}_{cfg_hash}.parquet"
    )


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


def load_or_compute_result(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    ema_range: Iterable[int],
    delta: float,
    delta_mode: str,
    skip_warmup: bool,
    category: str,
    *,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Return analyze_ema_touches output from cache, recomputing if missing or forced.

    The cache key includes ``(symbol, interval, start, end, category)`` plus a
    short hash of ``(ema_range, delta, delta_mode, skip_warmup)`` — so changing
    any analysis parameter creates a fresh cache entry.
    """
    path = result_cache_path(
        symbol, interval, start, end, ema_range, delta, delta_mode, skip_warmup, category
    )
    if path.exists() and not force_recompute:
        print(f"[ema_core] Loading result cache: {path.name}")
        return pd.read_parquet(path)
    print(f"[ema_core] Computing analyze_ema_touches for {len(list(ema_range))} EMAs ...")
    result = analyze_ema_touches(df, ema_range, delta, delta_mode, skip_warmup)
    result.to_parquet(path)
    print(f"[ema_core] Saved result cache: {path.name} ({len(result)} rows)")
    return result


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

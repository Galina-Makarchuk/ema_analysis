"""Backtest engine for the EMA touch / cross strategy.

- single source of truth
- version-controllable
- importable from other notebooks too

Public API
----------
Configs:    StopLossConfig, TakeProfitConfig, PositionSizingConfig, StrategyConfig
Engine:     backtest(df, cfg) -> (trades_df, equity_curve)
Metrics:    compute_metrics(trades, equity_curve, initial_capital), print_metrics(metrics)
Plots:      plot_equity_curve(...), plot_trades_on_chart(...)
Slicing:    walk_forward_split(df, train_pct=0.7)
Sweeps:     sweep_configs(df, base_cfg, stop_pcts, r_targets)
            sweep_grid(df, base_cfg, grid)

Strategy
--------
Entry timing: at the close of the touch candle, with slippage applied.

LONG entry:  candle's low touches the entry EMA (within delta) and the candle
             closes back at/above the EMA (rejection).
             Optionally gated by close > EMA(regime_filter).

SHORT entry: mirror — candle's high touches the entry EMA (within delta) and
             the candle closes back at/below the EMA (rejection).
             Optionally gated by close < EMA(regime_filter).

direction='both' tries LONG first then SHORT on each bar. The two are effectively
mutually exclusive because a long requires close >= EMA and a short requires
close <= EMA, so both can fire only when close == EMA exactly (essentially never
with float prices).

Conservative exit: if a candle's range hits both stop and target, stop is taken.
End-of-data positions are force-closed at the last close.
"""

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StopLossConfig:
    """
    mode:
      - "fixed_value":  long  stop = entry - value
                        short stop = entry + value
      - "fixed_pct":    long  stop = entry * (1 - value/100)
                        short stop = entry * (1 + value/100)
      - "beyond_ema":   long  stop = ema_at_entry - value * delta_at_entry
                        short stop = ema_at_entry + value * delta_at_entry
      - "trailing":     long  stop = highest_high * (1 - value/100), ratchets up only
                        short stop = lowest_low   * (1 + value/100), ratchets down only
      - "chandelier":   long  stop = highest_high - value * ATR(atr_period), ratchets up only
                        short stop = lowest_low   + value * ATR(atr_period), ratchets down only
    """
    mode: Literal["fixed_value", "fixed_pct", "beyond_ema", "trailing", "chandelier"]
    value: float
    atr_period: int = 11


@dataclass
class TakeProfitConfig:
    """
    mode:
      - "fixed_value":  long  target = entry + value
                        short target = entry - value
      - "fixed_pct":    long  target = entry * (1 + value/100)
                        short target = entry * (1 - value/100)
      - "r_multiple":   long  target = entry + value * (entry - stop)
                        short target = entry - value * (stop - entry)
      - "trailing":     no fixed target; exit when price retraces by value% from the
                        highest-high (long) or lowest-low (short) since entry.
                        Activates only above entry (long) / below entry (short).
    """
    mode: Literal["fixed_value", "fixed_pct", "r_multiple", "trailing"]
    value: float


@dataclass
class PositionSizingConfig:
    """
    mode:
      - "fixed_value":  notional in quote currency per trade
      - "fixed_pct":    % of current equity per trade
    Trades are skipped if the computed size < `min_size`. The engine prints a
    warning the first time a skip occurs and a count summary at the end of the run.
    """
    mode: Literal["fixed_value", "fixed_pct"]
    value: float
    min_size: float = 0.001     # BTCUSDT minimum


@dataclass
class StrategyConfig:
    direction: Literal["long", "short", "both"] = "long"
    ema_period: int = 50
    delta: float = 40.0
    delta_mode: Literal["percent", "absolute"] = "absolute"
    stop_loss: StopLossConfig = field(default_factory=lambda: StopLossConfig("fixed_pct", 1.0))
    take_profit: TakeProfitConfig = field(default_factory=lambda: TakeProfitConfig("r_multiple", 3.0))
    position_sizing: PositionSizingConfig = field(default_factory=lambda: PositionSizingConfig("fixed_pct", 10.0))
    fee_pct: float = 0.055           # ByBit taker, in %
    slippage_pct: float = 0.01       # in % of price
    initial_capital: float = 10_000.0
    # Regime filter: longs only when close > EMA(N); shorts only when close < EMA(N).
    # Use a (typically slower) EMA distinct from the entry EMA. None = no filter.
    regime_filter: Optional[int] = None


# ============================================================================
# Helpers
# ============================================================================

def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ATR: RMA of true range (alpha = 1/period, adjust=False).

    Matches TradingView/Welles-Wilder ATR and the project's `ewm(..., adjust=False)`
    convention. Engine warmup already gates `atr_period` bars, so the unseeded
    EWM startup is masked.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift()
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _initial_long_stop(cfg, entry_price, ema_val, tol_val, high, atr_val):
    sl = cfg.stop_loss
    if sl.mode == "fixed_value":
        return entry_price - sl.value
    if sl.mode == "fixed_pct":
        return entry_price * (1 - sl.value / 100)
    if sl.mode == "beyond_ema":
        return ema_val - sl.value * tol_val
    if sl.mode == "trailing":
        return entry_price * (1 - sl.value / 100)
    if sl.mode == "chandelier":
        if atr_val is None or np.isnan(atr_val):
            raise ValueError("ATR not available for chandelier stop — increase warmup")
        return high - sl.value * atr_val
    raise ValueError(f"Unknown stop_loss mode: {sl.mode}")


def _initial_short_stop(cfg, entry_price, ema_val, tol_val, low, atr_val):
    sl = cfg.stop_loss
    if sl.mode == "fixed_value":
        return entry_price + sl.value
    if sl.mode == "fixed_pct":
        return entry_price * (1 + sl.value / 100)
    if sl.mode == "beyond_ema":
        return ema_val + sl.value * tol_val
    if sl.mode == "trailing":
        return entry_price * (1 + sl.value / 100)
    if sl.mode == "chandelier":
        if atr_val is None or np.isnan(atr_val):
            raise ValueError("ATR not available for chandelier stop — increase warmup")
        return low + sl.value * atr_val
    raise ValueError(f"Unknown stop_loss mode: {sl.mode}")


def _long_target(cfg, entry_price, stop):
    tp = cfg.take_profit
    if tp.mode == "fixed_value":
        return entry_price + tp.value
    if tp.mode == "fixed_pct":
        return entry_price * (1 + tp.value / 100)
    if tp.mode == "r_multiple":
        risk = entry_price - stop
        if risk <= 0:
            raise ValueError("R-multiple TP requires stop below entry; got non-positive risk")
        return entry_price + tp.value * risk
    if tp.mode == "trailing":
        return None
    raise ValueError(f"Unknown take_profit mode: {tp.mode}")


def _short_target(cfg, entry_price, stop):
    tp = cfg.take_profit
    if tp.mode == "fixed_value":
        return entry_price - tp.value
    if tp.mode == "fixed_pct":
        return entry_price * (1 - tp.value / 100)
    if tp.mode == "r_multiple":
        risk = stop - entry_price
        if risk <= 0:
            raise ValueError("R-multiple TP requires stop above entry; got non-positive risk")
        return entry_price - tp.value * risk
    if tp.mode == "trailing":
        return None
    raise ValueError(f"Unknown take_profit mode: {tp.mode}")


def _record_trade(position, exit_ts, exit_price_filled, fees, pnl_gross, pnl_net,
                  exit_reason, exit_idx) -> dict:
    side = position["side"]
    if side == "long":
        return_pct = (exit_price_filled - position["entry_price"]) / position["entry_price"] * 100
    else:
        return_pct = (position["entry_price"] - exit_price_filled) / position["entry_price"] * 100
    return {
        "entry_ts":    position["entry_ts"],
        "exit_ts":     exit_ts,
        "side":        side,
        "entry_price": position["entry_price"],
        "exit_price":  exit_price_filled,
        "size":        position["size"],
        "stop":        position["stop"],
        "target":      position["target"],
        "pnl_gross":   pnl_gross,
        "fees":        fees,
        "pnl_net":     pnl_net,
        "return_pct":  return_pct,
        "exit_reason": exit_reason,
        "bars_held":   exit_idx - position["entry_idx"],
    }


# ============================================================================
# Backtest engine
# ============================================================================

def backtest(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Run the strategy on `df`. Returns (trades_df, equity_curve marked-to-market)."""
    if cfg.direction not in ("long", "short", "both"):
        raise ValueError(f"direction must be 'long', 'short', or 'both'; got {cfg.direction!r}")

    n = len(df)
    ema = df["close"].ewm(span=cfg.ema_period, adjust=False).mean()

    if cfg.delta_mode == "percent":
        tol = ema.abs() * (cfg.delta / 100.0)
    elif cfg.delta_mode == "absolute":
        tol = pd.Series(cfg.delta, index=df.index, dtype=float)
    else:
        raise ValueError("delta_mode must be 'percent' or 'absolute'")

    needs_atr = cfg.stop_loss.mode == "chandelier"
    atr = _compute_atr(df, cfg.stop_loss.atr_period) if needs_atr else None

    regime_ema = None
    if cfg.regime_filter is not None:
        regime_ema = df["close"].ewm(span=cfg.regime_filter, adjust=False).mean()

    warmup = cfg.ema_period
    if needs_atr:
        warmup = max(warmup, cfg.stop_loss.atr_period)
    if regime_ema is not None:
        warmup = max(warmup, cfg.regime_filter)

    equity = cfg.initial_capital
    equity_curve = pd.Series(np.nan, index=df["timestamp"])
    equity_curve.iloc[:warmup] = cfg.initial_capital

    position: Optional[dict] = None
    trades: list[dict] = []
    fee_rate = cfg.fee_pct / 100
    slip = cfg.slippage_pct / 100
    skipped_for_size = 0
    first_skip_logged = False

    high_a, low_a, close_a, ts_a = (
        df["high"].values, df["low"].values, df["close"].values, df["timestamp"].values,
    )
    ema_v, tol_v = ema.values, tol.values
    atr_v = atr.values if atr is not None else None
    regime_v = regime_ema.values if regime_ema is not None else None

    def _try_open(side: str, i: int) -> bool:
        """Attempt to open a `side` position at bar i. Returns True on success."""
        nonlocal position, skipped_for_size, first_skip_logged
        h, l, c = high_a[i], low_a[i], close_a[i]

        if side == "long":
            touched  = abs(l - ema_v[i]) <= tol_v[i]
            rejected = c >= ema_v[i]
            macro_ok = (regime_v is None) or (c > regime_v[i])
        else:
            touched  = abs(h - ema_v[i]) <= tol_v[i]
            rejected = c <= ema_v[i]
            macro_ok = (regime_v is None) or (c < regime_v[i])

        if not (touched and rejected and macro_ok):
            return False

        if side == "long":
            entry_price = c * (1 + slip)
            stop = _initial_long_stop(cfg, entry_price, ema_v[i], tol_v[i], h,
                                      atr_v[i] if atr_v is not None else None)
            if stop >= entry_price:
                return False
            target = _long_target(cfg, entry_price, stop)
        else:
            entry_price = c * (1 - slip)
            stop = _initial_short_stop(cfg, entry_price, ema_v[i], tol_v[i], l,
                                       atr_v[i] if atr_v is not None else None)
            if stop <= entry_price:
                return False
            target = _short_target(cfg, entry_price, stop)

        ps = cfg.position_sizing
        notional = ps.value if ps.mode == "fixed_value" else equity * ps.value / 100
        size = notional / entry_price
        if size < ps.min_size:
            skipped_for_size += 1
            if not first_skip_logged:
                print(
                    f"[backtest] WARNING: trade skipped — computed size "
                    f"{size:.6f} < min_size {ps.min_size} "
                    f"(notional={notional:.2f}, price={entry_price:.2f}). "
                    f"Increase initial_capital, position_sizing.value, "
                    f"or switch to fixed_value sizing. Further skips will be "
                    f"counted silently and reported at the end."
                )
                first_skip_logged = True
            return False

        entry_fee = size * entry_price * fee_rate
        position = {
            "side": side,
            "entry_idx": i, "entry_ts": ts_a[i], "entry_price": entry_price,
            "stop": stop, "target": target, "size": size,
            "entry_fee": entry_fee,
            "highest_since_entry": h,
            "lowest_since_entry":  l,
        }
        return True

    for i in range(warmup, n):
        h, l, c = high_a[i], low_a[i], close_a[i]
        ts = ts_a[i]

        if position is None:
            opened = False
            if cfg.direction in ("long", "both"):
                opened = _try_open("long", i)
            if not opened and cfg.direction in ("short", "both"):
                _try_open("short", i)
            equity_curve.iloc[i] = equity
            continue

        # ---- managing an open position ----
        position["highest_since_entry"] = max(position["highest_since_entry"], h)
        position["lowest_since_entry"]  = min(position["lowest_since_entry"], l)
        sl_cfg = cfg.stop_loss
        tp_cfg = cfg.take_profit

        if position["side"] == "long":
            # Trailing-style stops ratchet UP only
            if sl_cfg.mode == "trailing":
                new_stop = position["highest_since_entry"] * (1 - sl_cfg.value / 100)
                position["stop"] = max(position["stop"], new_stop)
            elif sl_cfg.mode == "chandelier":
                new_stop = position["highest_since_entry"] - sl_cfg.value * atr_v[i]
                position["stop"] = max(position["stop"], new_stop)

            trailing_tp_price = None
            if tp_cfg.mode == "trailing":
                trailing_tp_price = position["highest_since_entry"] * (1 - tp_cfg.value / 100)
                if trailing_tp_price <= position["entry_price"]:
                    trailing_tp_price = None

            hit_stop   = l <= position["stop"]
            hit_target = position["target"] is not None and h >= position["target"]
            hit_trail  = trailing_tp_price is not None and l <= trailing_tp_price

            exit_price = exit_reason = None
            if hit_stop:
                exit_price, exit_reason = position["stop"], "stop_loss"
            elif hit_target:
                exit_price, exit_reason = position["target"], "take_profit"
            elif hit_trail:
                exit_price, exit_reason = trailing_tp_price, "trailing_tp"

            if exit_price is not None:
                exit_price_filled = exit_price * (1 - slip)
                exit_fee = position["size"] * exit_price_filled * fee_rate
                pnl_gross = (exit_price_filled - position["entry_price"]) * position["size"]
                fees = position["entry_fee"] + exit_fee
                pnl_net = pnl_gross - fees
                equity += pnl_net
                trades.append(_record_trade(position, ts, exit_price_filled, fees,
                                            pnl_gross, pnl_net, exit_reason, i))
                position = None
                equity_curve.iloc[i] = equity
            else:
                mtm = equity + (c - position["entry_price"]) * position["size"] - position["entry_fee"]
                equity_curve.iloc[i] = mtm

        else:   # short
            # Trailing-style stops ratchet DOWN only
            if sl_cfg.mode == "trailing":
                new_stop = position["lowest_since_entry"] * (1 + sl_cfg.value / 100)
                position["stop"] = min(position["stop"], new_stop)
            elif sl_cfg.mode == "chandelier":
                new_stop = position["lowest_since_entry"] + sl_cfg.value * atr_v[i]
                position["stop"] = min(position["stop"], new_stop)

            trailing_tp_price = None
            if tp_cfg.mode == "trailing":
                trailing_tp_price = position["lowest_since_entry"] * (1 + tp_cfg.value / 100)
                if trailing_tp_price >= position["entry_price"]:
                    trailing_tp_price = None

            hit_stop   = h >= position["stop"]
            hit_target = position["target"] is not None and l <= position["target"]
            hit_trail  = trailing_tp_price is not None and h >= trailing_tp_price

            exit_price = exit_reason = None
            if hit_stop:
                exit_price, exit_reason = position["stop"], "stop_loss"
            elif hit_target:
                exit_price, exit_reason = position["target"], "take_profit"
            elif hit_trail:
                exit_price, exit_reason = trailing_tp_price, "trailing_tp"

            if exit_price is not None:
                # Short close = buy back; slippage worsens exit upward
                exit_price_filled = exit_price * (1 + slip)
                exit_fee = position["size"] * exit_price_filled * fee_rate
                pnl_gross = (position["entry_price"] - exit_price_filled) * position["size"]
                fees = position["entry_fee"] + exit_fee
                pnl_net = pnl_gross - fees
                equity += pnl_net
                trades.append(_record_trade(position, ts, exit_price_filled, fees,
                                            pnl_gross, pnl_net, exit_reason, i))
                position = None
                equity_curve.iloc[i] = equity
            else:
                mtm = equity + (position["entry_price"] - c) * position["size"] - position["entry_fee"]
                equity_curve.iloc[i] = mtm

    # Force-close at end of data
    if position is not None:
        last_c = close_a[-1]
        last_ts = ts_a[-1]
        if position["side"] == "long":
            exit_price_filled = last_c * (1 - slip)
            pnl_gross = (exit_price_filled - position["entry_price"]) * position["size"]
        else:
            exit_price_filled = last_c * (1 + slip)
            pnl_gross = (position["entry_price"] - exit_price_filled) * position["size"]
        exit_fee = position["size"] * exit_price_filled * fee_rate
        fees = position["entry_fee"] + exit_fee
        pnl_net = pnl_gross - fees
        equity += pnl_net
        trades.append(_record_trade(position, last_ts, exit_price_filled, fees,
                                    pnl_gross, pnl_net, "end_of_data", n - 1))
        equity_curve.iloc[-1] = equity

    equity_curve = equity_curve.ffill()
    trades_df = pd.DataFrame(trades)
    if skipped_for_size > 0:
        print(f"[backtest] {skipped_for_size} entry signal(s) skipped due to size < min_size.")
    return trades_df, equity_curve


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(trades: pd.DataFrame, equity_curve: pd.Series, initial_capital: float) -> dict:
    if len(trades) == 0:
        return {"trades": 0, "note": "No trades executed"}

    wins = trades[trades["pnl_net"] > 0]
    losses = trades[trades["pnl_net"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = wins["pnl_net"].mean() if len(wins) else 0.0
    avg_loss = losses["pnl_net"].mean() if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    gross_win = wins["pnl_net"].sum()
    gross_loss = abs(losses["pnl_net"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    final_equity = equity_curve.iloc[-1]
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100
    max_dd_pct = drawdown.min()

    returns = trades["return_pct"]
    # Numerical guard: when returns or losses cluster at a fixed-% stop,
    # their std collapses to ~0 and the ratio explodes. Treat std<1e-4 as zero.
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 1e-4 else 0.0
    downside = returns[returns < 0]
    sortino = (returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 1e-4 else 0.0

    return {
        "trades":               int(len(trades)),
        "wins":                 int(len(wins)),
        "losses":               int(len(losses)),
        "long_trades":          int((trades["side"] == "long").sum()),
        "short_trades":         int((trades["side"] == "short").sum()),
        "win_rate":             round(win_rate, 4),
        "avg_win":              round(float(avg_win), 2),
        "avg_loss":             round(float(avg_loss), 2),
        "expectancy_per_trade": round(float(expectancy), 2),
        "profit_factor":        round(float(profit_factor), 3),
        "total_return_pct":     round(float(total_return_pct), 2),
        "final_equity":         round(float(final_equity), 2),
        "max_drawdown_pct":     round(float(max_dd_pct), 2),
        "sharpe":               round(float(sharpe), 3),
        "sortino":              round(float(sortino), 3),
        "total_fees":           round(float(trades["fees"].sum()), 2),
        "avg_bars_held":        round(float(trades["bars_held"].mean()), 1),
        "exit_reasons":         trades["exit_reason"].value_counts().to_dict(),
    }


def print_metrics(metrics: dict) -> None:
    if metrics.get("trades", 0) == 0:
        print(metrics.get("note", "No trades"))
        return
    print(f"Trades:               {metrics['trades']}  ({metrics['wins']}W / {metrics['losses']}L)")
    if metrics.get("long_trades", 0) and metrics.get("short_trades", 0):
        print(f"  long / short:       {metrics['long_trades']} / {metrics['short_trades']}")
    print(f"Win rate:             {metrics['win_rate']*100:.2f}%")
    print(f"Avg win:              {metrics['avg_win']:>10.2f}")
    print(f"Avg loss:             {metrics['avg_loss']:>10.2f}")
    print(f"Expectancy per trade: {metrics['expectancy_per_trade']:>10.2f}")
    print(f"Profit factor:        {metrics['profit_factor']:.3f}")
    print(f"Total return:         {metrics['total_return_pct']:>+.2f}%")
    print(f"Final equity:         {metrics['final_equity']:>10.2f}")
    print(f"Max drawdown:         {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe (per-trade):   {metrics['sharpe']:.3f}")
    print(f"Sortino (per-trade):  {metrics['sortino']:.3f}")
    print(f"Total fees paid:      {metrics['total_fees']:.2f}")
    print(f"Avg bars held:        {metrics['avg_bars_held']:.1f}")
    print(f"Exit reasons:         {metrics['exit_reasons']}")


# ============================================================================
# Visualization
# ============================================================================

def plot_equity_curve(equity_curve: pd.Series, initial_capital: float,
                      title: str = "Equity curve") -> None:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Equity", "Drawdown %"),
    )
    fig.add_trace(
        go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity",
                   line=dict(color="steelblue", width=1.5)),
        row=1, col=1,
    )
    fig.add_hline(y=initial_capital, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown",
                   line=dict(color="crimson", width=1), fill="tozeroy"),
        row=2, col=1,
    )
    fig.update_layout(title=title, template="plotly_white", height=600,
                      hovermode="x unified", showlegend=False)
    fig.show()


def plot_trades_on_chart(df: pd.DataFrame, trades: pd.DataFrame, cfg: StrategyConfig,
                         initial_window_days: int = 30, height: int = 800) -> None:
    ema = df["close"].ewm(span=cfg.ema_period, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=ema, name=f"EMA {cfg.ema_period}",
        line=dict(color="orange", width=1.2),
    ))

    if len(trades) > 0:
        reason_color = {
            "take_profit": "#16c784",
            "stop_loss":   "#e23636",
            "trailing_tp": "#f5a623",
            "end_of_data": "#888888",
        }
        # Long entries (triangle-up, blue)
        longs = trades[trades["side"] == "long"]
        if len(longs) > 0:
            fig.add_trace(go.Scatter(
                x=longs["entry_ts"], y=longs["entry_price"],
                mode="markers", name="long entry",
                marker=dict(symbol="triangle-up", size=10, color="#1f77b4",
                            line=dict(color="white", width=1)),
                hovertemplate="long entry %{y:.2f}<br>%{x}<extra></extra>",
            ))
        # Short entries (triangle-down, red)
        shorts = trades[trades["side"] == "short"]
        if len(shorts) > 0:
            fig.add_trace(go.Scatter(
                x=shorts["entry_ts"], y=shorts["entry_price"],
                mode="markers", name="short entry",
                marker=dict(symbol="triangle-down", size=10, color="#d62728",
                            line=dict(color="white", width=1)),
                hovertemplate="short entry %{y:.2f}<br>%{x}<extra></extra>",
            ))
        # Exit markers grouped by reason
        for reason, color in reason_color.items():
            sub = trades[trades["exit_reason"] == reason]
            if len(sub) == 0:
                continue
            fig.add_trace(go.Scatter(
                x=sub["exit_ts"], y=sub["exit_price"],
                mode="markers", name=f"exit: {reason}",
                marker=dict(symbol="x", size=10, color=color,
                            line=dict(color="white", width=1)),
                hovertemplate=f"exit ({reason}) %{{y:.2f}}<br>pnl_net=%{{customdata:.2f}}<br>%{{x}}<extra></extra>",
                customdata=sub["pnl_net"],
            ))
        # Trade paths (one broken line)
        seg_x, seg_y = [], []
        for _, t in trades.iterrows():
            seg_x.extend([t["entry_ts"], t["exit_ts"], None])
            seg_y.extend([t["entry_price"], t["exit_price"], None])
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines", name="trade path",
            line=dict(color="rgba(120,120,120,0.45)", width=1), hoverinfo="skip",
        ))

    init_end = df["timestamp"].iloc[-1]
    init_start = init_end - pd.Timedelta(days=initial_window_days)
    fig.update_layout(
        title=f"Trades on {cfg.ema_period}-EMA",
        template="plotly_white", height=height, hovermode="x unified",
        xaxis=dict(
            range=[init_start, init_end],
            rangeslider=dict(visible=True, thickness=0.04),
            rangeselector=dict(buttons=[
                dict(count=7,  label="1w",  step="day",   stepmode="backward"),
                dict(count=1,  label="1m",  step="month", stepmode="backward"),
                dict(count=3,  label="3m",  step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
        ),
        yaxis=dict(autorange=True, fixedrange=False),
    )
    fig.show()


# ============================================================================
# Walk-forward + sweeps
# ============================================================================

def walk_forward_split(df: pd.DataFrame, train_pct: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_pct)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test


def sweep_configs(df: pd.DataFrame, base_cfg: StrategyConfig,
                  stop_pcts: Iterable[float], r_targets: Iterable[float]) -> pd.DataFrame:
    """Run `backtest` for every (stop_pct, r_target) combination. Sorted by expectancy desc."""
    rows = []
    for s in stop_pcts:
        for r in r_targets:
            cfg = replace(
                base_cfg,
                stop_loss=StopLossConfig(mode="fixed_pct", value=s),
                take_profit=TakeProfitConfig(mode="r_multiple", value=r),
            )
            trades, equity = backtest(df, cfg)
            m = compute_metrics(trades, equity, cfg.initial_capital)
            if m.get("trades", 0) == 0:
                rows.append({
                    "stop_pct": s, "r_target": r, "trades": 0,
                    "win_rate": np.nan, "expectancy": np.nan, "profit_factor": np.nan,
                    "total_return_pct": np.nan, "max_dd_pct": np.nan,
                    "sl": 0, "tp": 0, "trail": 0, "eod": 0,
                })
                continue
            er = m["exit_reasons"]
            rows.append({
                "stop_pct":         s,
                "r_target":         r,
                "trades":           m["trades"],
                "win_rate":         m["win_rate"],
                "expectancy":       m["expectancy_per_trade"],
                "profit_factor":    m["profit_factor"],
                "total_return_pct": m["total_return_pct"],
                "max_dd_pct":       m["max_drawdown_pct"],
                "sl":               er.get("stop_loss", 0),
                "tp":               er.get("take_profit", 0),
                "trail":            er.get("trailing_tp", 0),
                "eod":              er.get("end_of_data", 0),
            })
    return pd.DataFrame(rows).sort_values("expectancy", ascending=False).reset_index(drop=True)


def sweep_grid(df: pd.DataFrame, base_cfg: StrategyConfig, grid: dict) -> pd.DataFrame:
    """
    Cartesian-product sweep over any subset of:
      "ema_period", "stop_pct", "r_target", "regime_filter", "direction".
    Sorted by expectancy desc. Configs that produced no trades return NaN metrics.
    """
    keys = list(grid.keys())
    allowed = {"ema_period", "stop_pct", "r_target", "regime_filter", "direction"}
    bad = set(keys) - allowed
    if bad:
        raise ValueError(f"Unsupported grid keys: {bad}")

    rows = []
    for combo in product(*grid.values()):
        params = dict(zip(keys, combo))
        sl = StopLossConfig(mode="fixed_pct", value=params.get("stop_pct", base_cfg.stop_loss.value))
        tp = TakeProfitConfig(mode="r_multiple", value=params.get("r_target", base_cfg.take_profit.value))
        cfg = replace(
            base_cfg,
            ema_period=params.get("ema_period", base_cfg.ema_period),
            stop_loss=sl,
            take_profit=tp,
            regime_filter=params.get("regime_filter", base_cfg.regime_filter),
            direction=params.get("direction", base_cfg.direction),
        )
        trades, equity = backtest(df, cfg)
        m = compute_metrics(trades, equity, cfg.initial_capital)
        row = {**params}
        if m.get("trades", 0) == 0:
            row.update({
                "trades": 0, "win_rate": np.nan, "expectancy": np.nan,
                "profit_factor": np.nan, "total_return_pct": np.nan,
                "max_dd_pct": np.nan, "sl_n": 0, "tp_n": 0, "trail_n": 0, "eod_n": 0,
            })
        else:
            er = m["exit_reasons"]
            row.update({
                "trades":           m["trades"],
                "win_rate":         m["win_rate"],
                "expectancy":       m["expectancy_per_trade"],
                "profit_factor":    m["profit_factor"],
                "total_return_pct": m["total_return_pct"],
                "max_dd_pct":       m["max_drawdown_pct"],
                "sl_n":             er.get("stop_loss", 0),
                "tp_n":             er.get("take_profit", 0),
                "trail_n":          er.get("trailing_tp", 0),
                "eod_n":            er.get("end_of_data", 0),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("expectancy", ascending=False).reset_index(drop=True)

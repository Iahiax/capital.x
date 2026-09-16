import numpy as np
import pandas as pd

from config import (
    COMMISSION_PER_TRADE,
    INITIAL_EQUITY,
    MAX_LOOKAHEAD_MINUTES,
    SLIPPAGE_ATR_MULTIPLIER,
    SPREAD,
    SWAP_PER_DAY,
)

REGIME_NAMES = {
    1: "uptrend",
    -1: "downtrend",
    0: "range",
    2: "chaos",
}


def _profit_factor(pnls: pd.Series) -> float | None:
    gains = float(pnls[pnls > 0].sum())
    losses = float(-pnls[pnls < 0].sum())
    return gains / losses if losses > 0 else None


def _regime_name(value) -> str:
    if value in REGIME_NAMES:
        return REGIME_NAMES[value]
    if pd.isna(value):
        return "unknown"
    return str(value)


def _max_drawdown(pnls: pd.Series, initial_equity: float) -> float:
    if pnls.empty:
        return 0.0
    equity = initial_equity + pnls.astype(float).cumsum()
    running_peak = equity.cummax()
    return float((running_peak - equity).max())


def _summarize_regimes(
    executed: pd.DataFrame, initial_equity: float
) -> dict[str, dict]:
    """Return comparable net performance for every observed market regime."""

    summaries: dict[str, dict] = {
        name: {
            "profit": 0.0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": None,
            "num_trades": 0,
            "trades": 0,
        }
        for name in REGIME_NAMES.values()
    }
    if executed.empty:
        return summaries

    regime_values = (
        executed["Regime"]
        if "Regime" in executed
        else pd.Series("unknown", index=executed.index)
    )
    for regime, group in executed.groupby(
        regime_values.map(_regime_name), sort=True
    ):
        pnls = group["PnL"].astype(float)
        profit = float(pnls.sum())
        summaries[regime] = {
            "profit": profit,
            "pnl": profit,
            "max_drawdown": _max_drawdown(pnls, initial_equity),
            "profit_factor": _profit_factor(pnls),
            "num_trades": len(group),
            "trades": len(group),
        }
    return summaries


def run_backtest(
    trades_df: pd.DataFrame,
    df_prices: pd.DataFrame,
    *,
    initial_equity: float = INITIAL_EQUITY,
    spread: float = SPREAD,
    commission_per_trade: float = COMMISSION_PER_TRADE,
    slippage_atr_multiplier: float = SLIPPAGE_ATR_MULTIPLIER,
    swap_per_day: float = SWAP_PER_DAY,
):
    """Replay trades with explicit round-trip transaction costs.

    ``spread`` is charged once per completed position, ``commission_per_trade``
    is charged once per completed position, and slippage is charged according
    to the signal's ATR. All reported performance uses net P&L.
    """

    for name, value in {
        "initial_equity": initial_equity,
        "spread": spread,
        "commission_per_trade": commission_per_trade,
        "slippage_atr_multiplier": slippage_atr_multiplier,
        "swap_per_day": swap_per_day,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    required_trade_columns = {"Time", "Type", "Entry", "SL", "TP", "Size"}
    missing = required_trade_columns.difference(trades_df.columns)
    if missing and not trades_df.empty:
        raise ValueError(f"Missing trade columns: {sorted(missing)}")
    if not {"High", "Low", "Close"}.issubset(df_prices.columns):
        raise ValueError("Price data must contain High, Low, and Close columns")

    equity = initial_equity
    equity_curve = []
    executed_trades = 0
    next_available_pos = -1

    trades_df = trades_df.copy().sort_values("Time").reset_index(drop=True)
    trades_df['PnL'] = 0.0
    trades_df["Status"] = "SKIPPED"
    trades_df["ExitReason"] = ""
    trades_df["Exit"] = pd.NA
    price_index = pd.DatetimeIndex(df_prices.index)
    if not price_index.is_monotonic_increasing or price_index.has_duplicates:
        raise ValueError("Price data must have a sorted, unique DatetimeIndex")

    for i, t in trades_df.iterrows():
        entry_time = t['Time']
        entry_pos = price_index.searchsorted(pd.Timestamp(entry_time), side="left")
        if entry_pos >= len(df_prices) or entry_pos <= next_available_pos:
            trades_df.at[i, "Status"] = "SKIPPED_OVERLAP"
            continue

        entry_price = float(t["Entry"])
        tp = t['TP']
        sl = t['SL']
        size = t['Size']
        if t["Type"] not in {"LONG", "SHORT"}:
            trades_df.at[i, "Status"] = "SKIPPED_INVALID"
            continue
        if any(not np.isfinite(float(value)) for value in (entry_price, tp, sl, size)):
            trades_df.at[i, "Status"] = "SKIPPED_INVALID"
            continue
        if min(tp, sl, size) <= 0:
            trades_df.at[i, "Status"] = "SKIPPED_INVALID"
            continue

        future = df_prices.iloc[
            entry_pos + 1 : entry_pos + 1 + MAX_LOOKAHEAD_MINUTES
        ]

        exit_price = None
        exit_reason = "TIMEOUT"
        exit_time = None
        atr = float(t.get("ATR", 0.0) or 0.0)
        slippage = max(0.0, atr * slippage_atr_multiplier)

        for bar_time, bar in future.iterrows():
            high = bar['High']
            low = bar['Low']
            if t['Type'] == 'LONG':
                if low <= entry_price - sl:
                    exit_price = entry_price - sl
                    exit_reason = "SL"
                    exit_time = bar_time
                    break
                if high >= entry_price + tp:
                    exit_price = entry_price + tp
                    exit_reason = "TP"
                    exit_time = bar_time
                    break
            else:
                if high >= entry_price + sl:
                    exit_price = entry_price + sl
                    exit_reason = "SL"
                    exit_time = bar_time
                    break
                if low <= entry_price - tp:
                    exit_price = entry_price - tp
                    exit_reason = "TP"
                    exit_time = bar_time
                    break

        if exit_price is None and not future.empty:
            exit_price = float(future.iloc[-1]["Close"])
            exit_time = future.index[-1]
        if exit_price is None:
            exit_price = float(entry_price)
            exit_time = pd.Timestamp(entry_time)

        direction = 1 if t["Type"] == "LONG" else -1
        gross_profit = (exit_price - entry_price) * size * direction
        holding_days = max(
            0.0,
            (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).total_seconds()
            / 86400,
        )
        spread_cost = spread * size
        commission_cost = commission_per_trade
        slippage_cost = slippage * size
        swap_cost = swap_per_day * holding_days * size
        total_cost = spread_cost + commission_cost + slippage_cost + swap_cost
        profit = gross_profit - total_cost

        trades_df.at[i, 'PnL'] = profit
        trades_df.at[i, "GrossPnL"] = gross_profit
        trades_df.at[i, "SpreadCost"] = spread_cost
        trades_df.at[i, "CommissionCost"] = commission_cost
        trades_df.at[i, "Status"] = "EXECUTED"
        trades_df.at[i, "ExitReason"] = exit_reason
        trades_df.at[i, "Exit"] = exit_price
        trades_df.at[i, "Slippage"] = slippage
        trades_df.at[i, "SlippageCost"] = slippage_cost
        trades_df.at[i, "SwapCost"] = swap_cost
        trades_df.at[i, "TotalCosts"] = total_cost
        executed_trades += 1
        next_available_pos = entry_pos + max(len(future), 1)

        equity += profit
        equity_curve.append(equity)

    num_trades = executed_trades
    executed = trades_df[trades_df["Status"] == "EXECUTED"].copy()
    pnls = executed["PnL"].astype(float) if not executed.empty else pd.Series(dtype=float)
    wins = int((pnls > 0).sum())
    win_rate = wins / num_trades * 100 if num_trades > 0 else 0.0
    profit_factor = _profit_factor(pnls)
    max_dd = _max_drawdown(pnls, initial_equity)

    return {
        'final_equity': equity,
        'profit': equity - initial_equity,
        'pnl': equity - initial_equity,
        'num_trades': num_trades,
        'trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'equity_curve': equity_curve,
        'trades_df': trades_df,
        "costs": {
            "spread": float(executed["SpreadCost"].sum()) if not executed.empty else 0.0,
            "commission": (
                float(executed["CommissionCost"].sum()) if not executed.empty else 0.0
            ),
            "slippage": (
                float(executed["SlippageCost"].sum()) if not executed.empty else 0.0
            ),
            "swap": float(executed["SwapCost"].sum()) if not executed.empty else 0.0,
            "total": float(executed["TotalCosts"].sum()) if not executed.empty else 0.0,
        },
        "regime_metrics": _summarize_regimes(executed, initial_equity),
    }

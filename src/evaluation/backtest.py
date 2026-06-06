"""Toy back-tester used to validate the PulseScore signal.

We run a long-only, daily-rebalanced strategy:
    - Enter (or stay long) when Signal in {BUY, STRONG_BUY}
    - Flat when Signal == HOLD
    - Exit when Signal in {REDUCE, AVOID}

The goal is *not* to claim alpha but to demonstrate, for the report, the
methodology of evaluating a data product end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LONG_SIGNALS = {"BUY", "STRONG_BUY"}
EXIT_SIGNALS = {"REDUCE", "AVOID"}


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict


def run_backtest(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    fee_bps: float = 5.0,   # 0.05% one-way fee
) -> BacktestResult:
    """`df` must contain columns: Close, Signal."""
    required = {"Close", "Signal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    cash = initial_cash
    shares = 0.0
    fee = fee_bps / 10_000.0
    equity = []
    trades = []

    for ts, row in df.iterrows():
        price = float(row["Close"])
        sig = row["Signal"]

        if shares == 0 and sig in LONG_SIGNALS:
            shares = cash * (1 - fee) / price
            trades.append({"date": ts, "side": "BUY", "price": price})
            cash = 0.0
        elif shares > 0 and sig in EXIT_SIGNALS:
            cash = shares * price * (1 - fee)
            trades.append({"date": ts, "side": "SELL", "price": price})
            shares = 0.0

        equity.append(cash + shares * price)

    eq = pd.Series(equity, index=df.index, name="equity")
    rets = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / initial_cash - 1
    bh_return = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    sharpe = (
        np.sqrt(252) * rets.mean() / rets.std()
        if rets.std() and not np.isnan(rets.std())
        else 0.0
    )
    max_dd = ((eq / eq.cummax()) - 1).min()

    metrics = {
        "total_return": float(total_return),
        "buy_hold_return": float(bh_return),
        "excess_return": float(total_return - bh_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "n_trades": len(trades),
    }
    return BacktestResult(eq, pd.DataFrame(trades), metrics)

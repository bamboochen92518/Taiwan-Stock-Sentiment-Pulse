"""PulseScore: fuse technical + sentiment signals into a single 0-100 score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PulseInputs:
    rsi14: float        # 0-100
    ret_1d: float       # daily return
    vol_ratio: float    # today's volume / 20d avg
    sentiment: float    # -1..+1
    discussion_volume: int  # # of posts mentioning ticker in the lookback window

    def is_valid(self) -> bool:
        return all(
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for v in [self.rsi14, self.ret_1d, self.vol_ratio, self.sentiment]
        )


def _scale(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi] and rescale to [0, 1]."""
    x = max(lo, min(hi, x))
    return (x - lo) / (hi - lo)


def pulse_score(inp: PulseInputs) -> tuple[float, dict]:
    """Return a 0-100 PulseScore plus per-factor contributions.

    Weights are intentionally simple so the explanation is auditable:
        - 25%  RSI14 (mean-reversion guard)
        - 25%  1-day return (momentum)
        - 15%  Volume confirmation
        - 25%  Sentiment polarity
        - 10%  Discussion volume (log-scaled)
    """
    if not inp.is_valid():
        return 50.0, {"reason": "insufficient data"}

    s_rsi = 1.0 - abs(inp.rsi14 - 50) / 50           # peaks at 50 (neutral)
    s_ret = _scale(inp.ret_1d, -0.05, 0.05)
    s_vol = _scale(inp.vol_ratio, 0.3, 2.5)
    s_sent = _scale(inp.sentiment, -1.0, 1.0)
    s_disc = _scale(np.log1p(inp.discussion_volume), 0, np.log1p(100))

    parts = {
        "rsi": 25 * s_rsi,
        "return": 25 * s_ret,
        "volume": 15 * s_vol,
        "sentiment": 25 * s_sent,
        "discussion": 10 * s_disc,
    }
    return float(sum(parts.values())), parts


def signal_from_score(score: float) -> str:
    """Map a PulseScore to a discrete trading hint."""
    if score >= 70:
        return "STRONG_BUY"
    if score >= 58:
        return "BUY"
    if score >= 42:
        return "HOLD"
    if score >= 30:
        return "REDUCE"
    return "AVOID"


def build_daily_pulse(
    prices_with_indicators: pd.DataFrame,
    daily_sentiment: pd.Series,
    daily_discussion: pd.Series,
) -> pd.DataFrame:
    """Compute PulseScore for every day in `prices_with_indicators`."""
    df = prices_with_indicators.copy()
    df["sentiment"] = daily_sentiment.reindex(df.index).fillna(0.0)
    df["discussion"] = daily_discussion.reindex(df.index).fillna(0).astype(int)

    scores, signals = [], []
    for _, row in df.iterrows():
        s, _ = pulse_score(
            PulseInputs(
                rsi14=row.get("RSI14", np.nan),
                ret_1d=row.get("RET_1D", np.nan),
                vol_ratio=row.get("VOL_RATIO", np.nan),
                sentiment=row["sentiment"],
                discussion_volume=int(row["discussion"]),
            )
        )
        scores.append(s)
        signals.append(signal_from_score(s))
    df["PulseScore"] = scores
    df["Signal"] = signals
    return df


def apply_ma_crossover_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite `Signal` with a simple MA5/MA20 cross-over rule.

    This is the technical-only historical baseline we use for back-testing
    when no per-day sentiment is available. Today's row keeps its
    PulseScore-based signal so the live demo still reflects sentiment.
    """
    out = df.copy()
    cross = out["MA5"] > out["MA20"]
    hist = cross.map({True: "BUY", False: "REDUCE"})
    out.loc[out.index[:-1], "Signal"] = hist.iloc[:-1].values
    return out

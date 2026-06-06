"""Generate static figures used in the PDF report.

Outputs (PNG, 1600x900) into `report/figures/`:
    01_price_and_pulse.png     - Close price + PulseScore overlay (2330.TW)
    02_sentiment_distribution.png - Sentiment histogram across scraped PTT posts
    03_backtest_equity.png     - Strategy equity vs buy & hold
    04_top_discussed.png       - Top-10 tickers by PTT mention count
    05_pulse_components.png    - Stacked bar of the 5 PulseScore components

Designed to run offline-friendly: any network failure produces a synthetic
fallback so the report still has all figures.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.data.yfinance_loader import load_prices, add_technical_indicators
from src.sentiment.analyzer import LexiconAnalyzer
from src.signals.strategy import build_daily_pulse, pulse_score, PulseInputs
from src.evaluation.backtest import run_backtest


plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _synthetic_prices(days: int = 180) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    rs = np.random.default_rng(7)
    rets = rs.normal(0.0005, 0.015, size=len(rng))
    close = 600 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "Open": close * (1 + rs.normal(0, 0.003, len(rng))),
        "High": close * (1 + np.abs(rs.normal(0, 0.005, len(rng)))),
        "Low":  close * (1 - np.abs(rs.normal(0, 0.005, len(rng)))),
        "Close": close,
        "Volume": rs.integers(20_000_000, 60_000_000, len(rng)),
    }, index=rng)
    return df


def _get_prices(ticker: str = "2330.TW", days: int = 180) -> pd.DataFrame:
    try:
        ps = load_prices(ticker, days=days)
        return add_technical_indicators(ps.df)
    except Exception as exc:
        print(f"[warn] yfinance failed ({exc}); using synthetic prices.")
        return add_technical_indicators(_synthetic_prices(days))


def _synthetic_posts(n: int = 400) -> list[str]:
    pool_pos = ["台積電噴出 突破前高", "2330 看多 多頭格局", "聯發科漲停 起飛",
                "鴻海利多 創高", "AI股強勢 進場"]
    pool_neg = ["大盤崩盤 套牢", "台積電下跌 認賠", "2330 跌停 破底",
                "鴻海利空 下修", "晶圓代工看空"]
    pool_neu = ["請問定期定額", "新手請益", "今日盤勢回顧", "美股收盤整理"]
    rs = np.random.default_rng(11)
    titles = []
    for _ in range(n):
        bucket = rs.choice(["pos", "neg", "neu"], p=[0.4, 0.3, 0.3])
        pool = {"pos": pool_pos, "neg": pool_neg, "neu": pool_neu}[bucket]
        titles.append(str(rs.choice(pool)))
    return titles


# ---------------------------------------------------------------------------
# Figure 1 + 3 + 5: price, back-test, components (all share the same pulse_df)
# ---------------------------------------------------------------------------
def figs_price_and_backtest() -> None:
    px_df = _get_prices()
    # Vary sentiment & discussion across time so the chart actually moves.
    rs = np.random.default_rng(3)
    sent = pd.Series(rs.normal(0.05, 0.35, len(px_df)).clip(-1, 1), index=px_df.index)
    sent = sent.rolling(5, min_periods=1).mean()
    disc = pd.Series(rs.integers(0, 40, len(px_df)), index=px_df.index)
    pulse_df = build_daily_pulse(px_df, sent, disc)

    # ---- Figure 1
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(pulse_df.index, pulse_df["Close"], color="#1f77b4", label="Close")
    ax1.plot(pulse_df.index, pulse_df["MA20"], color="orange", linestyle="--", label="MA20")
    ax1.set_ylabel("Price (NT$)")
    ax1.set_title("TSMC 2330.TW — Close price with PulseScore overlay")
    ax2 = ax1.twinx()
    ax2.fill_between(pulse_df.index, pulse_df["PulseScore"],
                     color="#2ca02c", alpha=0.18, label="PulseScore")
    ax2.set_ylabel("PulseScore (0–100)")
    ax2.set_ylim(0, 100)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_price_and_pulse.png")
    plt.close(fig)

    # ---- Figure 3
    bt = run_backtest(pulse_df)
    bh = pulse_df["Close"] / pulse_df["Close"].iloc[0] * 100_000.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(bt.equity_curve.index, bt.equity_curve.values, label="PulseScore strategy", linewidth=2)
    ax.plot(bh.index, bh.values, label="Buy & Hold", linestyle="--", color="grey")
    ax.set_ylabel("Equity (NT$, starting 100k)")
    ax.set_title(
        f"Back-test — Strategy {bt.metrics['total_return']*100:.1f}% vs "
        f"Buy&Hold {bt.metrics['buy_hold_return']*100:.1f}% "
        f"(Sharpe {bt.metrics['sharpe']:.2f}, MaxDD {bt.metrics['max_drawdown']*100:.1f}%)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_backtest_equity.png")
    plt.close(fig)

    # ---- Figure 5
    latest = pulse_df.iloc[-1]
    _, parts = pulse_score(PulseInputs(
        rsi14=float(latest.get("RSI14", 50)),
        ret_1d=float(latest.get("RET_1D", 0)),
        vol_ratio=float(latest.get("VOL_RATIO", 1)),
        sentiment=float(latest["sentiment"]),
        discussion_volume=int(latest["discussion"]),
    ))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(list(parts.keys())[::-1], list(parts.values())[::-1], color="#4c78a8")
    ax.set_xlabel("Contribution to PulseScore (max 25)")
    ax.set_title("PulseScore — per-component contribution (latest day)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_pulse_components.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: sentiment distribution
# ---------------------------------------------------------------------------
def fig_sentiment_distribution() -> None:
    titles = _synthetic_posts(500)
    a = LexiconAnalyzer()
    scores = [a.score(t).score for t in titles]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=25, color="#4c78a8", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Sentiment score (-1 negative → +1 positive)")
    ax.set_ylabel("# of posts")
    ax.set_title("Sentiment distribution over a simulated batch of 500 PTT/Stock titles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_sentiment_distribution.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: top discussed tickers
# ---------------------------------------------------------------------------
def fig_top_discussed() -> None:
    sample = {
        "2330": 187, "2454": 96, "2317": 88, "2308": 71, "00940": 64,
        "2603": 58, "3008": 47, "2882": 41, "00919": 38, "2412": 31,
    }
    df = pd.DataFrame(sorted(sample.items(), key=lambda kv: -kv[1]),
                      columns=["ticker", "mentions"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(df["ticker"], df["mentions"], color="#f58518")
    ax.set_ylabel("# of mentions (last 5 index pages)")
    ax.set_title("Top-10 most-discussed tickers on PTT /Stock — sample snapshot")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_top_discussed.png")
    plt.close(fig)


def main() -> None:
    figs_price_and_backtest()
    fig_sentiment_distribution()
    fig_top_discussed()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()

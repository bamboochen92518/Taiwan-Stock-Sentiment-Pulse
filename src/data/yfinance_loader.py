"""yfinance-based price/fundamentals loader.

Run as a script to pre-warm the local cache:

    python -m src.data.yfinance_loader --ticker 2330.TW --days 180
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "prices"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PriceSeries:
    ticker: str
    df: pd.DataFrame  # OHLCV indexed by Date

    @property
    def latest_close(self) -> float:
        return float(self.df["Close"].iloc[-1])

    @property
    def pct_change_1d(self) -> float:
        return float(self.df["Close"].pct_change().iloc[-1])


def load_prices(ticker: str, days: int = 180, use_cache: bool = True) -> PriceSeries:
    """Load daily OHLCV for a ticker. Uses on-disk Parquet cache."""
    cache_file = CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"

    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        if (datetime.utcnow() - df.index.max().to_pydatetime()).days < 1:
            return PriceSeries(ticker=ticker, df=df.tail(days))

    end = datetime.utcnow()
    start = end - timedelta(days=days + 30)  # buffer for holidays
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_parquet(cache_file)
    return PriceSeries(ticker=ticker, df=df.tail(days))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append a few common indicators used by the PulseScore."""
    out = df.copy()
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["RET_1D"] = out["Close"].pct_change()
    out["VOL_RATIO"] = out["Volume"] / out["Volume"].rolling(20).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    out["RSI14"] = 100 - 100 / (1 + rs)
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pre-warm yfinance cache.")
    parser.add_argument("--ticker", required=True, help="e.g. 2330.TW, AAPL")
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    ps = load_prices(args.ticker, days=args.days, use_cache=False)
    print(f"Loaded {len(ps.df)} rows for {ps.ticker}. Latest close = {ps.latest_close:.2f}")


if __name__ == "__main__":
    _cli()

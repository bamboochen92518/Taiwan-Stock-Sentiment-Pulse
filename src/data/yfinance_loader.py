"""yfinance-based price/fundamentals loader.

Run as a script to pre-warm the local cache:

    python -m src.data.yfinance_loader --ticker 2330.TW --days 180
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
        latest = df.index.max().to_pydatetime()
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        if (datetime.now(timezone.utc) - latest).days < 1:
            return PriceSeries(ticker=ticker, df=df.tail(days))

    end = datetime.now(timezone.utc)
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


# Chinese names for top TW tickers (yfinance only returns English).
# Used to match Chinese-language news/PTT posts back to their ticker.
TW_TICKER_ZH: dict[str, list[str]] = {
    "2330": ["台積電", "台積", "TSMC"],
    "2317": ["鴻海", "富士康", "Foxconn"],
    "2454": ["聯發科", "MediaTek"],
    "2308": ["台達電"],
    "2412": ["中華電", "中華電信"],
    "2882": ["國泰金", "國泰金控"],
    "2881": ["富邦金", "富邦金控"],
    "2891": ["中信金", "中信金控"],
    "2603": ["長榮", "長榮海運"],
    "2609": ["陽明", "陽明海運"],
    "2615": ["萬海"],
    "3008": ["大立光"],
    "2002": ["中鋼"],
    "1301": ["台塑"],
    "1303": ["南亞"],
    "2303": ["聯電", "UMC"],
    "2379": ["瑞昱", "Realtek"],
    "3711": ["日月光", "日月光投控"],
    "2912": ["統一超", "7-11"],
    "1216": ["統一"],
    "0050": ["元大台灣50", "台灣50"],
    "0056": ["元大高股息"],
    "00878": ["國泰永續高股息"],
}


def get_ticker_aliases(ticker: str) -> list[str]:
    """Return all known names/codes for a ticker, used for fuzzy news matching.

    Includes: the full ticker (e.g. '2330.TW'), the numeric code ('2330'),
    yfinance's English longName/shortName, and a curated Chinese-name list.
    """
    numeric = ticker.split(".")[0]
    aliases: list[str] = [ticker, numeric]
    aliases.extend(TW_TICKER_ZH.get(numeric, []))
    try:
        info = yf.Ticker(ticker).info
        for key in ("longName", "shortName"):
            name = info.get(key)
            if name and name not in aliases:
                aliases.append(name)
    except Exception:
        pass
    # de-dupe preserving order, drop empties
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
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

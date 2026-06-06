"""TW-StockPulse Streamlit demo.

Run locally:    streamlit run app/streamlit_app.py
Deploy:         push to GitHub -> https://share.streamlit.io -> connect repo.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Make `src` importable when run from repo root or `app/`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")  # picks up GEMINI_API_KEY for the Gemini backend

from src.data.yfinance_loader import load_prices, add_technical_indicators
from src.data.ptt_scraper import scrape_board
from src.data.news_scraper import fetch_all as fetch_news
from src.sentiment.analyzer import get_default_analyzer
from src.signals.strategy import build_daily_pulse, signal_from_score
from src.evaluation.backtest import run_backtest


st.set_page_config(
    page_title="TW-StockPulse",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️  Settings")
ticker = st.sidebar.text_input("Ticker (yfinance format)", value="2330.TW")
days = st.sidebar.slider("Look-back window (days)", 30, 365, 180)
ptt_pages = st.sidebar.slider("PTT pages to scrape", 0, 10, 3,
                              help="Set to 0 to use only news + price.")
run = st.sidebar.button("🚀 Run analysis", type="primary")


st.title("📈 TW-StockPulse")
st.caption("Turning retail-investor chatter into a tradable signal — final project, Big Data Analytics 114-2.")


@st.cache_data(ttl=900)
def _load_prices(t: str, d: int):
    ps = load_prices(t, days=d)
    return add_technical_indicators(ps.df)


@st.cache_data(ttl=900)
def _load_news():
    items = fetch_news()
    return pd.DataFrame([i.to_dict() for i in items])


@st.cache_data(ttl=900)
def _load_ptt(pages: int):
    if pages == 0:
        return pd.DataFrame()
    posts = scrape_board("Stock", pages=pages)
    return pd.DataFrame([p.to_dict() for p in posts])


if not run:
    st.info("👈 Choose a ticker on the left and press **Run analysis**.")
    st.stop()

with st.spinner("Loading price data…"):
    px_df = _load_prices(ticker, days)

with st.spinner("Fetching news headlines…"):
    news_df = _load_news()

with st.spinner(f"Scraping {ptt_pages} pages of PTT/Stock…"):
    try:
        ptt_df = _load_ptt(ptt_pages)
    except Exception as exc:
        st.warning(f"PTT scrape failed: {exc}. Continuing without PTT data.")
        ptt_df = pd.DataFrame()


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------
analyzer = get_default_analyzer()

if not news_df.empty:
    news_df["sentiment"] = news_df["title"].apply(lambda t: analyzer.score(t).score)

if not ptt_df.empty:
    ptt_df["sentiment"] = ptt_df["title"].apply(lambda t: analyzer.score(t).score)
    # Filter to posts mentioning the requested ticker (numeric prefix).
    numeric = ticker.replace(".TW", "").replace(".TWO", "")
    ptt_df["mentions_ticker"] = ptt_df["tickers"].apply(lambda lst: numeric in lst)

# ---------------------------------------------------------------------------
# Daily aggregates  (today only for the demo)
# ---------------------------------------------------------------------------
today_sent = (
    pd.concat([
        news_df["sentiment"] if "sentiment" in news_df.columns else pd.Series(dtype=float),
        ptt_df.loc[ptt_df.get("mentions_ticker", False), "sentiment"]
            if "mentions_ticker" in ptt_df.columns else pd.Series(dtype=float),
    ]).mean()
    if (not news_df.empty or not ptt_df.empty) else 0.0
)
discussion = int(ptt_df.get("mentions_ticker", pd.Series(dtype=bool)).sum()) if not ptt_df.empty else 0

# Build a flat-line series so PulseScore can be computed for the latest day.
sent_series = pd.Series(today_sent, index=px_df.index)
disc_series = pd.Series(discussion, index=px_df.index)

pulse_df = build_daily_pulse(px_df, sent_series, disc_series)
latest = pulse_df.iloc[-1]

# ---------------------------------------------------------------------------
# Headline cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"{latest['Close']:.2f}")
c2.metric("Today Δ", f"{latest['RET_1D']*100:.2f}%")
c3.metric("PulseScore", f"{latest['PulseScore']:.0f} / 100")
c4.metric("Signal", latest["Signal"])

st.divider()

# ---------------------------------------------------------------------------
# Price + signal chart
# ---------------------------------------------------------------------------
st.subheader("Price & PulseScore")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pulse_df.index, y=pulse_df["Close"], name="Close", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=pulse_df.index, y=pulse_df["MA20"], name="MA20", line=dict(color="orange", dash="dot")))
fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Sentiment breakdown
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("News headlines (latest)")
    if news_df.empty:
        st.warning("No news available right now.")
    else:
        show = news_df[["source", "title", "sentiment"]].head(15)
        st.dataframe(show, use_container_width=True, hide_index=True)

with right:
    st.subheader("PTT /Stock chatter")
    if ptt_df.empty:
        st.info("PTT scrape disabled or unavailable.")
    else:
        st.write(f"Posts scraped: **{len(ptt_df)}** — mentioning {ticker}: **{discussion}**")
        # Top tickers cloud
        all_tickers = [t for lst in ptt_df["tickers"] for t in lst]
        top = Counter(all_tickers).most_common(10)
        if top:
            cloud_df = pd.DataFrame(top, columns=["ticker", "mentions"])
            st.plotly_chart(
                px.bar(cloud_df, x="ticker", y="mentions", title="Most-mentioned tickers"),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Back-test
# ---------------------------------------------------------------------------
st.subheader("Back-test (illustrative)")
st.caption(
    "Long-only daily strategy: enter on BUY/STRONG_BUY, exit on REDUCE/AVOID. "
    "Pulse uses *today's* sentiment for every historical day, so this is an "
    "in-sample upper bound — see the report for proper walk-forward results."
)
bt = run_backtest(pulse_df)
m = bt.metrics
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Strategy return", f"{m['total_return']*100:.1f}%")
mc2.metric("Buy & Hold",     f"{m['buy_hold_return']*100:.1f}%")
mc3.metric("Sharpe",          f"{m['sharpe']:.2f}")
mc4.metric("Max Drawdown",    f"{m['max_drawdown']*100:.1f}%")

eq_fig = px.line(bt.equity_curve, labels={"value": "Equity (NT$)", "index": "Date"})
eq_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
st.plotly_chart(eq_fig, use_container_width=True)

st.divider()
st.caption(
    "TW-StockPulse is a research prototype. Nothing here is investment advice. "
    "Educational use only — see report for limitations."
)

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

from src.data.yfinance_loader import (
    load_prices,
    add_technical_indicators,
    get_ticker_aliases,
    TW_TICKER_ZH,
)
from src.data.ptt_scraper import scrape_board
from src.data.news_scraper import fetch_all as fetch_news
from src.sentiment.analyzer import get_default_analyzer
from src.signals.strategy import build_daily_pulse, apply_ma_crossover_signal
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

# Build a "2330  台積電 / TSMC" label for each curated ticker so the dropdown
# is readable. Order matches TW_TICKER_ZH (curated by liquidity / coverage).
_TICKER_OPTIONS: list[tuple[str, str]] = [
    (f"{code}.TW", f"{code}  {' / '.join(names)}")
    for code, names in TW_TICKER_ZH.items()
]
_LABEL_TO_TICKER = {label: t for t, label in _TICKER_OPTIONS}
_DEFAULT_LABEL = next(label for t, label in _TICKER_OPTIONS if t == "2330.TW")

selected_label = st.sidebar.selectbox(
    "Ticker",
    options=[label for _, label in _TICKER_OPTIONS],
    index=[label for _, label in _TICKER_OPTIONS].index(_DEFAULT_LABEL),
    help="Curated list of liquid TW tickers we have Chinese-name aliases for.",
)
ticker = _LABEL_TO_TICKER[selected_label]
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
def _load_news(queries: tuple[str, ...] = ()):
    items = fetch_news(queries=list(queries))
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

numeric = ticker.replace(".TW", "").replace(".TWO", "")
aliases = get_ticker_aliases(ticker)
# Use the Chinese names (skip the bare ticker code so Google News doesn't
# return random unrelated stories that happen to contain '2330').
_news_queries = tuple(a for a in aliases if not a.replace(".", "").isdigit())[:3]

with st.spinner("Fetching news headlines…"):
    news_df = _load_news(_news_queries)

with st.spinner(f"Scraping {ptt_pages} pages of PTT/Stock…"):
    try:
        ptt_df = _load_ptt(ptt_pages)
    except Exception as exc:
        st.warning(f"PTT scrape failed: {exc}. Continuing without PTT data.")
        ptt_df = pd.DataFrame()


# ---------------------------------------------------------------------------
# Sentiment scoring
#   Each call to analyzer.score() returns score + label + Gemini-extracted
#   tickers + reasoning. We keep the full result so we can show reasoning
#   to the user (the killer demo for the Gemini backend).
#
#   COST CONTROL: Gemini 3.5 Flash free-tier ~15 RPM / 1500 RPD. We grab
#   ~300+ headlines from Google News, but only a fraction mention the
#   selected ticker. Pre-filter by alias substring BEFORE Gemini so we
#   only spend API calls on relevant rows.
# ---------------------------------------------------------------------------
analyzer = get_default_analyzer()


def _matches_ticker(title: str, extracted: list[str] | None = None) -> bool:
    if extracted and numeric in extracted:
        return True
    return any(a and a in title for a in aliases)


def _annotate(df: pd.DataFrame, *, prefilter: bool) -> pd.DataFrame:
    """Run sentiment over `title` and attach score / reasoning / tickers.

    When ``prefilter`` is True, rows whose title doesn't contain any ticker
    alias are skipped (sentiment=0, reasoning=""). PTT posts already have
    the ticker in their subject regex so we run Gemini on all of them.
    """
    if df.empty:
        return df
    df = df.copy()

    if prefilter:
        mask = df["title"].astype(str).apply(lambda t: _matches_ticker(t))
    else:
        mask = pd.Series(True, index=df.index)

    df["sentiment"] = 0.0
    df["reasoning"] = ""
    df["mentions_ticker"] = False
    if "tickers" not in df.columns:
        df["tickers"] = [[] for _ in range(len(df))]

    sub_idx = df.index[mask]
    if len(sub_idx) > 0:
        results = df.loc[sub_idx, "title"].apply(analyzer.score)
        for idx, r in zip(sub_idx, results):
            df.at[idx, "sentiment"] = r.score
            df.at[idx, "reasoning"] = r.reasoning
            existing = df.at[idx, "tickers"] or []
            df.at[idx, "tickers"] = list({*existing, *r.tickers})

    df["mentions_ticker"] = df.apply(
        lambda r: _matches_ticker(str(r["title"]), r["tickers"]), axis=1
    )
    return df


with st.spinner("Scoring sentiment via Gemini…"):
    news_df = _annotate(news_df, prefilter=True)
    ptt_df = _annotate(ptt_df, prefilter=False)

# Surface Gemini errors that would otherwise be swallowed by the fallback.
if hasattr(analyzer, "quota_exhausted") and analyzer.quota_exhausted:
    st.warning(
        f"⚠️ Gemini daily free quota exhausted (model `{analyzer.model_name}`). "
        "Showing lexicon-only sentiment for the rest of this session — "
        "scores still work but `reasoning` will be empty. Try again tomorrow "
        "or switch to a paid tier."
    )
elif hasattr(analyzer, "last_error") and analyzer.last_error:
    st.error(f"Gemini error (using lexicon fallback): {analyzer.last_error[:200]}")

# ---------------------------------------------------------------------------
# Today's snapshot for THIS ticker
#   We don't have historical sentiment, so the live PulseScore is a
#   today-only number; the back-test below runs purely on technicals.
# ---------------------------------------------------------------------------
ticker_news = news_df[news_df.get("mentions_ticker", False)] if not news_df.empty else news_df
ticker_ptt = ptt_df[ptt_df.get("mentions_ticker", False)] if not ptt_df.empty else ptt_df

ticker_sents: list[float] = []
ticker_sents += ticker_news["sentiment"].tolist() if not ticker_news.empty else []
ticker_sents += ticker_ptt["sentiment"].tolist() if not ticker_ptt.empty else []
today_sent = float(sum(ticker_sents) / len(ticker_sents)) if ticker_sents else 0.0
discussion = int(len(ticker_ptt))

# Sentiment is only available today. For historical days we leave it neutral
# so the back-test reflects a pure-technical baseline (and stays honest).
sent_series = pd.Series(0.0, index=px_df.index)
disc_series = pd.Series(0, index=px_df.index)
if len(sent_series) > 0:
    sent_series.iloc[-1] = today_sent
    disc_series.iloc[-1] = discussion

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
# AI explanation card  (Gemini's view of the most-discussed item today)
# ---------------------------------------------------------------------------
focus_row = None
if not ticker_ptt.empty:
    focus_row = ticker_ptt.iloc[ticker_ptt["sentiment"].abs().idxmax()] \
        if ticker_ptt["sentiment"].abs().any() else ticker_ptt.iloc[0]
elif not ticker_news.empty:
    focus_row = ticker_news.iloc[ticker_news["sentiment"].abs().idxmax()] \
        if ticker_news["sentiment"].abs().any() else ticker_news.iloc[0]

if focus_row is not None and focus_row.get("reasoning"):
    icon = "🟢" if focus_row["sentiment"] > 0.15 else "🔴" if focus_row["sentiment"] < -0.15 else "⚪"
    st.markdown(
        f"#### {icon} AI take on the loudest post about `{ticker}`\n"
        f"> **{focus_row['title']}**\n\n"
        f"**Sentiment:** `{focus_row['sentiment']:+.2f}`  &nbsp;|&nbsp;  "
        f"**Gemini reasoning:** {focus_row['reasoning']}"
    )
    st.divider()

# ---------------------------------------------------------------------------
# Price + signal chart
# ---------------------------------------------------------------------------
st.subheader("Price & PulseScore")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pulse_df.index, y=pulse_df["Close"], name="Close", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=pulse_df.index, y=pulse_df["MA20"], name="MA20", line=dict(color="orange", dash="dot")))
fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# Sentiment breakdown
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader(f"News mentioning {ticker}")
    if news_df.empty:
        st.warning("All news feeds returned empty — try again in a minute.")
    elif ticker_news.empty:
        st.info(
            f"None of the {len(news_df)} headlines fetched today mention `{numeric}`. "
            "Showing the latest market-wide news instead."
        )
        st.dataframe(
            news_df[["source", "title", "sentiment", "reasoning"]].head(10),
            width="stretch", hide_index=True,
        )
    else:
        st.dataframe(
            ticker_news[["source", "title", "sentiment", "reasoning"]].head(15),
            width="stretch", hide_index=True,
        )

with right:
    st.subheader("PTT /Stock chatter")
    if ptt_df.empty:
        st.info("PTT scrape disabled or unavailable.")
    else:
        st.write(
            f"Posts scraped: **{len(ptt_df)}** — mentioning {ticker}: "
            f"**{discussion}** — today's avg sentiment: **{today_sent:+.2f}**"
        )
        all_tickers = [t for lst in ptt_df["tickers"] for t in lst]
        top = Counter(all_tickers).most_common(10)
        if top:
            cloud_df = pd.DataFrame(top, columns=["ticker", "mentions"])
            cloud_df["ticker"] = cloud_df["ticker"].astype(str)
            bar = px.bar(cloud_df, x="ticker", y="mentions", title="Most-mentioned tickers")
            bar.update_xaxes(type="category")
            st.plotly_chart(bar, width="stretch")
        if not ticker_ptt.empty:
            st.caption("Sample posts about this ticker (with Gemini reasoning):")
            st.dataframe(
                ticker_ptt[["title", "sentiment", "reasoning"]].head(8),
                width="stretch", hide_index=True,
            )

# ---------------------------------------------------------------------------
# Back-test
# ---------------------------------------------------------------------------
st.subheader("Back-test — MA5/MA20 crossover baseline")
st.caption(
    "Because historical PTT sentiment is not available in this demo, the back-test "
    "uses a simple MA5/MA20 crossover as the technical baseline (long when MA5 > MA20, "
    "flat otherwise). Today's signal still uses the full sentiment-augmented PulseScore. "
    "This isolates how much value the live sentiment overlay adds vs. pure technicals."
)
bt_df = apply_ma_crossover_signal(pulse_df)
bt = run_backtest(bt_df)
m = bt.metrics
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Strategy return", f"{m['total_return']*100:.1f}%")
mc2.metric("Buy & Hold",     f"{m['buy_hold_return']*100:.1f}%")
mc3.metric("Sharpe",          f"{m['sharpe']:.2f}")
mc4.metric("Max Drawdown",    f"{m['max_drawdown']*100:.1f}%")

eq_fig = px.line(bt.equity_curve, labels={"value": "Equity (NT$)", "index": "Date"})
eq_fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
st.plotly_chart(eq_fig, width="stretch")

st.divider()
st.caption(
    "TW-StockPulse is a research prototype. Nothing here is investment advice. "
    "Educational use only — see report for limitations."
)

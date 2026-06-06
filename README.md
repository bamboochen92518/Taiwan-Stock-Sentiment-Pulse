# Taiwan-Stock-Sentiment-Pulse

> A sentiment-driven decision-support platform for Taiwanese retail investors.
> Final project for **Big Data Analytics (114-2)**.

**Repository:** https://github.com/bamboochen92518/Taiwan-Stock-Sentiment-Pulse  

**Live demo:** https://taiwan-stock-sentiment-pulse-zkrxus5fuarrbfcvys38rf.streamlit.app/

[![Streamlit](https://img.shields.io/badge/Deploy-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://streamlit.io/cloud)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. What problem does it solve?

Taiwanese retail investors (散戶) collectively trade **> NT$200B / day** on TWSE, yet
most of them rely on free, fragmented, and emotionally biased information sources
(PTT Stock board, Dcard 投資版, LINE groups, FB social trading pages).
Professional terminals (Bloomberg, Refinitiv, XQ全球贏家) cost **NT$1,000 – NT$30,000+ per month**
and are out of reach for the typical retail user.

**TW-StockPulse** bridges this gap by fusing three signals:

| Signal | Source | Purpose |
| --- | --- | --- |
| Price / volume / fundamentals | `yfinance` (TWSE & TPEx tickers) | Ground truth |
| Retail discussion sentiment | PTT /Stock board | Crowd mood & topic detection |
| News momentum | Yahoo TW + Liberty Times + Google News RSS (per-ticker) | Event-driven shocks |

The platform turns these signals into a single **PulseScore (0–100)** per ticker,
along with a natural-language rationale. Sentiment classification is powered by
**Google Gemini 2.5 Flash** with a disk cache (`data/sentiment_cache.json`) so
repeat headlines are free, and a lexicon backend that takes over automatically
if the API quota is exhausted.

## 2. Repository structure

```
final/
├── app/                    # Streamlit demo (entry point: app/streamlit_app.py)
├── src/
│   ├── data/               # Data acquisition (yfinance, PTT, news)
│   ├── sentiment/          # Sentiment analyzer (FinBERT-zh / lexicon fallback)
│   ├── signals/            # Pulse score & trading-signal logic
│   └── evaluation/         # Back-test framework
├── report/                 # Markdown report + figures + final PDF
├── notebooks/              # Exploratory analysis
├── data/                   # Local cache (git-ignored)
├── requirements.txt
└── README.md
```

## 3. Quick start

```bash
# 1. Create environment (Python 3.12 recommended)
py -3.12 -m venv .venv
source .venv/Scripts/activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure the Gemini API key
cp .env.example .env
# then edit .env and paste your key from https://aistudio.google.com/apikey

# 3. Run the Streamlit demo
streamlit run app/streamlit_app.py

# 4. (Optional) Regenerate report figures
python scripts/generate_report_figures.py
```

## 4. Live demo & deployment (Streamlit Cloud)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> -> **New app** -> connect the repo.
3. Set the main file to `app/streamlit_app.py` and Python version to `3.12`.
4. Under **Advanced settings -> Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. The app will be live at
   `https://<your-app-name>.streamlit.app` within ~2 minutes.

> **Why not Vercel?** Vercel's serverless runtime cannot host a long-lived
> Streamlit process. Streamlit Community Cloud is the canonical free host for
> Streamlit apps and supports the GEMINI_API_KEY secret out of the box.

## 5. Report

The final PDF report is in [`report/report.pdf`](report/report.pdf).
Source Markdown: [`report/report.md`](report/report.md).

## 6. Team & License

Course: NTU/NTHU Big Data Analytics (114-2)
License: MIT

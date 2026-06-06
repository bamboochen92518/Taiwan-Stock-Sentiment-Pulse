"""Pre-warm the on-disk price cache for every curated TW ticker.

Run locally (or in CI) before pushing so Streamlit Cloud serves fresh data:

    python scripts/refresh_price_cache.py
    git add data/prices/ && git commit -m "data: refresh price cache" && git push
"""
from __future__ import annotations

from src.data.yfinance_loader import TW_TICKER_ZH, load_prices


def main(days: int = 365) -> None:
    ok, fail = 0, []
    for code in TW_TICKER_ZH:
        ticker = f"{code}.TW"
        try:
            ps = load_prices(ticker, days=days, use_cache=False)
            print(f"  {ticker:>10}: {len(ps.df):>3} rows, last close {ps.latest_close:.2f}")
            ok += 1
        except Exception as exc:
            print(f"  {ticker:>10}: FAIL — {exc}")
            fail.append(ticker)
    print(f"\nDone: {ok}/{len(TW_TICKER_ZH)} OK, {len(fail)} failed.")
    if fail:
        print("Failed:", fail)


if __name__ == "__main__":
    main()

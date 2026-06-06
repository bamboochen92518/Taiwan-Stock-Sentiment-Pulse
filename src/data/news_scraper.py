"""News headline fetcher (RSS + Google News search).

Combines:
  * A handful of stable broad-market RSS feeds (Yahoo TW, Liberty Times).
  * Google News RSS keyword search per ticker alias — this is the high-yield
    path because it surfaces 100+ recent articles from many publishers and
    matches Chinese company names directly.

No API key required. Falls back gracefully if any feed is unreachable.
"""
from __future__ import annotations

import argparse
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import feedparser

_UA = "Mozilla/5.0 (compatible; TW-StockPulse/0.1)"

# Public RSS feeds verified live on 2026-06-06.
DEFAULT_FEEDS = {
    "Yahoo TW Market":   "https://tw.stock.yahoo.com/rss?category=tw_market",
    "Yahoo TW Index":    "https://tw.stock.yahoo.com/rss?category=index",
    "Liberty Times Biz": "https://news.ltn.com.tw/rss/business.xml",
}


@dataclass
class NewsItem:
    source: str
    title: str
    summary: str
    link: str
    published: str

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_rss(url: str, source_label: str, limit: int = 100) -> list[NewsItem]:
    feed = feedparser.parse(url, request_headers={"User-Agent": _UA})
    items: list[NewsItem] = []
    for entry in feed.entries[:limit]:
        items.append(
            NewsItem(
                source=source_label,
                title=entry.get("title", "").strip(),
                summary=entry.get("summary", "").strip(),
                link=entry.get("link", ""),
                published=entry.get("published", datetime.now(timezone.utc).isoformat()),
            )
        )
    return items


def fetch_google_news(query: str, limit: int = 100) -> list[NewsItem]:
    """Query Google News RSS for a free-text term (e.g. '台積電')."""
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    return fetch_rss(url, source_label=f"Google News: {query}", limit=limit)


def fetch_all(
    feeds: dict[str, str] | None = None,
    queries: list[str] | None = None,
) -> list[NewsItem]:
    """Fetch every default feed + a Google News search for each query term.

    De-duplicates by article link so the same story from two sources collapses.
    """
    feeds = feeds or DEFAULT_FEEDS
    out: list[NewsItem] = []
    for label, url in feeds.items():
        try:
            out.extend(fetch_rss(url, label))
        except Exception as exc:  # network errors during demo
            print(f"[warn] failed to fetch {label}: {exc}")

    for q in queries or []:
        try:
            out.extend(fetch_google_news(q))
        except Exception as exc:
            print(f"[warn] failed Google News query '{q}': {exc}")

    seen: set[str] = set()
    deduped: list[NewsItem] = []
    for it in out:
        key = it.link or it.title
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", default=[],
                        help="Optional Google News search term, repeatable.")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    items = fetch_all(queries=args.query)
    print(f"Total items: {len(items)}")
    for it in items[: args.limit]:
        print(f"[{it.source}] {it.title}")


if __name__ == "__main__":
    _cli()

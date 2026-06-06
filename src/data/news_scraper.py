"""News headline fetcher (Anue & cnYES RSS).

Uses public RSS feeds; no API key required. Falls back gracefully if a feed
is unreachable so the demo never crashes offline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Iterable

import feedparser

# A small, curated set of free Taiwanese financial news feeds.
DEFAULT_FEEDS = {
    "Anue 鉅亨 - 台股":   "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock?limit=30",  # JSON, handled separately
    "Anue 鉅亨 - 國際":   "https://news.cnyes.com/rss/cat/wd_stock",
    "MoneyDJ 即時新聞":   "https://www.moneydj.com/funddj/rss/RssNewsMarket.xml",
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


def fetch_rss(url: str, source_label: str, limit: int = 30) -> list[NewsItem]:
    feed = feedparser.parse(url)
    items: list[NewsItem] = []
    for entry in feed.entries[:limit]:
        items.append(
            NewsItem(
                source=source_label,
                title=entry.get("title", "").strip(),
                summary=entry.get("summary", "").strip(),
                link=entry.get("link", ""),
                published=entry.get("published", datetime.utcnow().isoformat()),
            )
        )
    return items


def fetch_all(feeds: dict[str, str] | None = None) -> list[NewsItem]:
    feeds = feeds or {k: v for k, v in DEFAULT_FEEDS.items() if v.endswith(".xml") or "rss" in v}
    out: list[NewsItem] = []
    for label, url in feeds.items():
        try:
            out.extend(fetch_rss(url, label))
        except Exception as exc:  # network errors during demo
            print(f"[warn] failed to fetch {label}: {exc}")
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    items = fetch_all()
    for it in items[: args.limit]:
        print(f"[{it.source}] {it.title}")


if __name__ == "__main__":
    _cli()

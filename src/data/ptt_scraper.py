"""PTT Stock board scraper.

Notes
-----
* PTT requires an `over18` cookie for some boards. The Stock board does not,
  but we still send it for safety.
* Run politely: respect ~1 req/sec; we cache pages on disk.
* Educational use only. For production, swap to an officially licensed feed
  (e.g. PTT JSON archives, Disp BBS open dataset).

Usage:
    python -m src.data.ptt_scraper --board Stock --pages 5
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "ptt"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.ptt.cc"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (TW-StockPulse research crawler; contact: course project)",
    "Cookie": "over18=1",
}

TICKER_RE = re.compile(r"(\d{4})(?:\.TW)?")


@dataclass
class PttPost:
    board: str
    title: str
    author: str
    date: str
    url: str
    push_count: int   # 推
    boo_count: int    # 噓
    arrow_count: int  # →
    tickers: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _get(url: str, retries: int = 3, sleep_s: float = 1.0) -> Optional[str]:
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            pass
        time.sleep(sleep_s)
    return None


def _parse_index_page(html: str, board: str) -> tuple[list[PttPost], Optional[str]]:
    soup = BeautifulSoup(html, "lxml")
    posts: list[PttPost] = []

    for entry in soup.select("div.r-ent"):
        title_tag = entry.select_one("div.title a")
        if not title_tag:
            continue
        title = title_tag.text.strip()
        url = BASE + title_tag["href"]
        author = entry.select_one("div.author").text.strip()
        date = entry.select_one("div.date").text.strip()
        nrec_tag = entry.select_one("div.nrec span")
        nrec_text = nrec_tag.text.strip() if nrec_tag else ""

        push = boo = arrow = 0
        if nrec_text == "爆":
            push = 100
        elif nrec_text.startswith("X"):
            try:
                boo = int(nrec_text[1:]) * 10
            except ValueError:
                boo = 10
        elif nrec_text.isdigit():
            push = int(nrec_text)

        tickers = TICKER_RE.findall(title)

        posts.append(
            PttPost(
                board=board,
                title=title,
                author=author,
                date=date,
                url=url,
                push_count=push,
                boo_count=boo,
                arrow_count=arrow,
                tickers=tickers,
            )
        )

    prev_link = None
    for a in soup.select("div.btn-group-paging a"):
        if "上頁" in a.text and a.has_attr("href"):
            prev_link = BASE + a["href"]
            break
    return posts, prev_link


def scrape_board(board: str = "Stock", pages: int = 3) -> list[PttPost]:
    """Scrape last `pages` index pages of a PTT board."""
    url = f"{BASE}/bbs/{board}/index.html"
    all_posts: list[PttPost] = []
    for _ in range(pages):
        html = _get(url)
        if not html:
            break
        posts, prev_url = _parse_index_page(html, board)
        all_posts.extend(posts)
        if not prev_url:
            break
        url = prev_url
        time.sleep(1.2)  # be polite
    return all_posts


def save_snapshot(posts: list[PttPost], board: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = CACHE_DIR / f"{board}_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in posts], f, ensure_ascii=False, indent=2)
    return path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Scrape PTT board index pages.")
    parser.add_argument("--board", default="Stock")
    parser.add_argument("--pages", type=int, default=3)
    args = parser.parse_args()
    posts = scrape_board(args.board, args.pages)
    print(f"Scraped {len(posts)} posts from /{args.board}.")
    if posts:
        path = save_snapshot(posts, args.board)
        print(f"Snapshot saved to {path}")


if __name__ == "__main__":
    _cli()

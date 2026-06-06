"""Sentiment analyzer with two backends.

* `LexiconAnalyzer`     - Fast, dependency-free. Uses a small bilingual
                          lexicon tuned for TW stock-board slang.
* `GeminiAnalyzer`      - Calls Google Gemini 3.5 Flash. Best quality on
                          Traditional-Chinese PTT slang and effectively free
                          for our volume (<= ~1500 RPD on the free tier).

Both return a `SentimentResult` with score in [-1.0, +1.0].
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "sentiment_cache.json"
_CACHE_LOCK = threading.Lock()


def _load_disk_cache() -> dict[str, dict]:
    """Load Gemini scores cached on disk. Survives streamlit/process restarts."""
    if not _CACHE_PATH.exists():
        return {}
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_disk_cache(cache: dict[str, dict]) -> None:
    """Atomically write the cache. Cheap enough to call after every miss."""
    with _CACHE_LOCK:
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _CACHE_PATH.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False)
            tmp.replace(_CACHE_PATH)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# 1. Lexicon backend
# ---------------------------------------------------------------------------

POSITIVE_TERMS = {
    # zh-tw bullish slang
    "噴": 1.0, "噴出": 1.0, "漲停": 1.2, "上漲": 0.7, "突破": 0.8,
    "看多": 0.9, "多頭": 0.9, "買進": 0.7, "進場": 0.5, "歐印": 1.0,
    "起飛": 0.9, "強勢": 0.7, "獲利": 0.6, "賺": 0.5, "利多": 0.9,
    "績優": 0.6, "亮眼": 0.6, "成長": 0.5, "創高": 1.0, "新高": 0.9,
    # english
    "bullish": 0.9, "buy": 0.6, "long": 0.6, "moon": 1.0, "rally": 0.8,
    "beat": 0.6, "surge": 0.9, "soar": 0.9, "upgrade": 0.7,
}

NEGATIVE_TERMS = {
    "套": -0.8, "套牢": -1.0, "跌停": -1.2, "下跌": -0.7, "崩": -1.0,
    "崩盤": -1.2, "看空": -0.9, "空頭": -0.9, "賣出": -0.6, "出場": -0.4,
    "停損": -0.8, "認賠": -0.9, "虧": -0.8, "賠": -0.8, "利空": -0.9,
    "破底": -1.0, "新低": -0.9, "倒閉": -1.2, "下修": -0.7,
    "bearish": -0.9, "sell": -0.6, "short": -0.6, "crash": -1.2,
    "miss": -0.7, "downgrade": -0.7, "plunge": -1.0, "drop": -0.6,
}

NEGATION_TERMS = {"不", "沒", "別", "勿", "not", "no", "never"}


@dataclass
class SentimentResult:
    score: float        # in [-1, 1]
    label: str          # "positive" / "neutral" / "negative"
    matched_terms: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)   # e.g. ["2330", "AAPL"]
    reasoning: str = ""                                # short rationale (Gemini only)


class LexiconAnalyzer:
    """Bilingual lexicon analyzer tuned for TW stock-board slang."""

    def __init__(
        self,
        positive=POSITIVE_TERMS,
        negative=NEGATIVE_TERMS,
        negations=NEGATION_TERMS,
    ):
        self.positive = positive
        self.negative = negative
        self.negations = negations

    def score(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(0.0, "neutral", [])
        text = text.lower()
        tokens = re.findall(r"[\u4e00-\u9fffA-Za-z]+", text)
        total = 0.0
        hits: list[str] = []
        for i, tok in enumerate(tokens):
            base = self.positive.get(tok, 0.0) + self.negative.get(tok, 0.0)
            if base == 0.0:
                continue
            # Simple negation flip if the previous token is a negation.
            if i > 0 and tokens[i - 1] in self.negations:
                base = -base
            total += base
            hits.append(tok)

        if not hits:
            return SentimentResult(0.0, "neutral", [])
        # Bound the score and apply a soft length normalization.
        norm = max(1.0, len(hits) ** 0.7)
        bounded = max(-1.0, min(1.0, total / (norm * 1.5)))
        label = (
            "positive" if bounded > 0.15
            else "negative" if bounded < -0.15
            else "neutral"
        )
        return SentimentResult(bounded, label, hits)

    def batch_score(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# 2. Gemini backend (recommended for production)
# ---------------------------------------------------------------------------

GEMINI_PROMPT = """You are a finance-savvy reader of Taiwanese investor chatter
(PTT Stock board, Dcard, news headlines). For the TEXT below, return STRICT JSON:

{"score": <float in [-1,1]>, "label": "positive|neutral|negative",
 "tickers": [<TW 4-digit or US ticker strings>], "reasoning": "<<=20 words>"}

Scoring rubric (from the perspective of the stock(s) discussed):
  +1.0 strongly bullish ("漲停", "歐印", "起飛", earnings beat)
   0.0 neutral / mixed / pure question
  -1.0 strongly bearish ("套牢", "停損", "崩盤", earnings miss)

Slang hints: 噴=bullish, 套=bearish, 歐印=all-in bullish, 韭菜=retail-loss,
畢業=blown account (bearish), 抄底=buy the dip (mildly bullish).

TEXT:
\"\"\"__TEXT__\"\"\"

Return ONLY the JSON object, no markdown fences."""


class GeminiAnalyzer:
    """Sentiment + ticker extraction via Google Gemini 2.5 Flash.

    Requires `google-genai` and the `GEMINI_API_KEY` env var. Falls back to
    `LexiconAnalyzer` on any API/network error so the app never crashes mid-demo.

    On quota-exhaustion (HTTP 429) we set ``self.quota_exhausted = True`` so
    callers can surface a visible warning and skip further API calls. The
    last error is also stashed on ``self.last_error`` for debugging.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        fallback: "LexiconAnalyzer | None" = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.fallback = fallback or LexiconAnalyzer()
        self._client = None
        self.quota_exhausted = False
        self.last_error: str = ""
        self._disk = _load_disk_cache()

    def _ensure(self):
        if self._client is None:
            from google import genai  # lazy import
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _parse(raw: str) -> dict:
        # Strip accidental ```json fences if the model adds them.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        return json.loads(cleaned)

    def _cache_key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_name}::{text}".encode("utf-8")).hexdigest()

    def score(self, text: str) -> SentimentResult:
        if not text or not self.api_key:
            return self.fallback.score(text)

        key = self._cache_key(text[:2000])
        if key in self._disk:
            d = self._disk[key]
            return SentimentResult(
                score=float(d["score"]),
                label=str(d["label"]),
                matched_terms=[],
                tickers=list(d.get("tickers", [])),
                reasoning=str(d.get("reasoning", "")),
            )

        if self.quota_exhausted:
            return self.fallback.score(text)

        return self._cached_score(text[:2000])

    @lru_cache(maxsize=4096)
    def _cached_score(self, text: str) -> SentimentResult:
        try:
            client = self._ensure()
            resp = client.models.generate_content(
                model=self.model_name,
                contents=GEMINI_PROMPT.replace("__TEXT__", text),
            )
            data = self._parse(resp.text or "")
            score = max(-1.0, min(1.0, float(data.get("score", 0.0))))
            label = str(data.get("label", "neutral")).lower()
            if label not in {"positive", "negative", "neutral"}:
                label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
            tickers = [str(t).upper() for t in data.get("tickers", []) if t]
            reasoning = str(data.get("reasoning", ""))[:200]
            result = SentimentResult(score, label, [], tickers, reasoning)
            # persist successful Gemini scores so reruns don't burn quota
            self._disk[self._cache_key(text)] = {
                "score": score, "label": label,
                "tickers": tickers, "reasoning": reasoning,
            }
            _save_disk_cache(self._disk)
            return result
        except Exception as exc:
            msg = str(exc)
            self.last_error = msg
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                self.quota_exhausted = True
            return self.fallback.score(text)

    def batch_score(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_default_analyzer():
    """Return the best analyzer available in the current environment.

    Priority:
      1. Gemini 3.5 Flash if `GEMINI_API_KEY` is set and `google-genai` installed
      2. Lexicon (always works, zero deps)
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            import google.genai  # noqa: F401  -- availability probe
            return GeminiAnalyzer()
        except ImportError:
            pass
    return LexiconAnalyzer()

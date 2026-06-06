"""Sentiment analyzer with three backends.

* `LexiconAnalyzer`     - Fast, dependency-free. Uses a small bilingual
                          lexicon tuned for TW stock-board slang.
* `GeminiAnalyzer`      - Calls Google Gemini 3.5 Flash. Best quality on
                          Traditional-Chinese PTT slang and effectively free
                          for our volume (<= ~1500 RPD on the free tier).
* `TransformerAnalyzer` - Optional local FinBERT (requires torch). Kept for
                          offline / no-network demos.

All return a `SentimentResult` with score in [-1.0, +1.0].
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, List

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
# 2. Transformer backend (optional)
# ---------------------------------------------------------------------------

class TransformerAnalyzer:
    """Wraps a HuggingFace finance sentiment pipeline.

    Default model: `ProsusAI/finbert` (English).  For Chinese, a good choice
    is `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`.  Model is loaded lazily
    so that import of this module stays cheap.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline  # local import keeps cold-start fast
            self._pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                truncation=True,
            )
        return self._pipe

    def score(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(0.0, "neutral", [])
        pipe = self._ensure()
        out = pipe(text[:512])[0]
        raw = str(out["label"]).lower()
        prob = float(out["score"])
        if "pos" in raw:
            return SentimentResult(prob, "positive", [])
        if "neg" in raw:
            return SentimentResult(-prob, "negative", [])
        return SentimentResult(0.0, "neutral", [])

    def batch_score(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# 3. Gemini backend (recommended for production)
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
    """Sentiment + ticker extraction via Google Gemini 3.5 Flash.

    Requires `google-genai` and the `GEMINI_API_KEY` env var. Falls back to
    `LexiconAnalyzer` on any API/network error so the app never crashes mid-demo.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.5-flash",
        api_key: str | None = None,
        fallback: "LexiconAnalyzer | None" = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.fallback = fallback or LexiconAnalyzer()
        self._client = None

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

    def score(self, text: str) -> SentimentResult:
        if not text or not self.api_key:
            return self.fallback.score(text)
        return self._cached_score(text[:2000])  # cap input length

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
            return SentimentResult(score, label, [], tickers, reasoning)
        except Exception:
            # Network down, quota hit, bad JSON, etc. — degrade gracefully.
            return self.fallback.score(text)

    def batch_score(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.score(t) for t in texts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_default_analyzer():
    """Return the best analyzer available in the current environment.

    Priority:
      1. Gemini 2.5 Flash if `GEMINI_API_KEY` is set and `google-genai` installed
      2. Lexicon (always works, zero deps)
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            import google.genai  # noqa: F401  -- availability probe
            return GeminiAnalyzer()
        except ImportError:
            pass
    return LexiconAnalyzer()

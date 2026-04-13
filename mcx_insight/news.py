from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import List

import feedparser
import requests

from mcx_insight.config import Instrument

# Google News RSS: no API key; reasonable for headline scanning
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


@dataclass
class NewsItem:
    title: str
    link: str
    published: str | None


def _fetch_rss(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "mcx-insight/0.1 (research; +https://example.local)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_headlines(instrument: Instrument, max_per_query: int = 6) -> List[NewsItem]:
    items: List[NewsItem] = []
    seen: set[str] = set()
    for q in instrument.news_queries:
        params = {"q": q, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
        url = f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"
        try:
            raw = _fetch_rss(url)
        except requests.RequestException:
            continue
        parsed = feedparser.parse(raw)
        for e in parsed.entries[:max_per_query]:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            key = title.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            pub = getattr(e, "published", None) or getattr(e, "updated", None)
            items.append(NewsItem(title=title.strip(), link=link, published=pub))
    return items

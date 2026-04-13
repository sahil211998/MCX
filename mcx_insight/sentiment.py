from __future__ import annotations

import re
from typing import List, Tuple

from mcx_insight.news import NewsItem

# Very small lexicon — good for demos; replace with a proper model if needed
BULLISH = re.compile(
    r"\b(rally|surge|rise|gain|bullish|higher|shortage|deficit|"
    r"cuts?|sanction|strike|disruption|outage|demand growth)\b",
    re.I,
)
BEARISH = re.compile(
    r"\b(slump|fall|drop|bearish|lower|glut|surplus|"
    r"recession|slowdown|inventory build|rate hike)\b",
    re.I,
)


def score_headlines(items: List[NewsItem]) -> Tuple[float, int, int]:
    bull = bear = 0
    for it in items:
        t = it.title
        if BULLISH.search(t):
            bull += 1
        if BEARISH.search(t):
            bear += 1
    total = bull + bear
    if total == 0:
        return 0.0, bull, bear
    # -1 .. +1
    score = (bull - bear) / total
    return score, bull, bear

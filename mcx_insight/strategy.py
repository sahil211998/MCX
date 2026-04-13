from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from mcx_insight import config
from mcx_insight.sentiment import score_headlines
from mcx_insight.news import NewsItem
from mcx_insight.technicals import enrich_indicators


Bias = Literal["cautious_long", "neutral", "cautious_short"]


@dataclass
class TradeLevels:
    """Demo stop/target from last close and ATR (not a recommendation)."""

    bias: Bias
    entry: float
    stop: float | None
    target: float | None
    atr: float
    rationale: str


@dataclass
class Outlook:
    instrument_label: str
    bias: Bias
    rationale: str
    last_close: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    suggested_entry_zone: str
    stop_loss_price: float | None
    stop_reason: str
    news_sentiment: float
    headline_count: int


def _latest(df: pd.DataFrame) -> pd.Series:
    keys = ["close", "rsi", "atr", f"ema_{config.EMA_FAST}", f"ema_{config.EMA_SLOW}"]
    keys = [k for k in keys if k in df.columns]
    sub = df.dropna(subset=keys) if keys else df.dropna()
    if sub.empty:
        sub = df.ffill().bfill()
    return sub.iloc[-1]


def _bias_from_row(row: pd.Series, news_score: float) -> tuple[Bias, str]:
    fast = row[f"ema_{config.EMA_FAST}"]
    slow = row[f"ema_{config.EMA_SLOW}"]
    rsi = row["rsi"]
    close = row["close"]

    news_long = news_score > 0.15
    news_short = news_score < -0.15

    tech_long = fast > slow and rsi > 45
    tech_short = fast < slow and rsi < 55
    # Pullback inside broader uptrend: easier long entry near slow EMA, fewer chase entries
    pullback_long = close > slow and fast < slow and rsi >= 40
    pullback_short = close < slow and fast > slow and rsi <= 60

    if tech_long or pullback_long or (fast > slow and news_long):
        if pullback_long and not tech_long:
            return (
                "cautious_long",
                "Uptrend pullback toward slow EMA — watch for stabilization; confirm on MCX.",
            )
        return "cautious_long", "Trend/momentum favors upside; confirm on MCX before acting."
    if tech_short or pullback_short or (fast < slow and news_short):
        if pullback_short and not tech_short:
            return (
                "cautious_short",
                "Downtrend pullback — fail at slow EMA would align shorts; confirm on MCX.",
            )
        return "cautious_short", "Trend/momentum favors downside; shorts need strong risk control."
    return "neutral", "Mixed technicals and news — no clear edge from this model."


def compute_trade_levels(
    ohlcv: pd.DataFrame,
    risk_reward: float | None = None,
    news_score: float = 0.0,
) -> TradeLevels | None:
    """ATR stop and fixed-R multiple target on the latest bar (technical + optional news score)."""
    if ohlcv.empty or len(ohlcv) < max(config.EMA_SLOW, config.ATR_PERIOD) + 2:
        return None
    need_lower = {"open", "high", "low", "close"}
    cols = {c.lower() for c in ohlcv.columns}
    if not need_lower.issubset(cols):
        ohlcv = ohlcv.rename(columns={c: c.lower() for c in ohlcv.columns})
    df = enrich_indicators(ohlcv)
    row = _latest(df)
    last = float(row["close"])
    atr_val = float(row["atr"])
    bias, rationale = _bias_from_row(row, news_score)
    rr = risk_reward if risk_reward is not None else config.TARGET_RISK_REWARD
    stop: float | None
    target: float | None
    if bias == "cautious_long":
        stop = last - config.ATR_STOP_MULTIPLIER * atr_val
        target = last + rr * (last - stop)
    elif bias == "cautious_short":
        stop = last + config.ATR_STOP_MULTIPLIER * atr_val
        target = last - rr * (stop - last)
    else:
        stop, target = None, None
    return TradeLevels(
        bias=bias,
        entry=last,
        stop=stop,
        target=target,
        atr=atr_val,
        rationale=rationale,
    )


def build_outlook(
    label: str,
    ohlcv: pd.DataFrame,
    headlines: list[NewsItem],
) -> Optional[Outlook]:
    if ohlcv.empty or len(ohlcv) < max(config.EMA_SLOW, config.ATR_PERIOD) + 2:
        return None

    df = enrich_indicators(ohlcv)
    row = _latest(df)
    last = float(row["close"])
    news_score, _, _ = score_headlines(headlines)
    bias, why = _bias_from_row(row, news_score)

    ema_ref = float(row[f"ema_{config.EMA_SLOW}"])
    atr_val = float(row["atr"])
    # "Easier" entry: zone near slow EMA (pullback) instead of chasing highs
    zone_lo = ema_ref - 0.15 * atr_val
    zone_hi = ema_ref + 0.15 * atr_val
    entry_zone = f"{zone_lo:.3f} – {zone_hi:.3f} (slow EMA pullback band)"

    stop: float | None
    stop_reason: str
    if bias == "cautious_long":
        stop = last - config.ATR_STOP_MULTIPLIER * atr_val
        stop_reason = f"ATR({config.ATR_PERIOD}) × {config.ATR_STOP_MULTIPLIER} below last close — wider stop, fewer noise wicks"
    elif bias == "cautious_short":
        stop = last + config.ATR_STOP_MULTIPLIER * atr_val
        stop_reason = f"ATR({config.ATR_PERIOD}) × {config.ATR_STOP_MULTIPLIER} above last close"
    else:
        stop = None
        stop_reason = "No directional bias — define risk only after you choose a side."

    return Outlook(
        instrument_label=label,
        bias=bias,
        rationale=why,
        last_close=last,
        ema_fast=float(row[f"ema_{config.EMA_FAST}"]),
        ema_slow=float(row[f"ema_{config.EMA_SLOW}"]),
        rsi=float(row["rsi"]),
        atr=atr_val,
        suggested_entry_zone=entry_zone,
        stop_loss_price=stop,
        stop_reason=stop_reason,
        news_sentiment=news_score,
        headline_count=len(headlines),
    )

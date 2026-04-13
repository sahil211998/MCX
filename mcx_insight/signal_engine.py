from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from mcx_insight import config
from mcx_insight.strategy import _latest, compute_trade_levels
from mcx_insight.technicals import enrich_signal_features


Direction = Literal["BUY", "SELL", "NO_TRADE"]
TrendLabel = Literal["Bullish", "Bearish", "Sideways"]


@dataclass
class SmartSignal:
    """Probability-based setup — not a profit guarantee (see disclaimer)."""

    symbol_key: str
    mcx_product: str
    direction: Direction
    timeframe: str
    entry: float | None
    target: float | None
    stop_loss: float | None
    risk_reward: float | None
    confidence_pct: int
    trend: TrendLabel
    pattern_summary: str
    indicators: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    disclaimer: str = "Probability-based analytics only. Not financial advice. Past signals ≠ future results."

    def to_db_row(self) -> dict[str, Any]:
        return {
            "symbol_key": self.symbol_key,
            "mcx_product": self.mcx_product,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "entry_price": self.entry,
            "target_price": self.target,
            "stop_loss": self.stop_loss,
            "risk_reward": self.risk_reward,
            "confidence_pct": self.confidence_pct,
            "trend": self.trend,
            "pattern_summary": self.pattern_summary,
            "indicators_json": self.indicators,
            "rationale": self.rationale,
            "disclaimer": self.disclaimer,
        }


def _normalize_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = ohlcv.copy()
    cols = {c: c.lower() for c in out.columns}
    out = out.rename(columns=cols)
    need = {"open", "high", "low", "close"}
    if not need.issubset(out.columns):
        raise ValueError("OHLCV must include open, high, low, close")
    return out


def _classify_trend(close: float, ema_slow: float, atr: float) -> TrendLabel:
    band = max(ema_slow * 0.002, atr * 0.1, 1e-9)
    if abs(close - ema_slow) <= band:
        return "Sideways"
    return "Bullish" if close > ema_slow else "Bearish"


def _pattern_tags(
    df: pd.DataFrame,
    row: pd.Series,
    bias: str,
    direction: Direction,
) -> list[str]:
    tags: list[str] = []
    if len(df) < 25:
        return tags
    h = df["high"]
    l = df["low"]
    high_10 = h.shift(1).rolling(10, min_periods=5).max()
    low_10 = l.shift(1).rolling(10, min_periods=5).min()
    last = float(row["close"])
    atr = float(row["atr"]) if pd.notna(row["atr"]) else 0.0

    if direction == "BUY" and last >= float(high_10.iloc[-1]) * 0.999:
        tags.append("Range breakout")
    if direction == "SELL" and last <= float(low_10.iloc[-1]) * 1.001:
        tags.append("Range breakdown")

    rsi_v = float(row["rsi"])
    if bias == "cautious_long" and 40 <= rsi_v <= 55:
        tags.append("RSI pullback zone")
    if bias == "cautious_short" and rsi_v >= 45 and rsi_v <= 60:
        tags.append("RSI relief bounce")

    hist = float(row["macd_hist"])
    hist_prev = float(df["macd_hist"].iloc[-2]) if len(df) > 1 else hist
    if direction == "BUY" and hist > 0 and hist > hist_prev:
        tags.append("MACD momentum")
    if direction == "SELL" and hist < 0 and hist < hist_prev:
        tags.append("MACD momentum")

    # Simplified double bottom / top: two lows within tight band over lookback
    look = l.iloc[-15:]
    if len(look) >= 12:
        lows = look.nsmallest(2)
        if lows.size == 2 and float(lows.max() - lows.min()) <= atr * 0.4:
            if direction == "BUY":
                tags.append("Double bottom (loose)")

    lookh = h.iloc[-15:]
    if len(lookh) >= 12:
        highs = lookh.nlargest(2)
        if highs.size == 2 and float(highs.max() - highs.min()) <= atr * 0.4:
            if direction == "SELL":
                tags.append("Double top (loose)")

    return list(dict.fromkeys(tags))


def _confidence(
    row: pd.Series,
    direction: Direction,
    pattern_count: int,
) -> int:
    if direction == "NO_TRADE":
        # Row-based neutral score so daily vs intraday bars don't both collapse to 55%.
        rsi_v = float(row.get("rsi", 50))
        hist = float(row.get("macd_hist", 0.0))
        close = float(row.get("close", 1.0))
        _atr = row.get("atr")
        atr = float(_atr) if _atr is not None and pd.notna(_atr) else 0.0
        atr_pct = atr / max(abs(close), 1e-9)
        base = 36
        if 40 <= rsi_v <= 60:
            base += 10
        elif rsi_v < 32 or rsi_v > 68:
            base += 6
        base += int(max(0, min(14, atr_pct * 800)))
        base += int(max(0, min(10, abs(hist) * 25)))
        return int(max(30, min(72, base)))
    score = 48
    rsi_v = float(row["rsi"])
    hist = float(row["macd_hist"])
    ema_f = float(row[f"ema_{config.EMA_FAST}"])
    ema_s = float(row[f"ema_{config.EMA_SLOW}"])
    close = float(row["close"])
    bb_mid = float(row["bb_mid"]) if pd.notna(row["bb_mid"]) else close

    if direction == "BUY":
        if ema_f > ema_s:
            score += 10
        if hist > 0:
            score += 12
        if 40 <= rsi_v <= 62:
            score += 8
        if close >= bb_mid:
            score += 5
    else:
        if ema_f < ema_s:
            score += 10
        if hist < 0:
            score += 12
        if 38 <= rsi_v <= 60:
            score += 8
        if close <= bb_mid:
            score += 5

    vol = row.get("volume")
    vs = row.get("vol_sma20")
    if vol is not None and vs is not None and pd.notna(vs) and float(vs) > 0:
        if float(vol) >= 1.1 * float(vs):
            score += 10

    score += min(12, pattern_count * 4)
    return int(max(0, min(95, round(score))))


def _compute_rr(entry: float, stop: float | None, target: float | None, direction: Direction) -> float | None:
    if stop is None or target is None or direction == "NO_TRADE":
        return None
    if direction == "BUY":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 4)


def generate_smart_signal(
    ohlcv: pd.DataFrame,
    *,
    symbol_key: str,
    mcx_product: str,
    timeframe: str = "1d",
    min_rr: float | None = None,
    news_score: float = 0.0,
) -> SmartSignal:
    """Confluence-based call; enforces minimum risk:reward before BUY/SELL."""
    min_rr = min_rr if min_rr is not None else float(config.TARGET_RISK_REWARD)
    df0 = _normalize_ohlcv(ohlcv)
    if df0.empty or len(df0) < max(config.EMA_SLOW, 26, 20) + 3:
        return SmartSignal(
            symbol_key=symbol_key,
            mcx_product=mcx_product,
            direction="NO_TRADE",
            timeframe=timeframe,
            entry=None,
            target=None,
            stop_loss=None,
            risk_reward=None,
            confidence_pct=0,
            trend="Sideways",
            pattern_summary="Insufficient history",
            indicators={},
            rationale="Need more bars for indicators and levels.",
        )

    levels = compute_trade_levels(df0, news_score=news_score)
    feat = enrich_signal_features(df0)
    row = _latest(feat)
    atr = float(row["atr"])
    trend = _classify_trend(float(row["close"]), float(row[f"ema_{config.EMA_SLOW}"]), atr)

    if levels is None or levels.bias == "neutral" or levels.stop is None or levels.target is None:
        ind = {
            "rsi": round(float(row["rsi"]), 2),
            "macd_hist": round(float(row["macd_hist"]), 6),
            "trend": trend,
        }
        return SmartSignal(
            symbol_key=symbol_key,
            mcx_product=mcx_product,
            direction="NO_TRADE",
            timeframe=timeframe,
            entry=float(row["close"]),
            target=None,
            stop_loss=None,
            risk_reward=None,
            confidence_pct=_confidence(row, "NO_TRADE", 0),
            trend=trend,
            pattern_summary="No qualifying directional bias",
            indicators=ind,
            rationale=levels.rationale if levels else "Neutral or incomplete levels.",
        )

    direction: Direction = "BUY" if levels.bias == "cautious_long" else "SELL"
    entry = float(levels.entry)
    stop = float(levels.stop)
    target = float(levels.target)
    rr = _compute_rr(entry, stop, target, direction)
    if rr is None or rr + 1e-6 < min_rr:
        ind = {
            "rsi": round(float(row["rsi"]), 2),
            "macd_hist": round(float(row["macd_hist"]), 6),
            "trend": trend,
            "computed_rr": rr,
            "min_rr": min_rr,
        }
        return SmartSignal(
            symbol_key=symbol_key,
            mcx_product=mcx_product,
            direction="NO_TRADE",
            timeframe=timeframe,
            entry=entry,
            target=target,
            stop_loss=stop,
            risk_reward=rr,
            confidence_pct=_confidence(row, "NO_TRADE", 0),
            trend=trend,
            pattern_summary="Risk-reward below minimum",
            indicators=ind,
            rationale=f"Setup filtered: R:R {rr} < required {min_rr}:1.",
        )

    tags = _pattern_tags(feat, row, levels.bias, direction)
    summary = " + ".join(tags) if tags else "Indicator confluence (no named pattern)"

    ind = {
        "rsi": round(float(row["rsi"]), 2),
        "macd_hist": round(float(row["macd_hist"]), 6),
        "ema_fast": round(float(row[f"ema_{config.EMA_FAST}"]), 4),
        "ema_slow": round(float(row[f"ema_{config.EMA_SLOW}"]), 4),
        "atr": round(atr, 4),
        "trend": trend,
        "volume_vs_sma20": None,
    }
    vol = row.get("volume")
    vs = row.get("vol_sma20")
    if vol is not None and vs is not None and pd.notna(vs) and float(vs) > 0:
        ind["volume_vs_sma20"] = round(float(vol) / float(vs), 3)

    conf = _confidence(row, direction, len(tags))

    return SmartSignal(
        symbol_key=symbol_key,
        mcx_product=mcx_product,
        direction=direction,
        timeframe=timeframe,
        entry=entry,
        target=target,
        stop_loss=stop,
        risk_reward=rr,
        confidence_pct=conf,
        trend=trend,
        pattern_summary=summary,
        indicators=ind,
        rationale=levels.rationale,
    )

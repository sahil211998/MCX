"""
Institutional-style context we can compute without venue order-book data.

Prop and desk workflows often layer **VWAP**, **relative volume**, **prior session/day levels**,
and tape/footprint tools (see e.g. common platform categories discussed in order-flow analytics).
This module adds the **price + volume structure** pieces; it is not a Bloomberg/ATAS substitute.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _norm_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def vwap_series(df: pd.DataFrame) -> pd.Series:
    """Anchored VWAP over the intraday Yahoo proxy series (e.g. 5d of 15m–30m bars)."""
    d = _norm_ohlcv_cols(df)
    high = pd.to_numeric(d["high"], errors="coerce")
    low = pd.to_numeric(d["low"], errors="coerce")
    close = pd.to_numeric(d["close"], errors="coerce")
    tp = (high + low + close) / 3.0
    vol = pd.to_numeric(d.get("volume", 0.0), errors="coerce").fillna(0.0)
    cv = vol.cumsum()
    ctp = (tp * vol).cumsum()
    return ctp / cv.replace(0, np.nan)


def relative_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float | None:
    d = _norm_ohlcv_cols(df)
    if "volume" not in d.columns or len(d) < lookback + 1:
        return None
    v = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    last = float(v.iloc[-1])
    base = float(v.iloc[-lookback:-1].mean())
    if base <= 0:
        return None
    return round(last / base, 3)


def volume_flow_proxy_last(df: pd.DataFrame) -> dict[str, Any] | None:
    """Coarse aggression proxy: close location in range × volume (no Time & Sales)."""
    d = _norm_ohlcv_cols(df)
    if d.empty or "volume" not in d.columns:
        return None
    row = d.iloc[-1]
    h = float(pd.to_numeric(row.get("high"), errors="coerce") or 0.0)
    low = float(pd.to_numeric(row.get("low"), errors="coerce") or 0.0)
    c = float(pd.to_numeric(row.get("close"), errors="coerce") or 0.0)
    vol = float(pd.to_numeric(row.get("volume"), errors="coerce") or 0.0)
    rng = h - low
    if rng <= 0 or vol <= 0:
        return {"signed_estimate": 0.0, "interpretation": "flat_or_no_volume"}
    pos = (c - low) / rng
    signed = (pos - 0.5) * 2.0 * vol
    if signed > vol * 0.15:
        interp = "buy_pressure"
    elif signed < -vol * 0.15:
        interp = "sell_pressure"
    else:
        interp = "balanced"
    return {
        "signed_estimate": round(signed, 2),
        "close_position_in_range": round(pos, 3),
        "interpretation": interp,
    }


def prev_trading_day_mcx_levels(daily_df: pd.DataFrame) -> dict[str, float] | None:
    """Previous calendar row in merged MCX daily (INR) — PD high/low/close for context."""
    if daily_df is None or len(daily_df) < 2:
        return None
    prev = daily_df.iloc[-2]

    def pick(*names: str) -> float | None:
        for n in names:
            if n in prev.index:
                try:
                    v = float(prev[n])
                    if math.isfinite(v):
                        return v
                except (TypeError, ValueError):
                    continue
        return None

    h = pick("High", "high")
    low = pick("Low", "low")
    c = pick("Close", "close")
    if h is None or low is None or c is None:
        return None
    return {"pd_high": round(h, 4), "pd_low": round(low, 4), "pd_close": round(c, 4)}


def build_institutional_context(
    intra_df: pd.DataFrame,
    daily_merged: pd.DataFrame,
    *,
    proxy_ticker: str | None,
    mcx_scale: float | None,
) -> dict[str, Any]:
    desk_practice = (
        "Many professional futures workflows stack **VWAP** and **relative volume** with **prior-day (PD) "
        "high/low/close** and classic trend tools (EMA, RSI, ATR, MACD, Bollinger). "
        "Dedicated order-flow software (footprint, delta, DOM) uses exchange microstructure; **this app does not** "
        "receive MCX L2/L3 tape—only public bhav, watch LTP, and Yahoo intraday proxy bars."
    )
    out: dict[str, Any] = {
        "summary": desk_practice,
        "metrics_implemented": [
            "Anchored intraday VWAP (Yahoo proxy) + price above/below",
            "Relative volume vs trailing bar average (RVOL-style)",
            "Bar-level volume pressure proxy (close-in-range, no true delta)",
            "Prior MCX session high/low/close (INR) for swing context",
            "Core signal still from EMA/RSI/ATR + MACD/BB confluence (see strategy/signal_engine)",
        ],
        "proxy_ticker": proxy_ticker,
        "prev_day_mcx_inr": prev_trading_day_mcx_levels(daily_merged),
        "vwap_proxy": None,
        "vwap_mcx_inr": None,
        "last_close_proxy": None,
        "last_close_mcx_inr": None,
        "price_vs_vwap": None,
        "relative_volume": None,
        "volume_flow_proxy": None,
    }

    if intra_df.empty or len(intra_df) < 5:
        out["note"] = "Insufficient intraday bars for VWAP / RVOL."
        return out

    d = _norm_ohlcv_cols(intra_df)
    if not {"high", "low", "close"}.issubset(d.columns):
        out["note"] = "Intraday frame missing OHLC."
        return out

    try:
        vw = vwap_series(d)
        vwap_last = float(vw.iloc[-1])
    except (TypeError, ValueError):
        vwap_last = float("nan")
    if not math.isfinite(vwap_last):
        return out

    close_px = float(pd.to_numeric(d["close"].iloc[-1], errors="coerce"))
    if not math.isfinite(close_px):
        return out

    mult = None
    if mcx_scale is not None and math.isfinite(mcx_scale) and mcx_scale > 0:
        mult = float(mcx_scale)

    eps = max(abs(vwap_last) * 1e-5, 1e-9)
    if close_px > vwap_last + eps:
        vs = "above"
    elif close_px < vwap_last - eps:
        vs = "below"
    else:
        vs = "at_vwap"
    bias = (
        "bullish_vs_vwap"
        if vs == "above"
        else "bearish_vs_vwap"
        if vs == "below"
        else "neutral_vs_vwap"
    )

    out["last_close_proxy"] = round(close_px, 6)
    out["vwap_proxy"] = round(vwap_last, 6)
    out["price_vs_vwap"] = {"position": vs, "bias_label": bias}
    if mult is not None:
        out["vwap_mcx_inr"] = round(vwap_last * mult, 4)
        out["last_close_mcx_inr"] = round(close_px * mult, 4)

    out["relative_volume"] = relative_volume_ratio(d)
    out["volume_flow_proxy"] = volume_flow_proxy_last(d)

    return out

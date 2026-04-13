from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

# Ensure non-interactive backend for servers (Render)
import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import

import mplfinance as mpf  # noqa: E402

from mcx_insight import config  # noqa: E402
from mcx_insight.catalog import yahoo_intraday_ticker  # noqa: E402
from mcx_insight.mcx_data import build_daily_ohlcv_mcx, live_quote, merge_live_into_ohlcv  # noqa: E402
from mcx_insight.prices import fetch_ohlcv  # noqa: E402
from mcx_insight.technicals import enrich_signal_features  # noqa: E402


Horizon = Literal["intraday", "long", "daily"]


@dataclass(frozen=True)
class ChartSpec:
    mcx_product: str
    horizon: Horizon
    entry: float | None
    stop: float | None
    target: float | None
    proxy_to_mcx_scale: float | None = None
    call_time_iso: str | None = None


def _scale_ohlc(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * float(scale)
    return out


def _yahoo_scaled_df(
    mcx_product: str,
    *,
    interval: str,
    period: str,
    scale: float | None,
) -> pd.DataFrame:
    """Yahoo proxy bars, optionally scaled into MCX INR space using a provided scale factor."""
    y = yahoo_intraday_ticker(mcx_product)
    if not y:
        return pd.DataFrame()

    proxy = fetch_ohlcv(y, period=period, interval=interval)
    if proxy.empty:
        return proxy
    if scale is None:
        return proxy
    try:
        s = float(scale)
    except Exception:
        return proxy
    if not (s > 0):
        return proxy
    return _scale_ohlc(proxy, s)


def _prep_df_for_mpf(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # mplfinance expects these exact column names (capitalized) or lowercase with `type`? We keep lowercase and pass `df`.
    # Ensure numeric and drop junk.
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"], how="any")
    return out


def render_call_chart_png(spec: ChartSpec) -> bytes:
    """
    Render a PNG chart showing:
    - Candles + volume
    - EMA fast/slow, Bollinger bands
    - RSI + MACD panels
    - Horizontal entry/stop/target lines (when present)
    """
    prod = spec.mcx_product.strip().upper()
    if spec.horizon == "daily":
        df = build_daily_ohlcv_mcx(prod, max_calendar_days=120, pause_seconds=0.02, skip_weekends=True, progress=False)
        q = live_quote(prod)
        if q is not None:
            df = merge_live_into_ohlcv(df, q)
    elif spec.horizon == "long":
        df = _yahoo_scaled_df(
            prod,
            interval=config.LONG_TERM_YAHOO_INTERVAL,
            period=config.LONG_TERM_YF_PERIOD,
            scale=spec.proxy_to_mcx_scale,
        )
    else:
        df = _yahoo_scaled_df(
            prod,
            interval=config.INTRADAY_INTERVAL,
            period=config.INTRADAY_YF_PERIOD,
            scale=spec.proxy_to_mcx_scale,
        )

    df = _prep_df_for_mpf(df)
    if df.empty or len(df) < 30:
        raise ValueError("Not enough bars to render chart")

    feat = enrich_signal_features(df)

    # overlays
    ema_fast = mpf.make_addplot(feat.get(f"ema_{config.EMA_FAST}"), color="#60a5fa", width=1.0)
    ema_slow = mpf.make_addplot(feat.get(f"ema_{config.EMA_SLOW}"), color="#f59e0b", width=1.0)
    ema_trend = mpf.make_addplot(feat.get(f"ema_{config.EMA_TREND}"), color="#22c55e", width=1.0, alpha=0.9)
    bb_u = mpf.make_addplot(feat.get("bb_upper"), color="#94a3b8", width=0.8, linestyle="dotted")
    bb_m = mpf.make_addplot(feat.get("bb_mid"), color="#64748b", width=0.8, linestyle="dotted")
    bb_l = mpf.make_addplot(feat.get("bb_lower"), color="#94a3b8", width=0.8, linestyle="dotted")

    # RSI panel (panel 1)
    rsi_ap = mpf.make_addplot(feat.get("rsi"), panel=1, color="#a78bfa", ylabel="RSI", width=1.0)
    # MACD panel (panel 2)
    macd_ap = mpf.make_addplot(feat.get("macd"), panel=2, color="#22c55e", ylabel="MACD", width=1.0)
    macd_sig_ap = mpf.make_addplot(feat.get("macd_signal"), panel=2, color="#ef4444", width=1.0)
    macd_hist = feat.get("macd_hist")
    hist_ap = mpf.make_addplot(macd_hist, panel=2, type="bar", color="#94a3b8", alpha=0.45)

    addplots = [ema_fast, ema_slow, ema_trend, bb_u, bb_m, bb_l, rsi_ap, macd_ap, macd_sig_ap, hist_ap]

    # horizontal levels
    hlines = []
    hcolors = []
    if spec.entry is not None:
        hlines.append(float(spec.entry))
        hcolors.append("#38bdf8")
    if spec.stop is not None:
        hlines.append(float(spec.stop))
        hcolors.append("#f87171")
    if spec.target is not None:
        hlines.append(float(spec.target))
        hcolors.append("#4ade80")
    hlines_kw: dict[str, Any] = {}
    if hlines:
        hlines_kw = dict(
            hlines=dict(hlines=hlines, colors=hcolors, linestyle="--", linewidths=[1.0] * len(hlines))
        )

    title = (
        f"{prod} · {spec.horizon} · EMA{config.EMA_FAST}/{config.EMA_SLOW}/{config.EMA_TREND} · "
        f"RSI{config.RSI_PERIOD} · MACD · BB(20)"
    )
    if spec.call_time_iso:
        title += f" · call {spec.call_time_iso}"

    fig, _axes = mpf.plot(
        feat,
        type="candle",
        volume=True,
        style="nightclouds",
        title=title,
        addplot=addplots,
        panel_ratios=(6, 2, 2),
        returnfig=True,
        figsize=(12, 8),
        tight_layout=True,
        **hlines_kw,
    )
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=140, bbox_inches="tight")
    bio.seek(0)
    return bio.read()


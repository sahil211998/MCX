from __future__ import annotations

import io
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

_PNG_CACHE: dict[tuple, tuple[float, bytes]] = {}
_PNG_CACHE_TTL_SECS = 60.0

# Cache raw OHLCV to keep chart loads <1s after warmup.
# Key includes product + horizon + scale (for proxy charts).
_DF_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_DF_CACHE_TTL_SECS = 300.0


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
    # Normalize column names from different sources:
    # - yfinance fetch_ohlcv() already outputs lowercase open/high/low/close/volume
    # - MCX bhav path may come in as Open/High/Low/Close/Volume or OPEN_/etc.
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "open_" in out.columns and "open" not in out.columns:
        out = out.rename(columns={"open_": "open"})
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    need = ["open", "high", "low", "close"]
    if not all(c in out.columns for c in need):
        missing = [c for c in need if c not in out.columns]
        raise ValueError(f"Missing OHLC columns for chart: {missing}")
    out = out.dropna(subset=need, how="any")
    return out


def _call_with_timeout(fn, *, timeout_secs: float, default):
    """Run `fn()` with a hard timeout (best-effort) to keep API responsive."""
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout_secs)
    except FuturesTimeoutError:
        fut.cancel()
        return default
    except Exception:
        return default
    finally:
        # Don't wait for stuck network calls.
        ex.shutdown(wait=False, cancel_futures=True)


def render_call_chart_png(spec: ChartSpec) -> bytes:
    """
    Render a PNG chart showing:
    - Candles + volume
    - EMA fast/slow, Bollinger bands
    - RSI + MACD panels
    - Horizontal entry/stop/target lines (when present)
    """
    cache_key = (
        spec.mcx_product.strip().upper(),
        spec.horizon,
        spec.entry,
        spec.stop,
        spec.target,
        spec.proxy_to_mcx_scale,
        spec.call_time_iso,
    )
    now = time.time()
    cached = _PNG_CACHE.get(cache_key)
    if cached and (now - cached[0]) <= _PNG_CACHE_TTL_SECS:
        return cached[1]

    prod = spec.mcx_product.strip().upper()
    df_cache_key = (prod, spec.horizon, spec.proxy_to_mcx_scale)
    df_cached = _DF_CACHE.get(df_cache_key)
    if df_cached and (now - df_cached[0]) <= _DF_CACHE_TTL_SECS:
        df = df_cached[1].copy()
    else:
        if spec.horizon == "daily":
            # Default to Yahoo daily bars to keep this endpoint fast (<1s).
            # MCX bhavcopy history is slow (1 request per day) and can hang on networks.
            df = _call_with_timeout(
                lambda: _yahoo_scaled_df(prod, interval="1d", period="6mo", scale=spec.proxy_to_mcx_scale),
                timeout_secs=6.0,
                default=pd.DataFrame(),
            )

            # Fallback: try MCX bhav daily only if Yahoo is unavailable.
            if df.empty:
                mcx_df = _call_with_timeout(
                    lambda: build_daily_ohlcv_mcx(
                        prod,
                        max_calendar_days=50,
                        pause_seconds=0.0,
                        skip_weekends=True,
                        progress=False,
                    ),
                    timeout_secs=6.0,
                    default=pd.DataFrame(),
                )
                if isinstance(mcx_df, pd.DataFrame) and not mcx_df.empty:
                    df = mcx_df
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
        _DF_CACHE[df_cache_key] = (now, df.copy())

    # Keep charts readable: render last N bars only.
    df = _prep_df_for_mpf(df).tail(240)
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

    # horizontal levels (only if they won't squash the candle panel)
    hlines = []
    hcolors = []
    last_close = float(feat["close"].iloc[-1])
    # If a level is too far from the current price, skip it to keep candles visible.
    max_pct = 0.35  # 35% away is likely a different scale / mismatch
    def _keep_level(v: float) -> bool:
        return abs(v - last_close) / max(abs(last_close), 1e-9) <= max_pct

    if spec.entry is not None:
        v = float(spec.entry)
        if _keep_level(v):
            hlines.append(v)
            hcolors.append("#38bdf8")
    if spec.stop is not None:
        v = float(spec.stop)
        if _keep_level(v):
            hlines.append(v)
            hcolors.append("#f87171")
    if spec.target is not None:
        v = float(spec.target)
        if _keep_level(v):
            hlines.append(v)
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

    # Proper green/red candle colors
    mc = mpf.make_marketcolors(
        up="#22c55e",
        down="#ef4444",
        edge="inherit",
        wick="inherit",
        volume="inherit",
    )
    s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridcolor="#243044")

    fig, _axes = mpf.plot(
        feat,
        type="candle",
        volume=True,
        volume_panel=3,
        style=s,
        title=title,
        addplot=addplots,
        panel_ratios=(7, 2, 2, 2),
        mav=(config.EMA_FAST, config.EMA_SLOW, config.EMA_TREND),
        returnfig=True,
        figsize=(13, 9),
        tight_layout=True,
        **hlines_kw,
    )
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=140, bbox_inches="tight")
    bio.seek(0)
    png = bio.read()
    _PNG_CACHE[cache_key] = (now, png)
    return png


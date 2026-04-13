from __future__ import annotations

import yfinance as yf
import pandas as pd

from mcx_insight import config


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(level0).strip().lower().replace(" ", "_") for level0, *_ in out.columns]
    else:
        out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    # Prefer unadjusted close for levels consistent with high/low on daily bars
    if "adj_close" in out.columns and "close" not in out.columns:
        out = out.rename(columns={"adj_close": "close"})
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    return out[keep].sort_index()


def fetch_ohlcv(ticker: str, period: str | None = None, interval: str | None = None) -> pd.DataFrame:
    period = period or config.DEFAULT_PERIOD
    interval = interval or config.DEFAULT_INTERVAL
    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    return _normalize_columns(raw)

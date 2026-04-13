from __future__ import annotations

import numpy as np
import pandas as pd

from mcx_insight import config


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal_span: int = 9):
    ema_f = ema(close, fast)
    ema_s = ema(close, slow)
    line = ema_f - ema_s
    sig = ema(line, signal_span)
    hist = line - sig
    return line, sig, hist


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    return upper, mid, lower


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "close" not in out.columns:
        raise ValueError("DataFrame must have lowercase OHLC columns")
    c = out["close"]
    out[f"ema_{config.EMA_FAST}"] = ema(c, config.EMA_FAST)
    out[f"ema_{config.EMA_SLOW}"] = ema(c, config.EMA_SLOW)
    out["rsi"] = rsi(c, config.RSI_PERIOD)
    out["atr"] = atr(out, config.ATR_PERIOD)
    return out


def enrich_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """EMA/RSI/ATR + MACD, Bollinger, volume trend for confluence scoring."""
    out = enrich_indicators(df)
    c = out["close"]
    line, sig, hist = macd(c)
    out["macd"] = line
    out["macd_signal"] = sig
    out["macd_hist"] = hist
    bu, bm, bl = bollinger(c)
    out["bb_upper"] = bu
    out["bb_mid"] = bm
    out["bb_lower"] = bl
    if "volume" in out.columns:
        v = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
        out["vol_sma20"] = v.rolling(20, min_periods=5).mean()
    else:
        out["vol_sma20"] = np.nan
    return out

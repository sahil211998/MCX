from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

import mcxlib
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


@dataclass
class LiveQuote:
    symbol: str
    expiry: str
    ltp: float
    open_: float
    high: float
    low: float
    volume: float
    previous_close: float


def get_live_futcom_row(product_code: str) -> Optional[pd.Series]:
    """Latest MCX FUTCOM row for product (highest volume among LTP>0)."""
    d = mcxlib.get_market_watch()
    m = d[(d["InstrumentName"] == "FUTCOM") & (d["ProductCode"] == product_code.upper())]
    m = m[pd.to_numeric(m["LTP"], errors="coerce").fillna(0) > 0]
    if m.empty:
        return None
    m = m.copy()
    m["_vol"] = pd.to_numeric(m["Volume"], errors="coerce").fillna(0)
    return m.sort_values("_vol", ascending=False).iloc[0]


def live_quote(product_code: str) -> Optional[LiveQuote]:
    row = get_live_futcom_row(product_code)
    if row is None:
        return None
    def f(col: str) -> float:
        return float(pd.to_numeric(row.get(col), errors="coerce") or 0.0)

    return LiveQuote(
        symbol=str(row["Symbol"]).strip(),
        expiry=str(row["ExpiryDate"]).strip(),
        ltp=f("LTP"),
        open_=f("Open") or f("LTP"),
        high=f("High") or f("LTP"),
        low=f("Low") or f("LTP"),
        volume=f("Volume"),
        previous_close=f("PreviousClose"),
    )


def _bhav_day(yyyymmdd: str) -> Optional[pd.DataFrame]:
    try:
        return mcxlib.get_bhav_copy(trade_date=yyyymmdd, instrument="FUTCOM")
    except ValueError:
        return None


def _pick_liquid_row(bhav: pd.DataFrame, product_code: str) -> Optional[pd.Series]:
    sym = bhav["Symbol"].astype(str).str.strip().str.upper()
    sub = bhav[sym == product_code.upper()].copy()
    if sub.empty:
        return None
    for col in ("Close", "Open", "High", "Low"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub["Volume"] = pd.to_numeric(sub["Volume"], errors="coerce").fillna(0)
    sub = sub[(sub["Close"] > 0) | (sub["Open"] > 0)]
    if sub.empty:
        return None
    return sub.sort_values("Volume", ascending=False).iloc[0]


def build_daily_ohlcv_mcx(
    product_code: str,
    max_calendar_days: int = 120,
    pause_seconds: float = 0.2,
    skip_weekends: bool = True,
    progress: bool = False,
) -> pd.DataFrame:
    """Daily OHLCV from MCX bhavcopy (public), one row per day (most active expiry)."""
    rows: list[dict] = []
    today = datetime.now(IST).date()
    for i in range(max_calendar_days):
        dt = today - timedelta(days=i)
        if skip_weekends and dt.weekday() >= 5:
            continue
        ymd = dt.strftime("%Y%m%d")
        if progress and i > 0 and i % 12 == 0:
            print(f"  bhav {ymd} … ({len(rows)} bars)", flush=True)
        bhav = _bhav_day(ymd)
        if pause_seconds:
            time.sleep(pause_seconds)
        if bhav is None or bhav.empty:
            continue
        row = _pick_liquid_row(bhav, product_code)
        if row is None:
            continue
        rows.append(
            {
                "Date": pd.Timestamp(dt),
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": float(row["Volume"]),
                "Expiry": str(row["ExpiryDate"]).strip(),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def merge_live_into_ohlcv(df: pd.DataFrame, quote: LiveQuote) -> pd.DataFrame:
    """Update or append today's bar using live LTP / session high-low (IST date)."""
    if df.empty:
        ts = pd.Timestamp(datetime.now(IST).date())
        return pd.DataFrame(
            [[quote.open_, quote.high, quote.low, quote.ltp, quote.volume]],
            index=[ts],
            columns=["Open", "High", "Low", "Close", "Volume"],
        )
    out = df.copy()
    ts = pd.Timestamp(datetime.now(IST).date())
    o, h, low, c, v = quote.open_, quote.high, quote.low, quote.ltp, quote.volume
    if ts in out.index:
        cur = out.loc[ts]
        out.loc[ts, "Open"] = float(o) if o else float(cur["Open"])
        out.loc[ts, "High"] = max(float(cur["High"]), float(h), float(c))
        out.loc[ts, "Low"] = min(float(cur["Low"]), float(low), float(c)) if low > 0 else float(cur["Low"])
        out.loc[ts, "Close"] = float(c)
        out.loc[ts, "Volume"] = max(float(cur["Volume"]), float(v))
    else:
        out.loc[ts] = [o, h, low, c, v]
        out = out.sort_index()
    return out

"""
MCX contracts do not map 1:1 to free APIs. We use highly correlated
international futures (COMEX/NYMEX) as *proxies* for directional context.
Treat outputs as research only, not trade recommendations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


CONFIDENCE_WEIGHT_DAILY = 0.45
CONFIDENCE_WEIGHT_INTRADAY = 0.55
CONFIDENCE_SCORE_CAP = 95

# Stored on each signal row + shown in UI (not win-rate or profit certainty).
CONFIDENCE_SCORE_EXPLANATION = (
    "Confluence score from indicators and pattern tags (0–95 scale). "
    "This is not probability of profit, expected win %, or a guarantee."
)

# Stored on each dual-horizon row: trade call is intraday-first; long-term is Yahoo higher-TF swing context.
PRIMARY_CALL_SCOPE = "intraday"
CALL_SCOPE_LABEL = (
    "Primary trade call is INTRADAY: Yahoo proxy bars (default 30m, configurable 15–30m) for structure, prices converted to MCX INR. "
    "Long-term leg uses Yahoo 2–4h-style bars (default 4h; 1h optional via env), scaled to MCX INR; MCX daily bhav remains for desk context (prior day, etc.)."
)


def blend_confidence_pct(daily_pct: int, intraday_pct: int) -> int:
    """Single headline “how strong is the setup” number for DB/UI."""
    v = CONFIDENCE_WEIGHT_DAILY * float(daily_pct) + CONFIDENCE_WEIGHT_INTRADAY * float(
        intraday_pct
    )
    return int(max(0, min(CONFIDENCE_SCORE_CAP, round(v))))


@dataclass(frozen=True)
class Instrument:
    """One tradable theme: MCX product code, yahoo proxy, news queries."""

    mcx_label: str
    mcx_product: str
    yahoo_ticker: str
    news_queries: List[str]


# CLI / research helpers — same six mini + gas roots as the web app (no COPPER / full CRUDEOIL / ALUMINIUM).
INSTRUMENTS = {
    "alumini": Instrument(
        mcx_label="MCX Aluminium Mini (ALUMINI)",
        mcx_product="ALUMINI",
        yahoo_ticker="HG=F",
        news_queries=["aluminium MCX India", "base metals"],
    ),
    "zincmini": Instrument(
        mcx_label="MCX Zinc Mini (ZINCMINI)",
        mcx_product="ZINCMINI",
        yahoo_ticker="HG=F",
        news_queries=["zinc LME", "base metals MCX"],
    ),
    "silverm": Instrument(
        mcx_label="MCX Silver Mini (SILVERM)",
        mcx_product="SILVERM",
        yahoo_ticker="SI=F",
        news_queries=["silver commodities", "MCX silver"],
    ),
    "crudeoilm": Instrument(
        mcx_label="MCX Crude Oil Mini (CRUDEOILM)",
        mcx_product="CRUDEOILM",
        yahoo_ticker="CL=F",
        news_queries=["crude oil OPEC", "energy commodities"],
    ),
    "naturalgas": Instrument(
        mcx_label="MCX Natural Gas (NATURALGAS)",
        mcx_product="NATURALGAS",
        yahoo_ticker="NG=F",
        news_queries=["natural gas LNG weather"],
    ),
    "natgasmini": Instrument(
        mcx_label="MCX Natural Gas Mini (NATGASMINI)",
        mcx_product="NATGASMINI",
        yahoo_ticker="NG=F",
        news_queries=["natural gas MCX", "NG futures"],
    ),
}


DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "1d"

# Intraday review window: Yahoo proxy — use 15m or 30m (2nd is smoother; both in 15–30m range).
INTRADAY_INTERVAL = _env_str("MCX_INTRADAY_INTERVAL", "30m")
INTRADAY_YF_PERIOD = _env_str("MCX_INTRADAY_YF_PERIOD", "5d")

# Long-term / swing: Yahoo bars in ~2–4h range (yfinance supports 1h and 4h; 2h not supported — use 1h or 4h).
LONG_TERM_YAHOO_INTERVAL = _env_str("MCX_LONG_TERM_INTERVAL", "4h")
LONG_TERM_YF_PERIOD = _env_str("MCX_LONG_TERM_YF_PERIOD", "730d")
LONG_TERM_MIN_BARS = 45

# Strategy defaults (tunable)
# Common trader set: 10/20 for momentum + 50 for trend filter.
EMA_FAST = 10
EMA_SLOW = 20
EMA_TREND = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 1.8  # wider than 1.0 → fewer noise stop-outs, larger $ risk per lot
TARGET_RISK_REWARD = 2.0  # minimum R:R for actionable signals (signal_engine); daily target from ATR model before % override
INTRADAY_MIN_RR = 1.0  # pre-refine gate; final intraday uses 1% SL + 1% T → ~1:1
# Intraday levels vs entry (after signal direction is set); scaled to MCX INR later.
INTRADAY_STOP_PCT = 0.01  # 1% adverse move from entry (stop placement)
INTRADAY_TARGET_PCT = 0.01  # 1% favorable move from intraday entry
LONG_TERM_TARGET_PCT = 0.03  # 3% favorable move from long-term (swing) entry

# Only these MCX FUTCOM roots are listed and get trade calls in this app (minis + gas).
# MCX codes: Silver Mini = SILVERM, Crude Mini = CRUDEOILM, Natural Gas Mini = NATGASMINI.
SIGNAL_ONLY_MCX_PRODUCTS: tuple[str, ...] = (
    "ALUMINI",
    "ZINCMINI",
    "SILVERM",
    "CRUDEOILM",
    "NATURALGAS",
    "NATGASMINI",
)
SIGNAL_ONLY_MCX_SET = frozenset(SIGNAL_ONLY_MCX_PRODUCTS)

SIGNAL_PRODUCT_LABEL: dict[str, str] = {
    "ALUMINI": "Aluminium Mini (ALUMINI)",
    "ZINCMINI": "Zinc Mini (ZINCMINI)",
    "SILVERM": "Silver Mini (SILVERM)",
    "CRUDEOILM": "Crude Oil Mini (CRUDEOILM)",
    "NATURALGAS": "Natural Gas (NATURALGAS)",
    "NATGASMINI": "Natural Gas Mini (NATGASMINI)",
}

# Back-compat alias for batch jobs that used theme keys (unused for minis-only UI).
DASHBOARD_SYMBOL_KEYS = SIGNAL_ONLY_MCX_PRODUCTS

# Intraday Yahoo proxy — only the six enabled roots (no full-size ALUMINIUM / CRUDEOIL / COPPER).
YAHOO_INTRADAY_BY_PRODUCT: dict[str, str] = {
    "ALUMINI": "HG=F",
    "ZINCMINI": "HG=F",
    "SILVERM": "SI=F",
    "CRUDEOILM": "CL=F",
    "NATURALGAS": "NG=F",
    "NATGASMINI": "NG=F",
}

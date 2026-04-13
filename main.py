#!/usr/bin/env python3
"""
MCX-oriented *research* dashboard: global proxy prices + headlines + simple rules.

Disclaimer: Not financial advice. Markets cannot be reliably predicted;
this tool combines public data for learning and risk-aware framing only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python main.py` from project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcx_insight.config import INSTRUMENTS
from mcx_insight.news import fetch_headlines
from mcx_insight.prices import fetch_ohlcv
from mcx_insight.strategy import build_outlook


def run(symbol: str, period: str, interval: str) -> int:
    key = symbol.strip().lower()
    if key not in INSTRUMENTS:
        print(f"Unknown symbol '{symbol}'. Choose one of: {', '.join(sorted(INSTRUMENTS))}")
        return 2

    ins = INSTRUMENTS[key]
    print(f"\n=== {ins.mcx_label} ===")
    print(f"Proxy ticker: {ins.yahoo_ticker}\n")

    ohlcv = fetch_ohlcv(ins.yahoo_ticker, period=period, interval=interval)
    headlines = fetch_headlines(ins)

    outlook = build_outlook(ins.mcx_label, ohlcv, headlines)
    if outlook is None:
        print("Not enough data to compute indicators.")
        return 1

    print("--- Latest technical snapshot (proxy contract) ---")
    print(f"  Last close:     {outlook.last_close:.4f}")
    print(f"  EMA fast/slow:  {outlook.ema_fast:.4f} / {outlook.ema_slow:.4f}")
    print(f"  RSI:            {outlook.rsi:.1f}")
    print(f"  ATR:            {outlook.atr:.4f}")

    print("\n--- Model bias (demo rules, not a forecast) ---")
    print(f"  Bias: {outlook.bias}")
    print(f"  Note: {outlook.rationale}")

    print("\n--- Entry / risk framing ---")
    print(f"  Pullback-style zone: {outlook.suggested_entry_zone}")
    if outlook.stop_loss_price is not None:
        print(f"  Volatility stop:     {outlook.stop_loss_price:.4f}")
    print(f"  Stop logic:          {outlook.stop_reason}")

    print("\n--- Headline sentiment (keyword scan) ---")
    print(f"  Score [-1,1]: {outlook.news_sentiment:+.2f}  (from {outlook.headline_count} headlines)")
    if headlines:
        print("  Recent headlines:")
        for h in headlines[:8]:
            print(f"    • {h.title}")

    print(
        "\n* Align with live MCX charts and your broker's margins before trading. "
        "Wider ATR stops reduce whipsaws but increase loss if wrong.*\n"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="MCX proxy + news research tool")
    p.add_argument(
        "symbol",
        nargs="?",
        default="alumini",
        help=f"One of: {', '.join(sorted(INSTRUMENTS))}",
    )
    p.add_argument("--period", default="6mo", help="yfinance period, e.g. 3mo, 6mo, 1y")
    p.add_argument("--interval", default="1d", help="yfinance interval, e.g. 1d, 1h")
    args = p.parse_args()
    raise SystemExit(run(args.symbol, args.period, args.interval))


if __name__ == "__main__":
    main()

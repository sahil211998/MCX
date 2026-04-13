#!/usr/bin/env python3
"""
Live-updating MCX FUTCOM chart (daily candles from MCX bhavcopy + latest LTP),
with ATR stop and fixed risk/reward target lines.

Data: public MCX endpoints via `mcxlib` (may lag / fail outside market hours).
Not financial advice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import mplfinance as mpf

from mcx_insight.config import INSTRUMENTS
from mcx_insight.mcx_data import build_daily_ohlcv_mcx, live_quote, merge_live_into_ohlcv
from mcx_insight.strategy import compute_trade_levels


def _load_ohlcv(symbol_key: str, calendar_days: int, pause: float, progress: bool):
    ins = INSTRUMENTS[symbol_key]
    df = build_daily_ohlcv_mcx(
        ins.mcx_product,
        max_calendar_days=calendar_days,
        pause_seconds=pause,
        progress=progress,
    )
    q = live_quote(ins.mcx_product)
    if q:
        df = merge_live_into_ohlcv(df, q)
    return ins, df, q


def _subtitle(ins, q, levels) -> str:
    parts = [ins.mcx_label, f"product={ins.mcx_product}"]
    if q:
        parts.append(f"LTP={q.ltp:g} exp {q.expiry} vol={q.volume:g}")
    if levels:
        parts.append(f"bias={levels.bias} entry≈{levels.entry:g}")
        if levels.stop and levels.target:
            parts.append(f"SL={levels.stop:g} T={levels.target:g}")
    return " | ".join(parts)


def run_static(symbol: str, calendar_days: int, pause: float, out: Path | None, progress: bool) -> int:
    ins, df, q = _load_ohlcv(symbol, calendar_days, pause, progress)
    if df.empty or len(df) < 8:
        print("Not enough MCX daily data yet (try --days 60 or check MCX/bhav availability).")
        return 1
    levels = compute_trade_levels(df.rename(columns=lambda c: c.lower()))
    title = _subtitle(ins, q, levels)
    if levels is None:
        title += " | (need more daily bars for EMA/ATR — increase --days)"
    hlines_kw = {}
    if levels and levels.stop is not None and levels.target is not None:
        hlines_kw = dict(
            hlines=dict(
                hlines=[float(levels.stop), float(levels.target)],
                colors=["crimson", "forestgreen"],
                linestyle="--",
                linewidths=[1.2, 1.2],
            )
        )
    fig, _axes = mpf.plot(
        df,
        type="candle",
        style="yahoo",
        volume=True,
        title=title,
        returnfig=True,
        figsize=(12, 7),
        tight_layout=True,
        **hlines_kw,
    )
    if out:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Saved {out.resolve()}")
    else:
        plt.show()
    plt.close(fig)
    return 0


def run_live(symbol: str, calendar_days: int, pause: float, refresh: float, progress: bool) -> int:
    ins = INSTRUMENTS[symbol]
    print("Loading MCX daily history (one bhav request per trading day; please wait)…", flush=True)
    base = build_daily_ohlcv_mcx(
        ins.mcx_product,
        max_calendar_days=calendar_days,
        pause_seconds=pause,
        progress=progress,
    )
    if base.empty or len(base) < 8:
        print("Not enough history. Increase --days.")
        return 1
    print("Live loop — Ctrl+C to stop. Re-fetches LTP each refresh; daily history is loaded once.")
    plt.ion()
    fig = None
    try:
        while True:
            q = live_quote(ins.mcx_product)
            df = merge_live_into_ohlcv(base, q) if q else base.copy()
            levels = compute_trade_levels(df.rename(columns=lambda c: c.lower()))
            title = _subtitle(ins, q, levels)
            if levels is None:
                title += " | (increase --days for EMA/ATR levels)"
            hlines_kw = {}
            if levels and levels.stop is not None and levels.target is not None:
                hlines_kw = dict(
                    hlines=dict(
                        hlines=[float(levels.stop), float(levels.target)],
                        colors=["crimson", "forestgreen"],
                        linestyle="--",
                        linewidths=[1.2, 1.2],
                    )
                )
            if fig is not None:
                plt.close(fig)
            fig, _axes = mpf.plot(
                df,
                type="candle",
                style="yahoo",
                volume=True,
                title=title,
                returnfig=True,
                figsize=(12, 7),
                tight_layout=True,
                **hlines_kw,
            )
            plt.show(block=False)
            plt.pause(refresh)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        if fig is not None:
            plt.close(fig)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="MCX live chart with stop/target lines")
    p.add_argument(
        "symbol",
        nargs="?",
        default="alumini",
        help=f"One of: {', '.join(sorted(INSTRUMENTS))}",
    )
    p.add_argument("--days", type=int, default=70, help="Calendar days to scan for daily bhav (more → more trading bars)")
    p.add_argument("--pause", type=float, default=0.05, help="Pause seconds between MCX bhav requests")
    p.add_argument("--live", action="store_true", help="Loop: refresh chart every --refresh seconds")
    p.add_argument("--refresh", type=float, default=60.0, help="Seconds between live updates (with --live)")
    p.add_argument("--save", type=Path, default=None, help="Save one PNG and exit (no interactive window)")
    p.add_argument("--progress", action="store_true", help="Print bhav download progress")
    args = p.parse_args()
    key = args.symbol.strip().lower()
    if key not in INSTRUMENTS:
        print(f"Unknown symbol. Choose: {', '.join(sorted(INSTRUMENTS))}")
        raise SystemExit(2)
    if args.live and args.save:
        print("Use either --live or --save, not both.")
        raise SystemExit(2)
    if args.live:
        raise SystemExit(run_live(key, args.days, args.pause, args.refresh, args.progress))
    raise SystemExit(run_static(key, args.days, args.pause, args.save, args.progress))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fetch MCX daily bars + live quote for allowed mini/gas products only,
compute ATR stop / target, store in PostgreSQL.

Not financial advice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcx_insight.config import SIGNAL_ONLY_MCX_PRODUCTS, SIGNAL_ONLY_MCX_SET
from mcx_insight.db import connect_pg, ensure_schema, insert_trade_levels, pg_config_from_env
from mcx_insight.mcx_data import build_daily_ohlcv_mcx, live_quote, merge_live_into_ohlcv
from mcx_insight.strategy import compute_trade_levels

DEFAULT_PRODUCTS = SIGNAL_ONLY_MCX_PRODUCTS


def predict_and_store(mcx_products: tuple[str, ...], calendar_days: int, pause: float) -> int:
    cfg = pg_config_from_env()
    try:
        conn = connect_pg(cfg)
    except Exception as e:
        print(f"PostgreSQL connection failed ({cfg.host}:{cfg.port}/{cfg.dbname}): {e}")
        print("Start Postgres and ensure database exists, or set PGHOST/PGUSER/PGPASSWORD.")
        return 1

    try:
        ensure_schema(conn)
        for raw in mcx_products:
            prod = raw.strip().upper()
            if prod not in SIGNAL_ONLY_MCX_SET:
                print(f"Skip {prod!r} — not in allowed mini/gas list.", flush=True)
                continue
            sk = prod.lower()
            print(f"Loading {prod}…", flush=True)
            base = build_daily_ohlcv_mcx(
                prod,
                max_calendar_days=calendar_days,
                pause_seconds=pause,
                skip_weekends=True,
                progress=False,
            )
            q = live_quote(prod)
            df = merge_live_into_ohlcv(base, q) if q is not None else base
            if df.empty:
                print(f"  No OHLCV for {prod}.")
                insert_trade_levels(
                    conn,
                    symbol_key=sk,
                    mcx_product=prod,
                    levels=None,
                    live_quote=q,
                )
                continue
            levels = compute_trade_levels(df.rename(columns=lambda c: c.lower()))
            insert_trade_levels(
                conn,
                symbol_key=sk,
                mcx_product=prod,
                levels=levels,
                live_quote=q,
            )
            if levels and levels.stop is not None and levels.target is not None:
                print(
                    f"  Stored: bias={levels.bias} entry≈{levels.entry:.4f} "
                    f"SL={levels.stop:.4f} target={levels.target:.4f}"
                )
            elif levels:
                print(f"  Stored: bias={levels.bias} entry≈{levels.entry:.4f} (neutral — no SL/target)")
            else:
                print("  Stored: insufficient bars for model.")
        print("Done.")
        return 0
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="MCX mini/gas targets → PostgreSQL")
    p.add_argument(
        "products",
        nargs="*",
        help=f"MCX product codes (default: all allowed — {', '.join(DEFAULT_PRODUCTS)})",
    )
    p.add_argument("--days", type=int, default=90, help="Calendar days to scan for bhav history")
    p.add_argument("--pause", type=float, default=0.05, help="Seconds between MCX bhav requests")
    args = p.parse_args()
    prods = tuple(args.products) if len(args.products) > 0 else DEFAULT_PRODUCTS
    raise SystemExit(predict_and_store(prods, args.days, args.pause))


if __name__ == "__main__":
    main()

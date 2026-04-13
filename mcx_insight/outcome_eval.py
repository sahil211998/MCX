"""
Classify stored intraday call outcome vs latest MCX LTP (FUTCOM watch).

This is a **snapshot check**, not a tick-accurate “which traded first” study.
"""

from __future__ import annotations

import math
from typing import Any


def classify_intraday_vs_ltp(
    *,
    intraday_direction: str | None,
    entry: Any,
    stop: Any,
    target: Any,
    ltp: float | None,
) -> tuple[str, str]:
    """
    Returns (outcome_code, human_note).
    outcome_code: target_hit | stop_hit | open | no_trade | unavailable
    """
    idir = (intraday_direction or "").strip().upper() or "NO_TRADE"
    if idir == "NO_TRADE":
        return "no_trade", "Intraday NO_TRADE — not scored for target accuracy."
    if idir not in ("BUY", "SELL"):
        return "no_trade", f"Intraday direction {intraday_direction!r} not scored."

    if ltp is None or not math.isfinite(float(ltp)):
        return "unavailable", "No usable live LTP from MCX market watch."

    def fnum(x: Any, label: str) -> float | None:
        if x is None:
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return v

    e = fnum(entry, "entry")
    sl = fnum(stop, "stop")
    tg = fnum(target, "target")
    if e is None or sl is None or tg is None:
        return "unavailable", "Missing or invalid intraday entry/stop/target for scoring."

    ltp = float(ltp)
    if idir == "BUY":
        if ltp >= tg:
            return (
                "target_hit",
                f"LTP {ltp:.4f} ≥ target {tg:.4f} (entry {e:.4f}, stop {sl:.4f}).",
            )
        if ltp <= sl:
            return "stop_hit", f"LTP {ltp:.4f} ≤ stop {sl:.4f} (target was {tg:.4f})."
        return "open", f"LTP {ltp:.4f} between stop {sl:.4f} and target {tg:.4f} — still open."
    # SELL
    if ltp <= tg:
        return (
            "target_hit",
            f"LTP {ltp:.4f} ≤ target {tg:.4f} (entry {e:.4f}, stop {sl:.4f}).",
        )
    if ltp >= sl:
        return "stop_hit", f"LTP {ltp:.4f} ≥ stop {sl:.4f} (target was {tg:.4f})."
    return "open", f"LTP {ltp:.4f} between target {tg:.4f} and stop {sl:.4f} — still open."


def intraday_live_status(
    *,
    intraday_direction: str | None,
    entry: Any,
    stop: Any,
    target: Any,
    ltp: float | None,
) -> tuple[str, str, str, int]:
    """
    Live snapshot for UI: phase code, short label, detail note, stage number.

    Stages:
      1 — Waiting (BUY only): LTP still below entry while between stop & target.
      2 — Active: stop/target not hit. BUY: LTP ≥ entry. SELL: any LTP between target & stop
         (short “zone” is the whole band; entry is a reference level, not a gate).
      3 — Target hit
      4 — Stoploss hit

    phase: stop_hit | target_hit | active | waiting_entry | no_trade | unavailable
    """
    idir = (intraday_direction or "").strip().upper() or "NO_TRADE"
    if idir == "NO_TRADE":
        return (
            "no_trade",
            "No trade",
            "Intraday NO_TRADE — no entry/stop/target stage.",
            0,
        )
    if idir not in ("BUY", "SELL"):
        return ("no_trade", "No trade", f"Direction {intraday_direction!r} not scored.", 0)

    if ltp is None or not math.isfinite(float(ltp)):
        return (
            "unavailable",
            "No live LTP",
            "No usable live LTP from MCX market watch.",
            0,
        )

    def fnum(x: Any) -> float | None:
        if x is None:
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        return v

    e = fnum(entry)
    sl = fnum(stop)
    tg = fnum(target)
    if e is None or sl is None or tg is None:
        return (
            "unavailable",
            "—",
            "Missing or invalid intraday entry/stop/target.",
            0,
        )

    ltpf = float(ltp)
    # Same resolution order as classify_intraday_vs_ltp: stop/target bands, then entry for Active vs Waiting.
    if idir == "BUY":
        if ltpf <= sl:
            note = f"LTP {ltpf:.4f} ≤ stop {sl:.4f} (target {tg:.4f})."
            return "stop_hit", "Stoploss hit", note, 4
        if ltpf >= tg:
            note = f"LTP {ltpf:.4f} ≥ target {tg:.4f} (entry {e:.4f}, stop {sl:.4f})."
            return "target_hit", "Target hit", note, 3
        # Strictly between stop and target: check entry (touch = at or above entry for long).
        if ltpf >= e:
            note = (
                f"BUY: LTP {ltpf:.4f} ≥ entry {e:.4f} (between stop {sl:.4f} and target {tg:.4f})."
            )
            return "active", "Active", note, 2
        note = (
            f"BUY: LTP {ltpf:.4f} still below entry {e:.4f}. "
            f"Active when LTP reaches entry or higher (until target {tg:.4f} or stop {sl:.4f})."
        )
        return "waiting_entry", "Waiting for entry", note, 1

    # SELL: whole band (target < LTP < stop) is the live short zone — no entry gate (unlike long).
    if ltpf >= sl:
        note = f"LTP {ltpf:.4f} ≥ stop {sl:.4f} (target {tg:.4f})."
        return "stop_hit", "Stoploss hit", note, 4
    if ltpf <= tg:
        note = f"LTP {ltpf:.4f} ≤ target {tg:.4f} (entry {e:.4f}, stop {sl:.4f})."
        return "target_hit", "Target hit", note, 3
    note = (
        f"SELL: LTP {ltpf:.4f} between target {tg:.4f} and stop {sl:.4f} "
        f"(entry reference {e:.4f})."
    )
    return "active", "Active", note, 2

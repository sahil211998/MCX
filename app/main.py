"""
MCX Smart Signal — FastAPI backend (daily MCX FUTCOM data + confluence signals).

Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections import deque
from contextlib import asynccontextmanager
from functools import partial
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Header, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcx_insight.config import SIGNAL_ONLY_MCX_PRODUCTS  # noqa: E402
from mcx_insight.config import SIGNAL_ONLY_MCX_SET  # noqa: E402
from mcx_insight.catalog import list_mcx_futcom_commodities  # noqa: E402
from mcx_insight.db import (  # noqa: E402
    connect_pg,
    ensure_schema,
    fetch_accuracy_summary,
    fetch_latest_signals,
    insert_dual_horizon_signal,
    refresh_smart_signal_outcomes,
    truncate_all_mcx_app_data,
)
from mcx_insight.dual_horizon import run_dual_analysis  # noqa: E402
from mcx_insight.mcx_data import live_quote  # noqa: E402
from mcx_insight.outcome_eval import intraday_live_status  # noqa: E402

logger = logging.getLogger(__name__)
STATIC_DIR = ROOT / "static"

_auto_run_history: deque[dict[str, Any]] = deque(maxlen=30)
_auto_worker_lock = asyncio.Lock()


def _generate_all_six(conn, days: int, pause: float) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    for mcx_product in SIGNAL_ONLY_MCX_PRODUCTS:
        dual = run_dual_analysis(
            mcx_product,
            calendar_days=min(max(40, days), 240),
            bhav_pause=max(0.0, pause),
        )
        new_id = insert_dual_horizon_signal(conn, dual.to_db_row(), disclaimer=dual.disclaimer)
        generated.append(
            {
                "id": new_id,
                "symbol_key": dual.symbol_key,
                "mcx_product": dual.mcx_product,
                "direction": dual.daily.direction,
                "intraday_direction": dual.intraday.direction,
                "confidence_pct": max(dual.daily.confidence_pct, dual.intraday.confidence_pct),
                "risk_reward": dual.long_term_risk_reward or dual.daily.risk_reward,
            }
        )
    return generated


def _auto_signal_enabled() -> bool:
    return os.environ.get("AUTO_SIGNAL_ENABLED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _auto_interval_secs() -> int:
    return max(60, int(os.environ.get("AUTO_SIGNAL_INTERVAL_SECS", "120")))


def _outcome_scheduler_enabled() -> bool:
    return os.environ.get("INTRADAY_OUTCOME_SCHEDULER_ENABLED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _outcome_schedule_config() -> tuple[str, int, int]:
    """Timezone (IANA) and local clock time for daily intraday outcome scoring."""
    tz = (os.environ.get("INTRADAY_OUTCOME_TZ", "Asia/Kolkata") or "Asia/Kolkata").strip()
    try:
        h = int(os.environ.get("INTRADAY_OUTCOME_LOCAL_HOUR", "23"))
        m = int(os.environ.get("INTRADAY_OUTCOME_LOCAL_MINUTE", "25"))
    except ValueError:
        h, m = 23, 25
    return tz, max(0, min(23, h)), max(0, min(59, m))


def next_intraday_outcome_run_utc() -> datetime:
    """Next wall-clock fire time in the configured zone, as UTC aware datetime."""
    tz_name, hour, minute = _outcome_schedule_config()
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_local >= target:
        target = target + timedelta(days=1)
    return target.astimezone(timezone.utc)


def _sync_refresh_intraday_outcomes() -> dict[str, Any]:
    """Blocking DB + LTP refresh; used by the nightly scheduler and optional manual POST."""
    conn = connect_pg()
    try:
        ensure_schema(conn)
        return refresh_smart_signal_outcomes(
            conn,
            limit=800,
            products=tuple(SIGNAL_ONLY_MCX_PRODUCTS),
        )
    finally:
        conn.close()


async def _intraday_outcome_scheduler_worker() -> None:
    """Score intraday target/stop vs MCX LTP once per day at the configured local time (default 23:25 IST)."""
    while True:
        try:
            next_utc = next_intraday_outcome_run_utc()
            delay = (next_utc - datetime.now(timezone.utc)).total_seconds()
            await asyncio.sleep(max(1.0, delay))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("intraday outcome scheduler: sleep scheduling failed")
            await asyncio.sleep(300)
            continue
        if not _outcome_scheduler_enabled():
            await asyncio.sleep(60)
            continue
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _sync_refresh_intraday_outcomes)
            logger.info("intraday outcome scheduler: refresh completed for scheduled run")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("intraday outcome scheduler: refresh failed")
        await asyncio.sleep(90)


def _actionable_intraday_count(signals: list[dict[str, Any]]) -> int:
    n = 0
    for s in signals:
        d = s.get("intraday_direction")
        if d and d != "NO_TRADE":
            n += 1
    return n


def _actionable_intraday_rows(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for s in signals:
        d = s.get("intraday_direction")
        if d in ("BUY", "SELL"):
            out.append(dict(s))
    return out


def _direction_changes_vs_previous(
    current: list[dict[str, Any]],
    previous: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    Rows where intraday or daily direction string changed vs last successful auto batch.
    Empty when previous is None (first run after startup — no baseline to diff).
    """
    if not previous:
        return []
    prev_by = {str(s["mcx_product"]): s for s in previous}
    out: list[dict[str, Any]] = []
    for s in current:
        key = str(s["mcx_product"])
        old = prev_by.get(key)
        if not old:
            continue
        ni, nd = s.get("intraday_direction"), s.get("direction")
        oi, od = old.get("intraday_direction"), old.get("direction")
        if ni != oi or nd != od:
            row = dict(s)
            row["prior_intraday_direction"] = oi
            row["prior_daily_direction"] = od
            out.append(row)
    return out


def _sync_auto_generate_batch(previous_signals: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Blocking: full six-product insert; used from background worker thread."""
    at = datetime.now(timezone.utc).isoformat()
    conn = None
    try:
        conn = connect_pg()
    except Exception as e:  # pragma: no cover
        return {
            "ok": False,
            "at": at,
            "error": str(e),
            "signals": [],
            "actionable_intraday": 0,
            "direction_changes_vs_previous": [],
            "direction_change_count": 0,
            "actionable_intraday_list": [],
            "has_previous_cycle": False,
        }
    try:
        ensure_schema(conn)
        generated = _generate_all_six(conn, days=90, pause=0.04)
        ac = _actionable_intraday_count(generated)
        changes = _direction_changes_vs_previous(generated, previous_signals)
        actionable_list = _actionable_intraday_rows(generated)
        return {
            "ok": True,
            "at": at,
            "generated_count": len(generated),
            "actionable_intraday": ac,
            "has_previous_cycle": previous_signals is not None,
            "direction_changes_vs_previous": changes,
            "direction_change_count": len(changes),
            "actionable_intraday_list": actionable_list,
            "signals": generated,
        }
    except Exception as e:
        logger.exception("auto_signal batch failed")
        return {
            "ok": False,
            "at": at,
            "error": str(e),
            "signals": [],
            "actionable_intraday": 0,
            "direction_changes_vs_previous": [],
            "direction_change_count": 0,
            "actionable_intraday_list": [],
            "has_previous_cycle": False,
        }
    finally:
        if conn is not None:
            conn.close()


async def _auto_signal_worker() -> None:
    await asyncio.sleep(8)
    interval = _auto_interval_secs()
    while True:
        async with _auto_worker_lock:
            prev_signals: list[dict[str, Any]] | None = None
            for entry in reversed(_auto_run_history):
                if entry.get("ok") and entry.get("signals"):
                    prev_signals = list(entry["signals"])
                    break
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                partial(_sync_auto_generate_batch, prev_signals),
            )
            _auto_run_history.append(result)
            if result.get("ok"):
                logger.info(
                    "auto_signal: stored %s rows (%s actionable intraday, %s direction change(s) vs prior)",
                    result.get("generated_count"),
                    result.get("actionable_intraday"),
                    result.get("direction_change_count"),
                )
            else:
                logger.warning("auto_signal failed: %s", result.get("error"))
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task[Any]] = []
    if _auto_signal_enabled():
        tasks.append(asyncio.create_task(_auto_signal_worker()))
    if _outcome_scheduler_enabled():
        tasks.append(asyncio.create_task(_intraday_outcome_scheduler_worker()))
    yield
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="MCX Smart Signal", version="0.1.0", lifespan=lifespan)

if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


def _serialize_signal(row: dict) -> dict[str, Any]:
    r = dict(row)
    for k in ("created_at", "call_generated_at", "outcome_evaluated_at"):
        ts = r.get(k)
        if isinstance(ts, datetime):
            r[k] = ts.isoformat()
    return r


def _enrich_signals_live_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach MCX watch LTP and live intraday phase (active / waiting / stop / target) per row."""
    products = {str(r.get("mcx_product") or "").strip().upper() for r in rows if r.get("mcx_product")}
    quotes: dict[str, Any] = {}
    for pid in products:
        try:
            quotes[pid] = live_quote(pid)
        except Exception:
            quotes[pid] = None

    out: list[dict[str, Any]] = []
    for r in rows:
        s = _serialize_signal(r)
        pid = str(s.get("mcx_product") or "").strip().upper()
        q = quotes.get(pid)
        ltp: float | None = None
        if q is not None and getattr(q, "ltp", None) is not None:
            try:
                v = float(q.ltp)
                ltp = v if math.isfinite(v) else None
            except (TypeError, ValueError):
                ltp = None
        phase, label, detail, stage = intraday_live_status(
            intraday_direction=s.get("intraday_direction"),
            entry=s.get("intraday_entry"),
            stop=s.get("intraday_stop"),
            target=s.get("intraday_target"),
            ltp=ltp,
        )
        s["intraday_live_ltp"] = ltp
        s["intraday_live_phase"] = phase
        s["intraday_status_label"] = label
        s["intraday_status_detail"] = detail
        s["intraday_stage"] = stage
        out.append(s)
    return out


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "product": "MCX Smart Signal"}


@app.get("/api/commodities")
def api_commodities() -> list[dict[str, Any]]:
    """Allowed mini / gas FUTCOM roots only (ALUMINI, ZINCMINI, SILVERM, CRUDEOILM, NATURALGAS, NATGASMINI)."""
    try:
        return list_mcx_futcom_commodities()
    except Exception as e:
        raise HTTPException(502, f"MCX catalog unavailable: {e}") from e


@app.post("/api/commodities/{mcx_product}/analyze")
def api_analyze_commodity(
    mcx_product: str,
    store: bool = True,
    days: int = 100,
    pause: float = 0.04,
) -> dict[str, Any]:
    """Load full daily MCX history + intraday proxy; intraday & swing targets; optional DB store."""
    prod = mcx_product.strip().upper()
    if not prod or len(prod) > 32:
        raise HTTPException(400, "Invalid product code")
    if prod not in SIGNAL_ONLY_MCX_SET:
        raise HTTPException(
            400,
            "Only enabled contracts: ALUMINI, ZINCMINI, SILVERM, CRUDEOILM, NATURALGAS, NATGASMINI.",
        )
    try:
        dual = run_dual_analysis(prod, calendar_days=min(max(40, days), 240), bhav_pause=max(0.0, pause))
    except Exception as e:
        raise HTTPException(502, f"Analysis failed: {e}") from e
    payload = dual.to_api()
    new_id: int | None = None
    if store:
        try:
            conn = connect_pg()
        except Exception as e:
            raise HTTPException(503, f"Database unavailable: {e}") from e
        try:
            ensure_schema(conn)
            new_id = insert_dual_horizon_signal(conn, dual.to_db_row(), disclaimer=dual.disclaimer)
        finally:
            conn.close()
    return {"ok": True, "stored_id": new_id, "analysis": payload}


@app.get("/")
def homepage() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(404, "static/index.html missing")
    return FileResponse(index)


@app.get("/api/signals")
def api_signals(
    response: Response,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Newest calls first (DB order); `Cache-Control: no-store` avoids stale browser caching."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    try:
        conn = connect_pg()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}") from e
    try:
        ensure_schema(conn)
        rows = fetch_latest_signals(
            conn,
            limit=max(1, min(limit, 200)),
            mcx_products_only=list(SIGNAL_ONLY_MCX_SET),
        )
        return _enrich_signals_live_status(rows)
    finally:
        conn.close()


@app.get("/api/signals/outcome-schedule")
def api_outcome_schedule() -> dict[str, Any]:
    """When intraday outcomes are scored (default: daily 23:25 in Asia/Kolkata)."""
    tz_name, h, m = _outcome_schedule_config()
    try:
        nxt = next_intraday_outcome_run_utc()
    except Exception as e:
        raise HTTPException(500, f"Invalid outcome schedule: {e}") from e
    return {
        "timezone": tz_name,
        "local_hour": h,
        "local_minute": m,
        "local_time_label": f"{h:02d}:{m:02d}",
        "next_run_utc": nxt.isoformat(),
        "scheduler_enabled": _outcome_scheduler_enabled(),
    }


@app.post("/api/signals/evaluate-outcomes")
def api_evaluate_outcomes(limit: int = 600) -> dict[str, Any]:
    """Manual re-score: intraday target/stop vs MCX watch LTP. Normally runs on the daily schedule only."""
    try:
        conn = connect_pg()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}") from e
    try:
        ensure_schema(conn)
        stats = refresh_smart_signal_outcomes(
            conn,
            limit=max(50, min(limit, 2000)),
            products=tuple(SIGNAL_ONLY_MCX_PRODUCTS),
        )
        summary = fetch_accuracy_summary(conn, products=tuple(SIGNAL_ONLY_MCX_PRODUCTS))
        return {"ok": True, "refresh": stats, "accuracy": summary}
    finally:
        conn.close()


@app.get("/api/signals/accuracy")
def api_accuracy() -> dict[str, Any]:
    """Rolling accuracy from stored outcomes (see methodology_note)."""
    try:
        conn = connect_pg()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}") from e
    try:
        ensure_schema(conn)
        return fetch_accuracy_summary(conn, products=tuple(SIGNAL_ONLY_MCX_PRODUCTS))
    finally:
        conn.close()


@app.get("/api/signals/auto-status")
def api_auto_status() -> dict[str, Any]:
    """Background worker: batch every `AUTO_SIGNAL_INTERVAL_SECS` (default 120). Lists recent runs + signals stored."""
    runs = list(_auto_run_history)[::-1]
    return {
        "auto_enabled": _auto_signal_enabled(),
        "interval_secs": _auto_interval_secs(),
        "last_run": runs[0] if runs else None,
        "recent_runs": runs[:15],
    }


@app.post("/api/signals/generate")
def api_generate(days: int = 80, pause: float = 0.04) -> dict[str, Any]:
    """Batch dual-horizon signals for the six allowed mini/gas contracts only."""
    try:
        conn = connect_pg()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}") from e
    try:
        ensure_schema(conn)
        generated = _generate_all_six(conn, days, pause)
        return {"ok": True, "signals": generated}
    finally:
        conn.close()


@app.post("/api/signals/reset-and-generate")
def api_reset_and_generate(
    days: int = 80,
    pause: float = 0.04,
    x_reset_key: str | None = Header(default=None, alias="X-Reset-Key"),
    reset_key: str | None = Query(
        default=None,
        description="Same value as X-Reset-Key when MCX_SIGNAL_RESET_KEY is set (for simple clients).",
    ),
) -> dict[str, Any]:
    """
    Truncate all `mcx_smart_signals` and `mcx_trade_levels` rows, then run the same batch as `/api/signals/generate`.
    If env `MCX_SIGNAL_RESET_KEY` is set, send it as header `X-Reset-Key` or query `reset_key`.
    """
    expected = os.environ.get("MCX_SIGNAL_RESET_KEY")
    provided = (x_reset_key or reset_key or "").strip()
    if expected:
        if provided != expected:
            raise HTTPException(
                403,
                "Invalid or missing reset credential (header X-Reset-Key or query reset_key).",
            )
    try:
        conn = connect_pg()
    except Exception as e:
        raise HTTPException(503, f"Database unavailable: {e}") from e
    try:
        ensure_schema(conn)
        truncate_all_mcx_app_data(conn)
        generated = _generate_all_six(conn, days, pause)
        return {
            "ok": True,
            "truncated": True,
            "regenerated": len(generated),
            "signals": generated,
        }
    finally:
        conn.close()

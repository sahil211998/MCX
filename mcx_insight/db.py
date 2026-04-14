from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

if TYPE_CHECKING:
    from mcx_insight.strategy import TradeLevels

from mcx_insight.config import SIGNAL_ONLY_MCX_PRODUCTS
from mcx_insight.mcx_data import LiveQuote, live_quote
from mcx_insight.outcome_eval import classify_intraday_vs_ltp

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None

# Serialize DDL so parallel /api/* calls do not deadlock on ALTER TABLE (AccessExclusiveLock).
_SCHEMA_ADV_LOCK_KEY1 = 0x4D435800  # "MCX" — arbitrary app id
_SCHEMA_ADV_LOCK_KEY2 = 1


@dataclass(frozen=True)
class PgConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


def pg_config_from_env() -> PgConfig:
    return PgConfig(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "postgres"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def connect_pg(cfg: Optional[PgConfig] = None):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. pip install psycopg2-binary")
    c = cfg or pg_config_from_env()
    try:
        connect_timeout = int(os.environ.get("PGCONNECT_TIMEOUT", "5"))
    except ValueError:
        connect_timeout = 5
    return psycopg2.connect(
        host=c.host,
        port=c.port,
        dbname=c.dbname,
        user=c.user,
        password=c.password,
        connect_timeout=max(2, min(connect_timeout, 30)),
    )


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_lock(%s, %s)",
            (_SCHEMA_ADV_LOCK_KEY1, _SCHEMA_ADV_LOCK_KEY2),
        )
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS mcx_trade_levels (
                    id              BIGSERIAL PRIMARY KEY,
                    symbol_key      TEXT NOT NULL,
                    mcx_product     TEXT NOT NULL,
                    as_of           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    bias            TEXT NOT NULL,
                    entry_price     DOUBLE PRECISION,
                    stop_loss       DOUBLE PRECISION,
                    target_price    DOUBLE PRECISION,
                    atr             DOUBLE PRECISION,
                    rationale       TEXT,
                    live_ltp        DOUBLE PRECISION,
                    live_expiry     TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_mcx_trade_levels_product_asof
                    ON mcx_trade_levels (mcx_product, as_of DESC);

                CREATE TABLE IF NOT EXISTS mcx_smart_signals (
                    id               BIGSERIAL PRIMARY KEY,
                    symbol_key       TEXT NOT NULL,
                    mcx_product      TEXT NOT NULL,
                    timeframe        TEXT NOT NULL DEFAULT '1d',
                    direction        TEXT NOT NULL,
                    entry_price      DOUBLE PRECISION,
                    target_price     DOUBLE PRECISION,
                    stop_loss        DOUBLE PRECISION,
                    risk_reward      DOUBLE PRECISION,
                    confidence_pct   INT NOT NULL DEFAULT 0,
                    trend            TEXT,
                    pattern_summary  TEXT,
                    indicators_json  JSONB,
                    rationale        TEXT,
                    disclaimer       TEXT,
                    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    outcome          TEXT DEFAULT 'pending'
                );
                CREATE INDEX IF NOT EXISTS ix_mcx_smart_signals_created
                    ON mcx_smart_signals (created_at DESC);
                CREATE INDEX IF NOT EXISTS ix_mcx_smart_signals_product
                    ON mcx_smart_signals (mcx_product, created_at DESC);
                """
            )
            _migrate_smart_signals_dual_columns(cur)
        finally:
            cur.execute(
                "SELECT pg_advisory_unlock(%s, %s)",
                (_SCHEMA_ADV_LOCK_KEY1, _SCHEMA_ADV_LOCK_KEY2),
            )
    conn.commit()


def _migrate_smart_signals_dual_columns(cur) -> None:
    """Add dual-horizon + call timestamp columns (idempotent)."""
    stmts = [
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS call_generated_at TIMESTAMPTZ",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_interval TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_direction TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_entry DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_stop DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_target DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_risk_reward DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_confidence_pct INT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_interval TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_direction TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_entry DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_stop DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_target DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_risk_reward DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS long_term_confidence_pct INT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS data_notes TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS confidence_explanation TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS call_scope TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS call_scope_label TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS outcome_eval_note TEXT",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS outcome_evaluated_at TIMESTAMPTZ",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS outcome_eval_ltp DOUBLE PRECISION",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_activated_at TIMESTAMPTZ",
        "ALTER TABLE mcx_smart_signals ADD COLUMN IF NOT EXISTS intraday_activated_ltp DOUBLE PRECISION",
    ]
    for sql in stmts:
        cur.execute(sql)


def _parse_ts(val: Any):
    from datetime import datetime

    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        s = val.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def insert_dual_horizon_signal(conn, row: dict, disclaimer: str | None = None) -> int:
    """Insert row with intraday + swing targets and explicit call time."""
    disc = disclaimer or row.get("disclaimer") or (
        "Probability-based analytics only. Not financial advice."
    )
    ind = row.get("indicators_json")
    if ind is None:
        ind_js = "{}"
    elif isinstance(ind, str):
        ind_js = ind
    else:
        ind_js = json.dumps(_json_scrub(ind), allow_nan=False)

    cg = _parse_ts(row.get("call_generated_at")) or _parse_ts(row.get("created_at"))

    payload = {
        "symbol_key": row["symbol_key"],
        "mcx_product": row["mcx_product"],
        "timeframe": row.get("timeframe") or "1d",
        "direction": row["direction"],
        "entry_price": row.get("entry_price"),
        "target_price": row.get("target_price"),
        "stop_loss": row.get("stop_loss"),
        "risk_reward": row.get("risk_reward"),
        "confidence_pct": int(row.get("confidence_pct") or 0),
        "trend": row.get("trend"),
        "pattern_summary": row.get("pattern_summary"),
        "rationale": row.get("rationale"),
        "disclaimer": disc,
        "ind_js": ind_js,
        "call_generated_at": cg,
        "intraday_interval": row.get("intraday_interval"),
        "intraday_direction": row.get("intraday_direction"),
        "intraday_entry": row.get("intraday_entry"),
        "intraday_stop": row.get("intraday_stop"),
        "intraday_target": row.get("intraday_target"),
        "intraday_risk_reward": row.get("intraday_risk_reward"),
        "intraday_confidence_pct": row.get("intraday_confidence_pct"),
        "long_term_interval": row.get("long_term_interval"),
        "long_term_direction": row.get("long_term_direction"),
        "long_term_entry": row.get("long_term_entry"),
        "long_term_stop": row.get("long_term_stop"),
        "long_term_target": row.get("long_term_target"),
        "long_term_risk_reward": row.get("long_term_risk_reward"),
        "long_term_confidence_pct": row.get("long_term_confidence_pct"),
        "data_notes": row.get("data_notes"),
        "confidence_explanation": row.get("confidence_explanation"),
        "call_scope": row.get("call_scope"),
        "call_scope_label": row.get("call_scope_label"),
    }
    with conn.cursor() as cur:
        # Serialize inserts with schema DDL across all processes/instances.
        # ensure_schema() uses the same advisory keys.
        cur.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            (_SCHEMA_ADV_LOCK_KEY1, _SCHEMA_ADV_LOCK_KEY2),
        )
        cur.execute(
            """
            INSERT INTO mcx_smart_signals (
                symbol_key, mcx_product, timeframe, direction,
                entry_price, target_price, stop_loss, risk_reward,
                confidence_pct, trend, pattern_summary, indicators_json,
                rationale, disclaimer, outcome,
                call_generated_at,
                intraday_interval, intraday_direction, intraday_entry, intraday_stop,
                intraday_target, intraday_risk_reward, intraday_confidence_pct,
                long_term_interval, long_term_direction, long_term_entry, long_term_stop,
                long_term_target, long_term_risk_reward, long_term_confidence_pct,
                data_notes, confidence_explanation,
                call_scope, call_scope_label
            ) VALUES (
                %(symbol_key)s, %(mcx_product)s, %(timeframe)s, %(direction)s,
                %(entry_price)s, %(target_price)s, %(stop_loss)s, %(risk_reward)s,
                %(confidence_pct)s, %(trend)s, %(pattern_summary)s,
                CAST(%(ind_js)s AS jsonb),
                %(rationale)s, %(disclaimer)s, 'pending',
                COALESCE(%(call_generated_at)s, NOW()),
                %(intraday_interval)s, %(intraday_direction)s, %(intraday_entry)s, %(intraday_stop)s,
                %(intraday_target)s, %(intraday_risk_reward)s, %(intraday_confidence_pct)s,
                %(long_term_interval)s, %(long_term_direction)s, %(long_term_entry)s, %(long_term_stop)s,
                %(long_term_target)s, %(long_term_risk_reward)s, %(long_term_confidence_pct)s,
                %(data_notes)s, %(confidence_explanation)s,
                %(call_scope)s, %(call_scope_label)s
            ) RETURNING id
            """,
            payload,
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return int(new_id)


def insert_trade_levels(
    conn,
    *,
    symbol_key: str,
    mcx_product: str,
    levels: Optional["TradeLevels"],
    live_quote: Optional["LiveQuote"] = None,
) -> None:
    """Append one prediction row (research / demo; not financial advice)."""
    if levels is None:
        bias = "neutral"
        entry = stop = target = atr = None
        rationale = "Insufficient MCX daily history for EMA/ATR model."
    else:
        bias = levels.bias
        entry = levels.entry
        stop = levels.stop
        target = levels.target
        atr = levels.atr
        rationale = levels.rationale
    ltp: Any = None
    exp: Any = None
    if live_quote is not None:
        ltp = float(live_quote.ltp)
        exp = live_quote.expiry
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            (_SCHEMA_ADV_LOCK_KEY1, _SCHEMA_ADV_LOCK_KEY2),
        )
        cur.execute(
            """
            INSERT INTO mcx_trade_levels (
                symbol_key, mcx_product, bias,
                entry_price, stop_loss, target_price, atr, rationale,
                live_ltp, live_expiry
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (symbol_key, mcx_product, bias, entry, stop, target, atr, rationale, ltp, exp),
        )
    conn.commit()


def _json_scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_scrub(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def insert_smart_signal(conn, row: dict, disclaimer: str | None = None) -> int:
    """Insert one MCX Smart Signal row; returns new id."""
    disc = disclaimer or row.get("disclaimer") or (
        "Probability-based analytics only. Not financial advice."
    )
    ind = row.get("indicators_json")
    if ind is None:
        ind_js = "{}"
    elif isinstance(ind, str):
        ind_js = ind
    else:
        ind_js = json.dumps(_json_scrub(ind), allow_nan=False)

    payload = {
        "symbol_key": row["symbol_key"],
        "mcx_product": row["mcx_product"],
        "timeframe": row.get("timeframe") or "1d",
        "direction": row["direction"],
        "entry_price": row.get("entry_price"),
        "target_price": row.get("target_price"),
        "stop_loss": row.get("stop_loss"),
        "risk_reward": row.get("risk_reward"),
        "confidence_pct": int(row.get("confidence_pct") or 0),
        "trend": row.get("trend"),
        "pattern_summary": row.get("pattern_summary"),
        "rationale": row.get("rationale"),
        "disclaimer": disc,
        "ind_js": ind_js,
    }
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            (_SCHEMA_ADV_LOCK_KEY1, _SCHEMA_ADV_LOCK_KEY2),
        )
        cur.execute(
            """
            INSERT INTO mcx_smart_signals (
                symbol_key, mcx_product, timeframe, direction,
                entry_price, target_price, stop_loss, risk_reward,
                confidence_pct, trend, pattern_summary, indicators_json,
                rationale, disclaimer, outcome
            ) VALUES (
                %(symbol_key)s, %(mcx_product)s, %(timeframe)s, %(direction)s,
                %(entry_price)s, %(target_price)s, %(stop_loss)s, %(risk_reward)s,
                %(confidence_pct)s, %(trend)s, %(pattern_summary)s,
                CAST(%(ind_js)s AS jsonb),
                %(rationale)s, %(disclaimer)s, 'pending'
            ) RETURNING id
            """,
            payload,
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return int(new_id)


def truncate_all_mcx_app_data(conn) -> None:
    """Delete every stored signal and trade-level row; reset ID sequences (fresh ids after regenerate)."""
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE TABLE mcx_smart_signals, mcx_trade_levels RESTART IDENTITY"
        )
    conn.commit()


def refresh_smart_signal_outcomes(
    conn,
    *,
    limit: int = 500,
    products: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Re-score recent rows: intraday target/stop vs current MCX watch LTP.
    Updates outcome, outcome_eval_note, outcome_evaluated_at, outcome_eval_ltp.
    """
    from datetime import datetime, timezone

    prods = list(products or SIGNAL_ONLY_MCX_PRODUCTS)
    lim = max(50, min(int(limit), 2000))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, mcx_product, intraday_direction, intraday_entry, intraday_stop, intraday_target
            FROM mcx_smart_signals
            WHERE mcx_product = ANY(%s::text[])
            ORDER BY COALESCE(call_generated_at, created_at) DESC, id DESC
            LIMIT %s
            """,
            (prods, lim),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    quotes: dict[str, Any] = {}
    now = datetime.now(timezone.utc)
    stats: dict[str, Any] = {
        "rows_examined": len(rows),
        "rows_updated": 0,
        "by_outcome": {},
    }
    by_out: dict[str, int] = {}

    with conn.cursor() as cur:
        for r in rows:
            pid = str(r["mcx_product"])
            if pid not in quotes:
                quotes[pid] = live_quote(pid)
            q = quotes[pid]
            ltp = float(q.ltp) if q is not None and q.ltp is not None else None
            code, note = classify_intraday_vs_ltp(
                intraday_direction=r.get("intraday_direction"),
                entry=r.get("intraday_entry"),
                stop=r.get("intraday_stop"),
                target=r.get("intraday_target"),
                ltp=ltp,
            )
            by_out[code] = by_out.get(code, 0) + 1
            cur.execute(
                """
                UPDATE mcx_smart_signals
                SET outcome = %s,
                    outcome_eval_note = %s,
                    outcome_evaluated_at = %s,
                    outcome_eval_ltp = %s
                WHERE id = %s
                """,
                (code, note, now, ltp, r["id"]),
            )
            stats["rows_updated"] += 1

    conn.commit()
    stats["by_outcome"] = by_out
    return stats


def fetch_accuracy_summary(
    conn,
    *,
    products: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Aggregate outcomes for intraday BUY/SELL rows; win rate = target_hit / (target_hit + stop_hit)."""
    prods = list(products or SIGNAL_ONLY_MCX_PRODUCTS)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT outcome, COUNT(*)::bigint
            FROM mcx_smart_signals
            WHERE mcx_product = ANY(%s::text[])
              AND COALESCE(UPPER(TRIM(intraday_direction)), 'NO_TRADE') IN ('BUY', 'SELL')
            GROUP BY outcome
            """,
            (prods,),
        )
        rows = {str(r[0] or ""): int(r[1]) for r in cur.fetchall()}

    wins = rows.get("target_hit", 0)
    losses = rows.get("stop_hit", 0)
    openish = rows.get("open", 0) + rows.get("pending", 0)
    denom = wins + losses
    win_rate = round(100.0 * wins / denom, 2) if denom > 0 else None
    return {
        "scoped_products": prods,
        "target_hits": wins,
        "stop_hits": losses,
        "open_or_pending": openish,
        "no_trade_scored": rows.get("no_trade", 0),
        "unavailable": rows.get("unavailable", 0),
        "win_rate_pct": win_rate,
        "resolved_trades": denom,
        "totals_by_outcome": rows,
        "methodology_note": (
            "Scores use latest MCX FUTCOM LTP vs stored intraday target/stop (snapshot). "
            "Only intraday BUY/SELL rows are included in hit-rate denominators. "
            "Win rate = target_hit ÷ (target_hit + stop_hit) among resolved LTP checks — "
            "not proof target filled before stop. Research only."
        ),
    }


def count_signals_for_product_calendar_day(
    conn,
    *,
    mcx_product: str,
    tz_name: str,
) -> int:
    """
    Rows whose call time falls on the current calendar date in tz_name (e.g. Asia/Kolkata).
    Used to cap automatic re-generation so published calls do not churn every batch.
    """
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    day_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    next_day_local = day_start_local + timedelta(days=1)
    start_utc = day_start_local.astimezone(timezone.utc)
    end_utc = next_day_local.astimezone(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)::bigint FROM mcx_smart_signals
            WHERE mcx_product = %s
              AND COALESCE(call_generated_at, created_at) >= %s
              AND COALESCE(call_generated_at, created_at) < %s
            """,
            (mcx_product, start_utc, end_utc),
        )
        return int(cur.fetchone()[0])


def fetch_latest_signals(
    conn,
    limit: int = 30,
    mcx_products_only: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """
    Newest first: by call time (or insert time), then id so batch inserts stay ordered.
    Optionally restrict to these MCX product codes so LIMIT applies after the filter.
    When restricted, each mcx_product appears at most once (latest stored call only).
    """
    limit = max(1, min(int(limit), 500))
    sel = (
        "id, symbol_key, mcx_product, timeframe, direction, "
        "entry_price, target_price, stop_loss, risk_reward, "
        "confidence_pct, trend, pattern_summary, indicators_json, "
        "rationale, created_at, outcome, "
        "outcome_eval_note, outcome_evaluated_at, outcome_eval_ltp, "
        "intraday_activated_at, intraday_activated_ltp, "
        "call_generated_at, intraday_interval, intraday_direction, "
        "intraday_entry, intraday_stop, intraday_target, "
        "intraday_risk_reward, intraday_confidence_pct, "
        "long_term_interval, long_term_direction, "
        "long_term_entry, long_term_stop, long_term_target, "
        "long_term_risk_reward, long_term_confidence_pct, data_notes, "
        "confidence_explanation, call_scope, call_scope_label"
    )
    order_newest = "COALESCE(call_generated_at, created_at) DESC, id DESC"
    with conn.cursor() as cur:
        if mcx_products_only:
            cur.execute(
                "SELECT "
                + sel
                + " FROM ( SELECT DISTINCT ON (mcx_product) "
                + sel
                + " FROM mcx_smart_signals WHERE mcx_product = ANY(%s::text[]) "
                + "ORDER BY mcx_product, "
                + order_newest
                + " ) AS latest_per_product ORDER BY "
                + order_newest
                + " LIMIT %s",
                (list(mcx_products_only), limit),
            )
        else:
            cur.execute(
                "SELECT "
                + sel
                + " FROM mcx_smart_signals ORDER BY "
                + order_newest
                + " LIMIT %s",
                (limit,),
            )
        cols = [d[0] for d in cur.description]
        out: list[dict] = []
        for r in cur.fetchall():
            row = dict(zip(cols, r))
            ij = row.get("indicators_json")
            if ij is not None and not isinstance(ij, dict) and isinstance(ij, str):
                try:
                    row["indicators_json"] = json.loads(ij)
                except json.JSONDecodeError:
                    pass
            out.append(row)
        return out


def mark_intraday_activated(conn, *, signal_id: int, ltp: float | None) -> None:
    """Persist activation latch so BUY doesn't flip back to waiting."""
    from datetime import datetime, timezone

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE mcx_smart_signals
            SET intraday_activated_at = COALESCE(intraday_activated_at, %s),
                intraday_activated_ltp = COALESCE(intraday_activated_ltp, %s)
            WHERE id = %s
            """,
            (datetime.now(timezone.utc), float(ltp) if ltp is not None else None, int(signal_id)),
        )
    conn.commit()

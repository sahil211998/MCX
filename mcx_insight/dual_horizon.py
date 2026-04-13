from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# INR levels stored with this precision — frozen at call generation (no later DB updates).
_FROZEN_INR_DECIMALS = 2
_FROZEN_RR_DECIMALS = 4


def _freeze_inr_price(x: Any) -> float | None:
    """Snapshot one MCX INR price for persistence (avoids float noise from scaling)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, _FROZEN_INR_DECIMALS)


def _freeze_rr(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, _FROZEN_RR_DECIMALS)

from mcx_insight import config
from mcx_insight.catalog import yahoo_intraday_ticker
from mcx_insight.institutional_context import build_institutional_context
from mcx_insight.mcx_data import build_daily_ohlcv_mcx, live_quote, merge_live_into_ohlcv
from mcx_insight.prices import fetch_ohlcv
from mcx_insight.signal_engine import SmartSignal, _compute_rr, generate_smart_signal


def _symbol_key_from_product(mcx_product: str) -> str:
    return mcx_product.strip().lower()


def _target_from_entry_pct(sig: SmartSignal, pct: float) -> tuple[float | None, float | None]:
    """Favorable move from entry by pct (e.g. 0.01 = 1%). Same stop as signal; R:R derived from that geometry."""
    if sig.direction == "NO_TRADE" or sig.entry is None or sig.stop_loss is None:
        return None, None
    entry = float(sig.entry)
    stop = float(sig.stop_loss)
    p = float(pct)
    if p <= 0:
        return None, None
    if sig.direction == "BUY":
        tgt = entry * (1.0 + p)
    else:
        tgt = entry * (1.0 - p)
    rr = _compute_rr(entry, stop, tgt, sig.direction)
    return tgt, rr


def _refine_intraday_levels(sig: SmartSignal) -> None:
    """Replace wide model stop with fixed 1% from entry, then 1% target (see config)."""
    if sig.direction == "NO_TRADE" or sig.entry is None:
        return
    entry = float(sig.entry)
    sp = float(config.INTRADAY_STOP_PCT)
    if sp <= 0 or not math.isfinite(sp):
        return
    if sig.direction == "BUY":
        sig.stop_loss = entry * (1.0 - sp)
    else:
        sig.stop_loss = entry * (1.0 + sp)
    tgt, rr = _target_from_entry_pct(sig, float(config.INTRADAY_TARGET_PCT))
    if tgt is None:
        return
    sig.target = tgt
    sig.risk_reward = rr


def _mcx_reference_price(daily_merged: pd.DataFrame, q: Any) -> float | None:
    """Prefer live MCX LTP; else last MCX daily close (INR contract price)."""
    if q is not None:
        try:
            v = float(q.ltp)
        except (TypeError, ValueError):
            v = float("nan")
        if math.isfinite(v) and v > 0:
            return v
    if daily_merged.empty or "close" not in daily_merged.columns:
        return None
    try:
        v = float(daily_merged["close"].iloc[-1])
    except (TypeError, ValueError):
        return None
    if math.isfinite(v) and v > 0:
        return v
    return None


def _yahoo_proxy_to_mcx_scale(intra_df: pd.DataFrame, mcx_ref: float | None) -> float | None:
    """Map Yahoo proxy OHLC to MCX INR: multiply by (mcx_ref / proxy last close)."""
    if mcx_ref is None or intra_df.empty or "close" not in intra_df.columns:
        return None
    try:
        p = float(intra_df["close"].iloc[-1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p <= 0 or not math.isfinite(mcx_ref) or mcx_ref <= 0:
        return None
    return float(mcx_ref) / p


def _scale_ohlcv_df(df: pd.DataFrame, mult: float) -> pd.DataFrame:
    """Apply one INR scale factor to OHLC columns (same mapping as intraday signal scaling)."""
    if df.empty or not math.isfinite(mult) or mult <= 0:
        return df.copy()
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * mult
    return out


def _scale_signal_prices_to_mcx_inr(sig: SmartSignal, mult: float) -> None:
    """Apply one multiplier to entry/stop/target (R:R unchanged)."""
    if not math.isfinite(mult) or mult <= 0 or abs(mult - 1.0) < 1e-15:
        return
    for name in ("entry", "stop_loss", "target"):
        v = getattr(sig, name)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            setattr(sig, name, x * mult)


@dataclass
class DualHorizonResult:
    mcx_product: str
    symbol_key: str
    call_generated_at: str
    call_generated_at_ist_note: str
    live_ltp: float | None
    live_expiry: str | None
    daily_bars_used: int
    intraday_bars_used: int
    intraday_interval: str
    intraday_data_note: str
    daily: SmartSignal
    intraday: SmartSignal
    long_term_target_price: float | None
    long_term_risk_reward: float | None
    long_term_bars_used: int = 0
    institutional_context: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Probability-based analytics only. Not financial advice. "
        "Intraday structure uses a Yahoo international proxy; entry/stop/target are scaled to MCX Indian rupee prices for execution reference."
    )

    def confidence_blended_pct(self) -> int:
        return config.blend_confidence_pct(
            self.daily.confidence_pct, self.intraday.confidence_pct
        )

    def to_api(self) -> dict[str, Any]:
        blended = self.confidence_blended_pct()
        return {
            "mcx_product": self.mcx_product,
            "symbol_key": self.symbol_key,
            "call_generated_at": self.call_generated_at,
            "call_generated_at_note_ist": self.call_generated_at_ist_note,
            "live_ltp": self.live_ltp,
            "live_expiry": self.live_expiry,
            "daily_bars_used": self.daily_bars_used,
            "long_term_bars_used": self.long_term_bars_used,
            "intraday_bars_used": self.intraday_bars_used,
            "intraday_interval": self.intraday_interval,
            "intraday_data_note": self.intraday_data_note,
            "confidence_scores": {
                "blended_pct": blended,
                "daily_mcx_pct": self.daily.confidence_pct,
                "intraday_proxy_pct": self.intraday.confidence_pct,
                "weights": f"{int(config.CONFIDENCE_WEIGHT_DAILY * 100)}% daily / "
                f"{int(config.CONFIDENCE_WEIGHT_INTRADAY * 100)}% intraday",
                "explanation": config.CONFIDENCE_SCORE_EXPLANATION,
            },
            "intraday": _sig_dict(self.intraday, "intraday"),
            "long_term": _sig_dict(
                self.daily,
                "long_term",
                extra_target=self.long_term_target_price,
                extra_rr=self.long_term_risk_reward,
            ),
            "disclaimer": self.disclaimer,
            "call_scope": config.PRIMARY_CALL_SCOPE,
            "call_scope_label": config.CALL_SCOPE_LABEL,
            "institutional_context": self.institutional_context,
        }

    def to_db_row(self) -> dict[str, Any]:
        blended = self.confidence_blended_pct()
        ind = {
            "daily": self.daily.indicators,
            "intraday": self.intraday.indicators,
            "intraday_stop_pct": config.INTRADAY_STOP_PCT,
            "intraday_target_pct": config.INTRADAY_TARGET_PCT,
            "long_term_target_pct": config.LONG_TERM_TARGET_PCT,
            "institutional_context": self.institutional_context,
            "confidence_scores": {
                "blended_pct": blended,
                "daily_mcx_pct": self.daily.confidence_pct,
                "intraday_proxy_pct": self.intraday.confidence_pct,
                "weights": f"{int(config.CONFIDENCE_WEIGHT_DAILY * 100)}% daily / "
                f"{int(config.CONFIDENCE_WEIGHT_INTRADAY * 100)}% intraday",
                "explanation": config.CONFIDENCE_SCORE_EXPLANATION,
            },
        }
        primary_dir = (
            self.daily.direction
            if self.daily.direction != "NO_TRADE"
            else self.intraday.direction
        )
        primary_entry = (
            self.daily.entry if self.daily.entry is not None else self.intraday.entry
        )
        primary_stop = self.daily.stop_loss
        primary_tgt = self.long_term_target_price or self.daily.target
        primary_rr = self.long_term_risk_reward or self.daily.risk_reward
        pat = (
            f"Long-term ({self.daily.timeframe} Yahoo→MCX): {self.daily.pattern_summary} "
            f"| Intraday ({self.intraday_interval}): {self.intraday.pattern_summary}"
        )
        # Intraday/daily prices are frozen at this insert — rows are never updated later for entry/stop/target.
        return {
            "symbol_key": self.symbol_key,
            "mcx_product": self.mcx_product,
            "timeframe": f"dual:{self.daily.timeframe}+{self.intraday_interval}",
            "direction": primary_dir if primary_dir else "NO_TRADE",
            "entry_price": _freeze_inr_price(primary_entry),
            "target_price": _freeze_inr_price(primary_tgt),
            "stop_loss": _freeze_inr_price(primary_stop),
            "risk_reward": _freeze_rr(primary_rr),
            "confidence_pct": blended,
            "confidence_explanation": config.CONFIDENCE_SCORE_EXPLANATION,
            "trend": self.daily.trend,
            "pattern_summary": pat[:2000],
            "indicators_json": ind,
            "rationale": self.daily.rationale or self.intraday.rationale,
            "disclaimer": self.disclaimer,
            "call_generated_at": self.call_generated_at,
            "intraday_interval": self.intraday_interval,
            "intraday_direction": self.intraday.direction,
            "intraday_entry": _freeze_inr_price(self.intraday.entry),
            "intraday_stop": _freeze_inr_price(self.intraday.stop_loss),
            "intraday_target": _freeze_inr_price(self.intraday.target),
            "intraday_risk_reward": _freeze_rr(self.intraday.risk_reward),
            "intraday_confidence_pct": self.intraday.confidence_pct,
            "long_term_interval": self.daily.timeframe,
            "long_term_direction": self.daily.direction,
            "long_term_entry": _freeze_inr_price(self.daily.entry),
            "long_term_stop": _freeze_inr_price(self.daily.stop_loss),
            "long_term_target": _freeze_inr_price(self.long_term_target_price),
            "long_term_risk_reward": _freeze_rr(self.long_term_risk_reward),
            "long_term_confidence_pct": self.daily.confidence_pct,
            "data_notes": self.intraday_data_note,
            "call_scope": config.PRIMARY_CALL_SCOPE,
            "call_scope_label": config.CALL_SCOPE_LABEL,
        }


def _sig_dict(
    sig: SmartSignal,
    kind: str,
    *,
    extra_target: float | None = None,
    extra_rr: float | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "kind": kind,
        "direction": sig.direction,
        "timeframe": sig.timeframe,
        "entry": sig.entry,
        "stop_loss": sig.stop_loss,
        "target": sig.target,
        "risk_reward": sig.risk_reward,
        "confidence_pct": sig.confidence_pct,
        "trend": sig.trend,
        "pattern_summary": sig.pattern_summary,
    }
    if extra_target is not None:
        d["extended_swing_target"] = extra_target
        d["note"] = (
            f"Swing target = {float(config.LONG_TERM_TARGET_PCT) * 100:.0f}% favorable move from long-term entry "
            f"({sig.timeframe} bars); R:R follows entry/stop/target geometry."
        )
    if extra_rr is not None:
        d["extended_swing_rr"] = extra_rr
    if sig.indicators:
        d["features"] = sig.indicators
    return d


def run_dual_analysis(
    mcx_product: str,
    *,
    calendar_days: int = 120,
    bhav_pause: float = 0.04,
) -> DualHorizonResult:
    prod = mcx_product.strip().upper()
    sk = _symbol_key_from_product(prod)
    now = datetime.now(timezone.utc)
    iso = now.isoformat()
    try:
        from zoneinfo import ZoneInfo

        ist = now.astimezone(ZoneInfo("Asia/Kolkata"))
        ist_note = f"IST: {ist.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception:
        ist_note = "IST: (unavailable)"

    daily = build_daily_ohlcv_mcx(
        prod,
        max_calendar_days=min(max(35, calendar_days), 240),
        pause_seconds=bhav_pause,
        skip_weekends=True,
        progress=False,
    )
    q = live_quote(prod)
    daily_merged = merge_live_into_ohlcv(daily, q) if q is not None else daily
    mcx_ref = _mcx_reference_price(daily_merged, q)

    y_sym = yahoo_intraday_ticker(prod)
    lt_df = (
        fetch_ohlcv(
            y_sym,
            period=config.LONG_TERM_YF_PERIOD,
            interval=config.LONG_TERM_YAHOO_INTERVAL,
        )
        if y_sym
        else pd.DataFrame()
    )
    intra_df = (
        fetch_ohlcv(y_sym, period=config.INTRADAY_YF_PERIOD, interval=config.INTRADAY_INTERVAL)
        if y_sym
        else pd.DataFrame()
    )

    long_term_bars_used = len(daily_merged)
    sig_daily = generate_smart_signal(
        daily_merged,
        symbol_key=sk,
        mcx_product=prod,
        timeframe="1d",
        min_rr=float(config.TARGET_RISK_REWARD),
        news_score=0.0,
    )
    if (
        y_sym
        and not lt_df.empty
        and len(lt_df) >= int(config.LONG_TERM_MIN_BARS)
        and mcx_ref is not None
    ):
        scale_lt = _yahoo_proxy_to_mcx_scale(lt_df, mcx_ref)
        if scale_lt is not None:
            lt_mcx = _scale_ohlcv_df(lt_df, scale_lt)
            sig_daily = generate_smart_signal(
                lt_mcx,
                symbol_key=sk,
                mcx_product=prod,
                timeframe=config.LONG_TERM_YAHOO_INTERVAL,
                min_rr=float(config.TARGET_RISK_REWARD),
                news_score=0.0,
            )
            long_term_bars_used = len(lt_df)
    lt_tgt, lt_rr = _target_from_entry_pct(sig_daily, float(config.LONG_TERM_TARGET_PCT))

    scale_for_ctx: float | None = None
    if intra_df.empty or len(intra_df) < 30:
        sig_intra = SmartSignal(
            symbol_key=sk,
            mcx_product=prod,
            direction="NO_TRADE",
            timeframe=config.INTRADAY_INTERVAL,
            entry=None,
            target=None,
            stop_loss=None,
            risk_reward=None,
            confidence_pct=0,
            trend="Sideways",
            pattern_summary="No intraday proxy data",
            indicators={"proxy": y_sym, "bars": len(intra_df)},
            rationale="Yahoo intraday empty or too short.",
        )
    else:
        sig_intra = generate_smart_signal(
            intra_df,
            symbol_key=sk,
            mcx_product=prod,
            timeframe=config.INTRADAY_INTERVAL,
            min_rr=float(config.INTRADAY_MIN_RR),
            news_score=0.0,
        )
        _refine_intraday_levels(sig_intra)
        scale_for_ctx = _yahoo_proxy_to_mcx_scale(intra_df, mcx_ref)
        ind = dict(sig_intra.indicators)
        if scale_for_ctx is not None:
            _scale_signal_prices_to_mcx_inr(sig_intra, scale_for_ctx)
            ind["prices_scaled_to_mcx_inr"] = True
            ind["proxy_to_mcx_scale"] = round(scale_for_ctx, 8)
            ind["mcx_reference_price"] = round(float(mcx_ref), 4) if mcx_ref is not None else None
        else:
            ind["prices_scaled_to_mcx_inr"] = False
        ind.setdefault("proxy_ticker", y_sym)
        sig_intra.indicators = ind

    inst_ctx = build_institutional_context(
        intra_df,
        daily_merged,
        proxy_ticker=y_sym,
        mcx_scale=scale_for_ctx,
    )

    if y_sym:
        note = (
            f"Intraday review: Yahoo {config.INTRADAY_INTERVAL} bars from {y_sym}. "
            f"Long-term swing: Yahoo {config.LONG_TERM_YAHOO_INTERVAL} bars (same proxy), scaled to MCX INR. "
            "MCX daily bhav remains for calendar context / prior-day metrics in the desk block."
        )
    else:
        note = "No Yahoo proxy mapped; long-term uses MCX daily only."

    return DualHorizonResult(
        mcx_product=prod,
        symbol_key=sk,
        call_generated_at=iso,
        call_generated_at_ist_note=ist_note,
        live_ltp=float(q.ltp) if q is not None else None,
        live_expiry=q.expiry if q is not None else None,
        daily_bars_used=len(daily_merged),
        intraday_bars_used=len(intra_df),
        intraday_interval=config.INTRADAY_INTERVAL,
        intraday_data_note=note,
        daily=sig_daily,
        intraday=sig_intra,
        long_term_target_price=lt_tgt,
        long_term_risk_reward=lt_rr,
        long_term_bars_used=long_term_bars_used,
        institutional_context=inst_ctx,
    )

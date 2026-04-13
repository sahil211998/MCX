from __future__ import annotations

from typing import Any

from mcx_insight import config


def list_mcx_futcom_commodities() -> list[dict[str, Any]]:
    """Enabled FUTCOM roots only (minis + gas); order fixed; mark if seen on MCX watch."""
    import mcxlib

    listed: set[str] = set()
    try:
        d = mcxlib.get_market_watch()
        if not d.empty and "InstrumentName" in d.columns:
            f = d[d["InstrumentName"] == "FUTCOM"]
            listed = set(
                f["ProductCode"].dropna().astype(str).str.strip().str.upper().tolist()
            )
    except Exception:
        listed = set()

    out: list[dict[str, Any]] = []
    for c in config.SIGNAL_ONLY_MCX_PRODUCTS:
        yh = config.YAHOO_INTRADAY_BY_PRODUCT.get(c)
        out.append(
            {
                "mcx_product": c,
                "symbol_key": c.lower(),
                "label": config.SIGNAL_PRODUCT_LABEL.get(c, f"MCX {c}"),
                "intraday_proxy_ticker": yh,
                "has_intraday_proxy": yh is not None,
                "on_mcx_watch": c in listed,
            }
        )
    return out


def yahoo_intraday_ticker(mcx_product: str) -> str | None:
    p = mcx_product.strip().upper()
    if p not in config.SIGNAL_ONLY_MCX_SET:
        return None
    if p in config.YAHOO_INTRADAY_BY_PRODUCT:
        return config.YAHOO_INTRADAY_BY_PRODUCT[p]
    if "OIL" in p:
        return "CL=F"
    if "GAS" in p or p.startswith("NAT"):
        return "NG=F"
    return "HG=F"

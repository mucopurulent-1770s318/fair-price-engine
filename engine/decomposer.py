"""
engine/decomposer.py — Cost decomposition.

Pipeline position:  ItemIdentification + retail_price → CostBreakdown
Grounding modes:
  "llm"  — LLM estimates the BOM from item context + margin tables (MVP default)
  "live" — FRED/BLS commodity prices injected into BOM (v1.1, see price_sources/)

Knowledge tables (CATEGORY_MARGINS, BOM_SPLITS) live here in MVP.
They will be extracted to knowledge/margins/ and knowledge/bom_templates/
in v1.1 to enable community contributions per vertical.

LLM fallback: if Ollama call fails or produces invalid JSON, the generic
margin-table split is used and data_source is set to "margin_table".
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx

from engine.identifier import ItemIdentification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Knowledge tables (MVP — extracted to knowledge/ in v1.1)
#
# Margins represent the RETAILER'S gross margin as a fraction of retail price.
# e.g. furniture 0.50 → true cost = retail * (1 - 0.50)
# Source: industry benchmarks; see knowledge/margins/ for v1.1 sourcing notes.
# ---------------------------------------------------------------------------

CATEGORY_MARGINS: dict[str, dict] = {
    #          typical  low    high
    "furniture":   {"typ": 0.50, "low": 0.40, "high": 0.60},
    "appliances":  {"typ": 0.22, "low": 0.15, "high": 0.28},
    "electronics": {"typ": 0.15, "low": 0.08, "high": 0.22},
    "home_repair": {"typ": 0.60, "low": 0.50, "high": 0.70},  # 2-3x material
    "apparel":     {"typ": 0.60, "low": 0.50, "high": 0.70},
    "general":     {"typ": 0.40, "low": 0.30, "high": 0.50},
}

# BOM splits as fraction of TRUE cost (not retail).
# Always sums to 1.0. "margin" = manufacturer's margin, not retailer's.
BOM_SPLITS: dict[str, dict] = {
    "furniture":   {"materials": 0.38, "manufacturing": 0.35, "overhead": 0.15, "margin": 0.12},
    "appliances":  {"materials": 0.55, "manufacturing": 0.25, "overhead": 0.12, "margin": 0.08},
    "electronics": {"materials": 0.60, "manufacturing": 0.22, "overhead": 0.10, "margin": 0.08},
    "home_repair": {"materials": 0.45, "manufacturing": 0.28, "overhead": 0.15, "margin": 0.12},
    "apparel":     {"materials": 0.32, "manufacturing": 0.42, "overhead": 0.14, "margin": 0.12},
    "general":     {"materials": 0.44, "manufacturing": 0.30, "overhead": 0.15, "margin": 0.11},
}

# Grade multipliers — premium/luxury items have higher true costs relative
# to their retail price (manufacturer captures more margin, not just retailer).
GRADE_MARGIN_ADJUST: dict[str, float] = {
    "budget":  +0.05,   # higher retailer margin squeezes true cost further
    "mid":      0.00,
    "premium": -0.05,   # retailer margin slightly lower; true cost is higher share
    "luxury":  -0.10,
}

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class BOMLine:
    component: str    # e.g. "solid oak lumber (6 board-feet)"
    cost: float       # USD
    pct_of_cost: float  # fraction of true cost (0–1)


@dataclass
class CostBreakdown:
    item:             ItemIdentification
    retail_price:     float
    fair_price_low:   float   # retail * (1 - margin_high)
    fair_price_high:  float   # retail * (1 - margin_low)
    fair_price_mid:   float   # retail * (1 - margin_typ)  ← primary fair price
    overpay_amount:   float   # retail - fair_price_mid
    overpay_pct:      float   # overpay / retail * 100
    bom:              list[BOMLine]
    margin_pct:       float   # retailer margin used (0–1)
    reasoning:        str
    data_source:      str     # "llm" | "margin_table" | "live"
    confidence:       float   # breakdown confidence (inherits item.confidence)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["item"] = self.item.to_dict()
        return d

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

def _build_prompt(item: ItemIdentification, retail: float, margins: dict) -> str:
    true_low  = retail * (1 - margins["high"])
    true_high = retail * (1 - margins["low"])
    return (
        f"You are a cost-of-goods analyst. Break down the true manufacturing cost "
        f"of a {item.name}.\n\n"
        f"Product context:\n"
        f"  Category : {item.category}\n"
        f"  Material : {item.material}\n"
        f"  Grade    : {item.grade}\n"
        f"  Origin   : {item.origin}\n"
        f"  Retail   : ${retail:.2f}\n\n"
        f"Industry data: {item.category} items at {item.grade} grade carry a "
        f"{int(margins['low']*100)}–{int(margins['high']*100)}% retailer margin, "
        f"making the true manufacturing cost approximately ${true_low:.0f}–${true_high:.0f}.\n\n"
        "Produce a specific Bill of Materials — name individual components, not generic buckets.\n"
        "Respond with ONLY valid JSON:\n"
        "{\n"
        "  \"bom\": [\n"
        "    {\"component\": \"<specific part/labor/overhead item>\", "
        "\"cost\": <float>, \"pct_of_cost\": <0.0-1.0>},\n"
        "    ...\n"
        "  ],\n"
        "  \"fair_price_low\": <float>,\n"
        "  \"fair_price_high\": <float>,\n"
        "  \"reasoning\": \"<one sentence explaining the key cost drivers>\"\n"
        "}\n\n"
        "Rules: bom[].pct_of_cost must sum to ~1.0. "
        "Include materials, labour, overhead, AND manufacturer margin as line items."
    )

# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class CostDecomposer:
    """
    Two-stage decomposition:
      1. Pure math from CATEGORY_MARGINS → fair price range (always succeeds)
      2. LLM call for specific BOM line items (falls back to generic split)
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llava",
        grounding_mode: str = "llm",
    ) -> None:
        self._host   = host.rstrip("/")
        self._model  = model
        self._mode   = grounding_mode
        self._client = httpx.AsyncClient(timeout=90.0)

    async def decompose(
        self, item: ItemIdentification, retail_price: float
    ) -> CostBreakdown:
        margins = self._margins(item)
        fair_low, fair_mid, fair_high = self._fair_prices(retail_price, margins)

        bom, reasoning, source = await self._build_bom(item, retail_price, fair_mid, margins)

        return CostBreakdown(
            item           = item,
            retail_price   = retail_price,
            fair_price_low = round(fair_low,  2),
            fair_price_high= round(fair_high, 2),
            fair_price_mid = round(fair_mid,  2),
            overpay_amount = round(retail_price - fair_mid, 2),
            overpay_pct    = round((retail_price - fair_mid) / retail_price * 100, 1),
            bom            = bom,
            margin_pct     = margins["typ"],
            reasoning      = reasoning,
            data_source    = source,
            confidence     = item.confidence,
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _margins(self, item: ItemIdentification) -> dict:
        base = CATEGORY_MARGINS.get(item.category, CATEGORY_MARGINS["general"]).copy()
        adj  = GRADE_MARGIN_ADJUST.get(item.grade, 0.0)
        return {k: max(0.0, min(0.9, v + adj)) for k, v in base.items()}

    def _fair_prices(
        self, retail: float, margins: dict
    ) -> tuple[float, float, float]:
        return (
            retail * (1 - margins["high"]),   # fair_low  (most optimistic)
            retail * (1 - margins["typ"]),    # fair_mid
            retail * (1 - margins["low"]),    # fair_high (most conservative)
        )

    async def _build_bom(
        self,
        item: ItemIdentification,
        retail: float,
        fair_mid: float,
        margins: dict,
    ) -> tuple[list[BOMLine], str, str]:
        """Try LLM first; fall back to generic BOM split on any failure."""
        if self._mode == "llm":
            try:
                return await self._llm_bom(item, retail, margins)
            except Exception as exc:
                logger.warning("LLM BOM failed, using table fallback: %s", exc)

        return self._table_bom(item.category, fair_mid), "margin table estimate", "margin_table"

    async def _llm_bom(
        self,
        item: ItemIdentification,
        retail: float,
        margins: dict,
    ) -> tuple[list[BOMLine], str, str]:
        prompt = _build_prompt(item, retail, margins)
        resp = await self._client.post(
            f"{self._host}/api/generate",
            json={
                "model":  self._model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        d   = json.loads(raw)

        raw_bom   = d.get("bom", [])
        reasoning = str(d.get("reasoning", "LLM cost estimate"))

        # Parse + validate BOM lines
        lines: list[BOMLine] = []
        for entry in raw_bom:
            try:
                lines.append(BOMLine(
                    component   = str(entry["component"]).strip(),
                    cost        = float(entry["cost"]),
                    pct_of_cost = float(entry.get("pct_of_cost", 0.0)),
                ))
            except (KeyError, ValueError, TypeError):
                continue

        if not lines:
            raise ValueError("LLM returned empty BOM")

        # Normalise pct_of_cost to sum to 1.0
        total_pct = sum(l.pct_of_cost for l in lines) or 1.0
        for line in lines:
            line.pct_of_cost = round(line.pct_of_cost / total_pct, 4)

        return lines, reasoning, "llm"

    def _table_bom(self, category: str, fair_mid: float) -> list[BOMLine]:
        """Generic BOM split from BOM_SPLITS table — used when LLM is unavailable."""
        splits = BOM_SPLITS.get(category, BOM_SPLITS["general"])
        labels = {
            "materials":     "Raw materials",
            "manufacturing": "Manufacturing & assembly labor",
            "overhead":      "Factory overhead & quality control",
            "margin":        "Manufacturer profit margin",
        }
        return [
            BOMLine(
                component   = labels[k],
                cost        = round(fair_mid * v, 2),
                pct_of_cost = v,
            )
            for k, v in splits.items()
        ]

    async def close(self) -> None:
        await self._client.aclose()

"""
engine/identifier.py — Item identification via LLaVA (Ollama).

Pipeline position:  image bytes → ItemIdentification
Confidence gate:    enforced by the API layer (main.py), NOT here.
                    If confidence < CONFIDENCE_THRESHOLD, the caller
                    shows the UI interstitial ("Looks like X. Confirm?")
                    before proceeding to decomposer.

LLM output:         strict JSON enforced by prompt + Ollama format=json.
Fallback:           returns ItemIdentification with confidence=0.0 on
                    any Ollama error, so the caller can always gate on it.
"""
import base64
import json
import logging
from dataclasses import dataclass, field, asdict

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed vocabulary for constrained fields — keeps LLM output parseable
# ---------------------------------------------------------------------------
VALID_GRADES      = {"budget", "mid", "premium", "luxury"}
VALID_CONDITIONS  = {"new", "used", "unknown"}
VALID_CATEGORIES  = {
    "furniture", "appliances", "electronics",
    "home_repair", "apparel", "general",
}

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ItemIdentification:
    name:       str    # e.g. "solid oak dining chair"
    material:   str    # e.g. "solid oak, polyester fabric seat"
    grade:      str    # budget | mid | premium | luxury
    origin:     str    # e.g. "likely China, imported"
    condition:  str    # new | used | unknown
    category:   str    # see VALID_CATEGORIES
    confidence: float  # 0.0–1.0 — how certain the model is
    notes:      str    # any extra observations (brand, model, defects)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def fallback(cls, reason: str = "") -> "ItemIdentification":
        """Zero-confidence sentinel returned on Ollama error."""
        return cls(
            name="unknown item", material="unknown", grade="mid",
            origin="unknown", condition="unknown", category="general",
            confidence=0.0, notes=reason,
        )

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT = """You are a product identification expert. Analyze this image.

Respond with ONLY a valid JSON object — no markdown, no prose:
{
  "name":       "<specific product name, e.g. 'solid oak dining chair'>",
  "material":   "<primary materials, e.g. 'solid oak, polyester fabric'>",
  "grade":      "<budget|mid|premium|luxury>",
  "origin":     "<likely manufacturing origin, e.g. 'likely China, imported'>",
  "condition":  "<new|used|unknown>",
  "category":   "<furniture|appliances|electronics|home_repair|apparel|general>",
  "confidence": <0.0-1.0>,
  "notes":      "<brand/model if visible; image quality issues; partial visibility>"
}

Confidence rules:
  1.0 — brand + model clearly visible and legible
  0.8 — item type certain, material clear, no brand
  0.6 — item type likely, some ambiguity in material or grade
  0.4 — item partially visible or image is blurry
  0.2 — very uncertain; multiple plausible interpretations
"""

# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------

class ItemIdentifier:
    """
    Async. One instance per application (reuses httpx client).
    Call identify() with raw JPEG/PNG bytes.
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "llava") -> None:
        self._host  = host.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=60.0)

    async def identify(self, image_bytes: bytes) -> ItemIdentification:
        """
        Send image to LLaVA. Parse and validate response.
        Never raises — returns fallback with confidence=0.0 on any error.
        """
        img_b64 = base64.b64encode(image_bytes).decode()
        try:
            resp = await self._client.post(
                f"{self._host}/api/generate",
                json={
                    "model":  self._model,
                    "prompt": _PROMPT,
                    "images": [img_b64],
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            raw: str = resp.json().get("response", "{}")
            return self._parse(raw)

        except (httpx.HTTPError, Exception) as exc:
            logger.error("Ollama identifier error: %s", exc)
            return ItemIdentification.fallback(str(exc))

    def _parse(self, raw: str) -> ItemIdentification:
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s | raw=%r", exc, raw[:200])
            return ItemIdentification.fallback("JSON parse error")

        # Normalise + validate constrained fields
        grade     = str(d.get("grade", "mid")).lower()
        condition = str(d.get("condition", "unknown")).lower()
        category  = str(d.get("category", "general")).lower()

        if grade     not in VALID_GRADES:      grade     = "mid"
        if condition not in VALID_CONDITIONS:  condition = "unknown"
        if category  not in VALID_CATEGORIES:  category  = "general"

        confidence = float(d.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp

        return ItemIdentification(
            name       = str(d.get("name",     "unknown item")).strip(),
            material   = str(d.get("material", "unknown")).strip(),
            grade      = grade,
            origin     = str(d.get("origin",   "unknown")).strip(),
            condition  = condition,
            category   = category,
            confidence = confidence,
            notes      = str(d.get("notes",    "")).strip(),
        )

    async def close(self) -> None:
        await self._client.aclose()

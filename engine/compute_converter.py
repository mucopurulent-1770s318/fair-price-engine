"""
engine/compute_converter.py — Dollar amounts → compute equivalents.

MVP ships with hardcoded FALLBACK_RATES sourced from the rates table below.
v1.1 will add live rate fetching via price_sources/compute_rates.py; the
fetch_live_rates() stub in that module defines the interface.

MVP Rates Table (as of March 2026):
  Claude Sonnet output tokens : $3.00  / 1M tokens   (Anthropic pricing)
  GPT-4o output tokens        : $5.00  / 1M tokens   (OpenAI pricing)
  Local Llama (electricity)   : $0.10  / 1M tokens   (community benchmark)
  H100 GPU hours              : $2.49  / hr           (Lambda Cloud on-demand)
  RTX 4090 hours              : $0.35  / hr           (vast.ai spot median)
  AI conversation (avg 500tok): $0.0015/ conversation (derived: 500 * $3/1M)
  AI image generation         : $0.02  / image        (Midjourney API estimate)
  AI code completion          : $0.001 / completion   (GitHub Copilot API est.)

The viral hook lives in dual_convert() — it produces retail, fair, AND markup
equivalents so the card can show:
  RETAIL ($299)  →  99.7M Claude tokens
  FAIR   ($224)  →  74.7M Claude tokens
  MARKUP  ($75)  →  25.0M Claude tokens  ← "you're losing this"
"""
import math
from dataclasses import dataclass, asdict
from datetime import date

# ---------------------------------------------------------------------------
# Rates (MVP hardcoded — replaced by live fetch in v1.1)
# ---------------------------------------------------------------------------

FALLBACK_RATES: dict[str, float] = {
    # Tokens (per 1M)
    "claude_sonnet_per_1m_tokens": 3.00,
    "gpt4o_per_1m_tokens":         5.00,
    "local_llama_per_1m_tokens":   0.10,

    # GPU compute (per hour)
    "h100_per_hour":     2.49,
    "rtx4090_per_hour":  0.35,

    # AI work units (per unit)
    "ai_conversation":      0.0015,   # derived: 500 tok * $3/1M
    "ai_image_gen":         0.02,
    "ai_code_completion":   0.001,
}

# Human-scale reference texts (approximate token counts)
_REFERENCES = {
    "war_and_peace":   783_000,   # ~580k words × 1.35 tok/word
    "great_gatsby":      47_000,   # ~36k words
    "phd_dissertation":  52_000,   # ~40k words
}

RATES_DATE = "March 2026"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ComputeEquivalent:
    dollar_amount: float

    # ── Token counts ──────────────────────────────────────────────
    claude_tokens:      int    # at Claude Sonnet rate
    gpt4o_tokens:       int    # at GPT-4o rate
    local_llama_tokens: int    # at local electricity rate

    # ── GPU hours ─────────────────────────────────────────────────
    h100_hours:    float
    rtx4090_hours: float

    # ── AI work units ─────────────────────────────────────────────
    ai_conversations:   int
    ai_image_gens:      int
    ai_code_completions: int

    # ── Human context string ──────────────────────────────────────
    # The most viscerally memorable equivalent — used on the card.
    context_str: str

    # ── Metadata ──────────────────────────────────────────────────
    rates_date: str

    def to_dict(self) -> dict:
        return asdict(self)

    def claude_tokens_display(self) -> str:
        """Format: '99.7M', '1.2B', '450K', '12' etc."""
        return _fmt_tokens(self.claude_tokens)


@dataclass
class DualConversion:
    """
    The three-row display — the viral hook.

    Card renders:
      RETAIL PRICE (${retail})  →  {retail_equiv.claude_tokens_display()} tokens
      FAIR PRICE   (${fair})    →  {fair_equiv.claude_tokens_display()} tokens
      MARKUP       (${markup})  →  {markup_equiv.claude_tokens_display()} tokens  ← viral
    """
    retail_price:  float
    fair_price:    float
    markup_amount: float   # = retail_price - fair_price

    retail_equiv: ComputeEquivalent
    fair_equiv:   ComputeEquivalent
    markup_equiv: ComputeEquivalent

    def to_dict(self) -> dict:
        return {
            "retail_price":    self.retail_price,
            "fair_price":      self.fair_price,
            "markup_amount":   self.markup_amount,
            "retail_equiv":    self.retail_equiv.to_dict(),
            "fair_equiv":      self.fair_equiv.to_dict(),
            "markup_equiv":    self.markup_equiv.to_dict(),
        }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(amount: float, rates: dict | None = None) -> ComputeEquivalent:
    """Convert a single dollar amount to all compute equivalents."""
    r = rates or FALLBACK_RATES
    if amount <= 0:
        return _zero(amount, r)

    claude_tokens       = int(amount / r["claude_sonnet_per_1m_tokens"] * 1_000_000)
    gpt4o_tokens        = int(amount / r["gpt4o_per_1m_tokens"]         * 1_000_000)
    local_llama_tokens  = int(amount / r["local_llama_per_1m_tokens"]   * 1_000_000)
    h100_hours          = round(amount / r["h100_per_hour"],    2)
    rtx4090_hours       = round(amount / r["rtx4090_per_hour"], 2)
    ai_conversations    = int(amount / r["ai_conversation"])
    ai_image_gens       = int(amount / r["ai_image_gen"])
    ai_code_completions = int(amount / r["ai_code_completion"])

    return ComputeEquivalent(
        dollar_amount        = amount,
        claude_tokens        = claude_tokens,
        gpt4o_tokens         = gpt4o_tokens,
        local_llama_tokens   = local_llama_tokens,
        h100_hours           = h100_hours,
        rtx4090_hours        = rtx4090_hours,
        ai_conversations     = ai_conversations,
        ai_image_gens        = ai_image_gens,
        ai_code_completions  = ai_code_completions,
        context_str          = _context_str(amount, claude_tokens, ai_conversations, ai_image_gens, r),
        rates_date           = RATES_DATE,
    )


def dual_convert(
    retail_price: float,
    fair_price:   float,
    rates: dict | None = None,
) -> DualConversion:
    """
    Produce retail + fair + markup equivalents in one call.
    This is what the reporter and frontend consume.

    Markup = retail_price - fair_price
    (If fair_price > retail_price, markup = 0 — item is underpriced.)
    """
    markup = max(0.0, retail_price - fair_price)
    return DualConversion(
        retail_price  = retail_price,
        fair_price    = fair_price,
        markup_amount = round(markup, 2),
        retail_equiv  = convert(retail_price, rates),
        fair_equiv    = convert(fair_price,   rates),
        markup_equiv  = convert(markup,       rates),
    )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _context_str(
    amount: float,
    claude_tokens: int,
    conversations: int,
    images: int,
    rates: dict,
) -> str:
    """
    Return the most viscerally memorable human equivalent for this price.
    Scales with amount so the number is always striking.
    """
    wp_cost = _REFERENCES["war_and_peace"] * rates["claude_sonnet_per_1m_tokens"] / 1_000_000
    wp_readings = int(amount / wp_cost)

    gatsby_cost = _REFERENCES["great_gatsby"] * rates["claude_sonnet_per_1m_tokens"] / 1_000_000
    gatsby_readings = int(amount / gatsby_cost)

    if amount >= 200:
        # Token-scale is most impressive at large amounts
        return (
            f"{_fmt_tokens(claude_tokens)} Claude tokens — "
            f"enough to read War and Peace {wp_readings:,}x"
        )
    elif amount >= 50:
        return (
            f"{conversations:,} AI conversations, or "
            f"read The Great Gatsby {gatsby_readings:,}x with Claude"
        )
    elif amount >= 10:
        return f"{images:,} AI-generated images (at Midjourney API rates)"
    else:
        code_completions = int(amount / rates["ai_code_completion"])
        return f"{code_completions:,} AI code completions"


def _fmt_tokens(n: int) -> str:
    """Format token count: 99666667 → '99.7M', 1200000000 → '1.2B', 450000 → '450K'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _zero(amount: float, rates: dict) -> ComputeEquivalent:
    return ComputeEquivalent(
        dollar_amount=amount, claude_tokens=0, gpt4o_tokens=0,
        local_llama_tokens=0, h100_hours=0.0, rtx4090_hours=0.0,
        ai_conversations=0, ai_image_gens=0, ai_code_completions=0,
        context_str="$0 — nothing to convert", rates_date=RATES_DATE,
    )

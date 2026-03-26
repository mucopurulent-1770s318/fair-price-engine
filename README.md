---

```markdown
# fair-price-engine

> **Point at anything. Know the real cost. See it in tokens.**

You aren't just overpaying in dollars. You're handing over **millions of AI tokens** every time you buy retail.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)

---

## The Narrative

Claude Sonnet costs **$3.00 per million output tokens**.

That $299 KALLAX bookshelf you just bought from IKEA? The actual bill-of-materials cost — lumber, hardware, labor, shipping — is around **$75**. You paid **$224 in pure markup**.

That markup is **74.7 million Claude tokens**.

Enough to have a 150,000-turn conversation with an AI. Enough to generate 11,200 images. Enough to run an H100 for 30 hours.

**That's the number we show you.** Not just the dollar overpay — the compute opportunity cost. Because compute is the new unit of value, and seeing a price in tokens makes it viscerally real in a way that dollars no longer do.

---

## Demo

```
POST /analyze
  image: [photo of KALLAX bookshelf]
  retail_price: 299.00

→ {
    "item": { "name": "KALLAX Bookshelf", "category": "furniture" },
    "breakdown": {
      "retail_price": 299,
      "fair_price_mid": 74.75,
      "overpay_pct": 300.0,
      "bom": [
        { "component": "Particleboard & lumber",  "cost": 28.40, "pct_of_cost": 38.0 },
        { "component": "Hardware & cam locks",     "cost": 8.20,  "pct_of_cost": 11.0 },
        { "component": "Manufacturing labor",      "cost": 18.65, "pct_of_cost": 25.0 },
        { "component": "Shipping & logistics",     "cost": 11.18, "pct_of_cost": 15.0 },
        { "component": "Retail overhead",          "cost": 8.32,  "pct_of_cost": 11.0 }
      ]
    },
    "dual": {
      "retail_equiv":  { "claude_tokens": "99.7M", "ai_conversations": 66466 },
      "fair_equiv":    { "claude_tokens": "24.9M" },
      "markup_equiv":  { "claude_tokens": "74.7M", "h100_hours": 30.1 }
    },
    "viral_quote": "You're being charged 74.7M extra Claude tokens for this KALLAX Bookshelf."
  }
```

Share that result. Watch people's faces change.

---

## Features

### Local Vision — No API Costs
Item identification uses **LLaVA via Ollama**. Snap a photo; the model returns structured JSON: item name, category, brand grade, condition, confidence score. Zero cloud calls. Zero cost per scan.

### Dual-Token Display
Every analysis shows three numbers side by side:

```
RETAIL PRICE   $299  →  99.7M tokens
FAIR PRICE      $75  →  24.9M tokens
MARKUP         $224  →  74.7M tokens   ← the one that hurts
```

The markup row pulses red. That's intentional.

### Pure-Math Fallback — Never Fails
The fair price calculation is a two-stage pipeline:

1. **Stage 1 (always runs):** Category margin tables hardcoded from real-world data. Pure arithmetic. Returns a low/mid/high fair price range in milliseconds regardless of LLM availability.

2. **Stage 2 (enrichment):** LLaVA generates a detailed Bill of Materials with specific line items. If the LLM is slow or unavailable, Stage 1 result is returned with `data_source: "margin_table"`. You always get an answer.

```python
# Margin tables by category (sample)
CATEGORY_MARGINS = {
    "furniture":   {"typ": 0.50, "low": 0.40, "high": 0.60},
    "appliances":  {"typ": 0.22, "low": 0.15, "high": 0.28},
    "electronics": {"typ": 0.15, "low": 0.08, "high": 0.22},
    "home_repair": {"typ": 0.60, "low": 0.50, "high": 0.70},
    "apparel":     {"typ": 0.60, "low": 0.50, "high": 0.70},
}
```

### Shareable Result Cards
Every analysis generates a share URL: `/?result=BASE64`

When Twitter or Discord crawls that URL, the server injects `og:image` meta tags server-side (crawlers don't run JS) pointing to `/og-image?data=BASE64`, which renders a **1200×630 PNG card** on-the-fly via Pillow:

```
┌─────────────────────────────────────────────────────────────────────┐
│ fair-price-engine                                                    │
│ KALLAX Bookshelf                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ RETAIL PRICE    $299  →  99.7M tokens                               │
│ FAIR PRICE       $75  →  24.9M tokens                               │
│ ████ MARKUP     $224  →  74.7M tokens ████                          │
├─────────────────────────────────────────────────────────────────────┤
│ "You're being charged 74.7M extra Claude tokens for this            │
│  KALLAX Bookshelf."                                                  │
│                                                          overpaying 300%  │
└─────────────────────────────────────────────────────────────────────┘
```

No storage. No CDN. The entire card is stateless — encoded in the URL.

### Confidence Gate
When LLaVA isn't certain what it's looking at (confidence < 0.65), the API returns HTTP 202 with an interstitial:

```json
{
  "gate": true,
  "item": { "name": "Sectional Sofa", "confidence": 0.58 },
  "message": "Looks like a Sectional Sofa. Please confirm before we run the analysis."
}
```

The UI shows *"Looks like X — is that right?"* with an editable name field. User confirms or corrects, then resubmits. Bypass with `?autoconfirm=1` for programmatic use.

### Hall of Shame Leaderboard
After each scan, the frontend anonymously submits the item name and markup to `/leaderboard`. The top 20 most-overpriced items scanned are displayed in real time. No PII, no session tracking, resets on restart.

*Current top entries in testing:*
- **Designer Sunglasses** — 89.2M markup tokens (460% overpay)
- **Hotel Mini Bar Water** — 3.1M markup tokens (1200% overpay)
- **Airport USB Cable** — 2.4M markup tokens (900% overpay)

---

## Quickstart

```bash
# Prerequisites: Ollama running with a vision model
ollama pull llava:13b

git clone https://github.com/ninja-otaku/fair-price-engine
cd fair-price-engine
pip install -r requirements.txt

cp .env.example .env
# Set OLLAMA_HOST if not localhost

uvicorn main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama instance URL |
| `IDENTIFIER_MODEL` | `llava:13b` | Vision model for item ID |
| `DECOMPOSER_MODEL` | `llama3:8b` | Text model for BOM generation |
| `GROUNDING_MODE` | `llm` | `llm` (MVP) or `live` (v1.1 with FRED/BLS) |
| `CONFIDENCE_THRESHOLD` | `0.65` | Below this → confidence gate fires |
| `PORT` | `8000` | Server port |
| `TLS_ENABLED` | `false` | HTTPS for production |

---

## Project Structure

```
fair-price-engine/
├── main.py                        # FastAPI: /analyze, /og-image, /leaderboard
├── config.py                      # Pydantic settings
├── engine/
│   ├── identifier.py              # LLaVA → ItemIdentification (with confidence)
│   ├── decomposer.py              # Margin tables + LLM BOM → CostBreakdown
│   ├── compute_converter.py       # Dollar amounts → Claude token equivalents
│   └── reporter.py                # JSON formatters + og:image PNG generator
├── price_sources/
│   ├── base_source.py             # Abstract interface
│   ├── cache.py                   # SQLite TTL cache
│   ├── compute_rates.py           # MVP hardcoded rates (v1.1: live scraping)
│   ├── fred_source.py             # FRED commodity prices — v1.1 STUB
│   └── bls_source.py              # BLS labor rates — v1.1 STUB
└── static/
    └── index.html                 # SPA: upload → gate → results → share
```

---

## How the Token Math Works

Current MVP rates (March 2026, hardcoded with source links):

| Rate | Value | Source |
|---|---|---|
| Claude Sonnet output | $3.00 / 1M tokens | anthropic.com/pricing |
| H100 GPU | $2.49 / hour | lambdalabs.com |
| AI conversation (500 tok) | $0.0015 | derived |
| Midjourney image | $0.02 | docs.midjourney.com |

```python
# Example: $224 markup
claude_tokens = int(224 / 3.00 * 1_000_000)  # → 74,666,666 → "74.7M"
h100_hours    = 224 / 2.49                     # → 89.9 hours
conversations = int(224 / 0.0015)              # → 149,333
```

Context string scales with magnitude:
- `> $200` → *"74.7M tokens — that's War & Peace read by an AI 95 times"*
- `> $50`  → *"equivalent to 149,333 AI conversations"*
- `> $10`  → *"equivalent to 11,200 AI image generations"*

---

## Roadmap

### v1.1 — Live Data Grounding
The biggest weakness of v1.0 is that BOM costs are LLM-estimated. v1.1 will ground them in real commodity and labor data:

**FRED Commodity Prices (Federal Reserve Economic Data)**
- Lumber index (WPU0811) — furniture
- Steel mill products (WPU101) — appliances
- Copper & brass (WPU1021) — plumbing/electronics
- Cotton (WPU0561) — apparel

**BLS Regional Labor Rates (Bureau of Labor Statistics)**
- Manufacturing avg hourly earnings (CEU3000000003)
- Construction avg hourly earnings (CEU2000000003)
- Retail trade avg hourly earnings (CEU6500000003)

The stubs and full implementation guides are already in `price_sources/fred_source.py` and `price_sources/bls_source.py`. Set `GROUNDING_MODE=live` to activate once implemented.

When live: analysis results will carry `data_source: "live-fred+bls"` and cite the actual index values used in the BOM calculation.

### v1.2 — Barcode Mode
Skip the vision model entirely. Scan a barcode → look up product → analyze. Faster and more accurate for packaged goods.

### v2.0 — Browser Extension
Right-click any product image on Amazon, Etsy, or any retailer → "Analyze with Fair Price Engine" → token overlay injected inline.

---

## Contributing

### Improve the Margin Tables

The most impactful contribution right now: better category margins with sources.

Edit `engine/decomposer.py`:

```python
CATEGORY_MARGINS = {
    # PR welcome: cite your source in a comment
    "furniture":   {"typ": 0.50, "low": 0.40, "high": 0.60},  # NRF 2024
    "appliances":  {"typ": 0.22, "low": 0.15, "high": 0.28},  # Consumer Reports
    # Add subcategories: "furniture_outdoor", "furniture_office", etc.
}
```

If you have retail industry data, a cited PR to tighten the margin ranges is extremely valuable.

### Add a Category

1. Add an entry to `CATEGORY_MARGINS` in `decomposer.py`
2. Add the category string to `VALID_CATEGORIES` in `identifier.py`
3. Open a PR with your source

### Implement v1.1 Live Sources

Pick a stub from `price_sources/` — each file contains a complete implementation guide in the docstring. The interfaces and caching layer are already wired up.

---

## API Reference

```
POST /analyze
  image:        (file, optional)   Product photo
  retail_price: (float, required)  Price you paid or are about to pay
  item_json:    (str, optional)    Pre-confirmed ItemIdentification JSON
  ?autoconfirm: (bool)             Skip confidence gate

GET  /og-image?data=BASE64         1200×630 PNG for social sharing
GET  /leaderboard                  Top 20 most-overpriced items
POST /leaderboard                  Submit an entry anonymously
GET  /health                       Server status
GET  /                             SPA (injects og:image meta when ?result= present)
```

---

## License

MIT. Scan everything. Share the numbers. Make people think twice.

If this saves you $50, consider starring the repo — it costs 16.7M tokens, which is actually a bargain.
```
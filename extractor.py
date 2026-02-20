import os
import json
import base64
import time
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PDF_PATH      = Path("data/raw/budget.pdf")
OUTPUT_PATH   = Path("data/processed/budget.json")
PAGES_DIR     = Path("data/processed/pages")
DPI           = 150
MODEL_NAME    = "gemini-2.5-flash"
MAX_RETRIES   = 4
RETRY_BACKOFF = 5

# ── Prompt ────────────────────────────────────────────────────────────────────
EXTRACT_PROMPT = """You are extracting structured data from a Town of Wallingford, Connecticut municipal budget PDF.

STEP 1 — Classify this page as exactly one of these types:
  - "dept_summary"     : Department/Activity Summary — one row per department with totals
  - "revenue"          : Revenue Budget — revenue line items with account numbers
  - "expense_detail"   : Department Budget Estimate (expenses) — line items within one department
  - "staffing"         : Department Budget Estimate (staffing) — headcount by category
  - "capital"          : Capital / Non-Recurring / Appropriations Reserve — capital expenditure items
  - "utility"          : Utility/Enterprise fund pages (Electric, Sewer, Water operating budgets)
  - "other"            : ONLY if the page has zero numbers (pure cover, pure narrative, blank)

IMPORTANT: If the page has ANY numbers, it must NOT be "other".

STEP 2 — Extract every data row with numbers. Use the column mapping below.

The document uses these column headers (may vary slightly by page):
  col1  = FY ENDED 6/30/2024 ACTUAL  (or "FY 6-30-24 ACTUAL")
  col2  = FY 24/25 Thru 1/31/2025 ACTUAL  (or "FY 1-31-25 ACTUAL")
  col3  = 2024-25 APPROP. Original  (or "2024-25 Original")
  col4  = 2024-25 APPROP. ADJ. Thru 1/31/2025  (or "2024-25 Adjusted")
  col5  = FISCAL YEAR 2025-26 DEPT. REQUEST  (or "2025-26 REQUEST")
  col6  = 2025-26 MAYOR  (or "MAYOR APPROVED")
  col7  = 2025-26 FINAL ADOPTED  (or "2025-26 FINAL") ← most important

Return ONLY valid JSON, no markdown, no code fences, no explanation.

Use this exact structure:
{
  "page_type": "<one of the types above>",
  "fund":       "<the fund or division from the page title, e.g. GENERAL FUND, ELECTRIC DIVISION, SEWER DIVISION, WATER DIVISION, CAPITAL AND NON-RECURRING FUND, SPECIAL FUNDS BOARD OF EDUCATION>",
  "department": "<department name from page header e.g. POLICE, MAYOR, REGISTRAR OF VOTERS. For utility/summary pages with no specific department, use the full page title e.g. ELECTRIC DIVISION - SUMMARY>",
  "function":   "<function group if shown, e.g. PUBLIC SAFETY, GENERAL GOVERNMENT — else null>",
  "items": [
    {
      "acct_no":              "string or null",
      "line_item":            "string — name of the row",
      "category":             "string or null — e.g. STAFFING, REVENUE, EXPENDITURE, CAPITAL, SOURCE OF FUNDS, USE OF FUNDS",
      "sub_category":         "string or null — e.g. PROPERTY TAXES, PERSONAL SERVICES, ELECTED & APPOINTED",
      "fy2024_actual":        number or null,
      "fy2025_actual_ytd":    number or null,
      "budget_2425_original": number or null,
      "budget_2425_adjusted": number or null,
      "budget_2526_request":  number or null,
      "budget_2526_mayor":    number or null,
      "budget_2526_final":    number or null
    }
  ]
}

Rules:
- Numbers → plain integers or decimals, no $ or commas.
- Parentheses mean negative: (1,234) → -1234. Dashes "-" mean null.
- Extract ALL rows including subtotals, totals, and grand totals.
- For staffing pages, numbers represent headcount (people), not dollars — still use the same fields.
- For capital/appropriations pages that only show 3 columns (Request, Mayor, Final), put nulls for the earlier columns.
- Always set fund and department from the page header title — never leave them null if the page has a title.
- For utility/enterprise pages (Electric, Sewer, Water), fund = division name (e.g. "ELECTRIC DIVISION"), department = full page title (e.g. "ELECTRIC DIVISION - SUMMARY").
- If a field is not on this page, use null.
- items must always be an array (empty [] only for true "other" pages).
- Do NOT include any text before or after the JSON."""


# ── Helpers ───────────────────────────────────────────────────────────────────
def page_to_base64(page: fitz.Page, dpi: int = DPI) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")


def ask_gemini(image_b64: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                config=types.GenerateContentConfig(max_output_tokens=65536),
                contents=[
                    types.Part(text=EXTRACT_PROMPT),
                    types.Part(
                        inline_data=types.Blob(
                            data=base64.b64decode(image_b64),
                            mime_type="image/png",
                        )
                    ),
                ],
            )
            return response.text.strip()
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = RETRY_BACKOFF * attempt
                print(f"\n    ⏳ Rate limited — waiting {wait}s (attempt {attempt}/{MAX_RETRIES}) …",
                      end="", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries due to rate limiting.")


def parse_response(raw: str) -> tuple[str, dict, list[dict]]:
    """Returns (page_type, metadata, items)."""
    cleaned = raw.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if "```" in cleaned:
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()

    # Find the outermost JSON object
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]

    # Try to parse, and if it fails attempt to fix common truncation issues
    for fix in [lambda s: s, lambda s: s + ']}', lambda s: s + '"]}']:
        try:
            data = json.loads(fix(cleaned))
            page_type = data.get("page_type", "other")
            metadata  = {
                "fund":       data.get("fund"),
                "department": data.get("department"),
                "function":   data.get("function"),
            }
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            # Stamp metadata onto each item
            for item in items:
                for k, v in metadata.items():
                    if not item.get(k):
                        item[k] = v

            # Fallback: infer fund from raw text if Gemini left it null
            if not metadata.get("fund"):
                raw_upper = raw.upper()
                for known in KNOWN_FUNDS:
                    if known in raw_upper:
                        metadata["fund"] = known
                        for item in items:
                            if not item.get("fund"):
                                item["fund"] = known
                        break

            return page_type, metadata, items
        except json.JSONDecodeError:
            continue
    print(f"\n    ⚠️  Could not parse JSON. Last 200 chars: ...{cleaned[-200:]}")
    return "other", {}, []


# Known fund/division names for fallback inference
KNOWN_FUNDS = [
    "ELECTRIC DIVISION", "SEWER DIVISION", "WATER DIVISION",
    "GENERAL FUND", "CAPITAL AND NON-RECURRING", "SPECIAL FUNDS BOARD OF EDUCATION",
    "SPECIAL FUNDS TOWN GOVERNMENT"
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=None,
                        help="Only process first N pages (default: all)")
    parser.add_argument("--page", type=int, default=None,
                        help="Process a single specific page number")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pages that already have a per-page JSON file")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(PDF_PATH)

    if args.page:
        page_range = [args.page - 1]  # convert to 0-indexed
    elif args.pages:
        page_range = range(min(args.pages, len(doc)))
    else:
        page_range = range(len(doc))

    total = len(page_range) if isinstance(page_range, list) else len(doc)
    print(f"📄 Opened '{PDF_PATH}' — processing {len(page_range)} page(s)")
    print(f"🤖 Model : {MODEL_NAME}\n")

    all_items:     list[dict] = []
    page_manifest: list[dict] = []

    for page_num in page_range:
        page = doc[page_num]
        display_num = page_num + 1
        print(f"  [{display_num:>3}/{total}] ", end="", flush=True)

        # ── Resume: load from cache if available ──────────────────────────────
        cached_files = list(PAGES_DIR.glob(f"page_{display_num:03d}_*.json"))
        if args.resume and cached_files:
            cached = json.loads(cached_files[0].read_text())
            page_type = cached["page_type"]
            metadata  = {
                "fund":       cached.get("fund"),
                "department": cached.get("department"),
                "function":   cached.get("function"),
            }
            items = cached.get("items", [])
            dept  = metadata.get("department") or ""
            fund  = metadata.get("fund") or ""
            label = f"{dept} / {fund}".strip(" /") or "—"
            print(f"[{page_type:<14}]  → {len(items)} item(s)  [{label}]  (cached)")
            for item in items:
                item["page_type"]   = page_type
                item["source_page"] = display_num
            all_items.extend(items)
            page_manifest.append({
                "page":            display_num,
                "type":            page_type,
                "fund":            metadata.get("fund"),
                "department":      metadata.get("department"),
                "items_extracted": len(items),
            })
            continue

        # ── Call Gemini ────────────────────────────────────────────────────────
        image_b64 = page_to_base64(page)
        raw       = ask_gemini(image_b64)
        page_type, metadata, items = parse_response(raw)

        for item in items:
            item["page_type"]   = page_type
            item["source_page"] = display_num

        all_items.extend(items)

        dept  = metadata.get("department") or ""
        fund  = metadata.get("fund") or ""
        label = f"{dept} / {fund}".strip(" /") or "—"
        status = "→ skipped" if page_type == "other" else f"→ {len(items)} item(s)  [{label}]"
        print(f"[{page_type:<14}]  {status}")

        # ── Save per-page JSON ─────────────────────────────────────────────────
        page_file = PAGES_DIR / f"page_{display_num:03d}_{page_type}.json"
        with open(page_file, "w") as f:
            json.dump({
                "source_page": display_num,
                "page_type":   page_type,
                "fund":        metadata.get("fund"),
                "department":  metadata.get("department"),
                "function":    metadata.get("function"),
                "items_extracted": len(items),
                "items":       items,
            }, f, indent=2)

        page_manifest.append({
            "page":            display_num,
            "type":            page_type,
            "fund":            metadata.get("fund"),
            "department":      metadata.get("department"),
            "items_extracted": len(items),
        })

    doc.close()

    output = {
        "source_file":    str(PDF_PATH),
        "total_pages":    total,
        "pages_manifest": page_manifest,
        "total_items":    len(all_items),
        "line_items":     all_items,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary stats ─────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(p["type"] for p in page_manifest)
    print(f"\n✅ Done! {len(all_items)} total items → '{OUTPUT_PATH}'")
    print("\n📊 Page breakdown:")
    for ptype, count in sorted(counts.items()):
        print(f"   {ptype:<16} {count} page(s)")


if __name__ == "__main__":
    main()

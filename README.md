# Wallingford Budget Explorer

A tool for extracting, analyzing, and visualizing the Town of Wallingford, Connecticut municipal budget using Gemini vision AI.

## Overview

This project processes the official Wallingford budget PDF using Google Gemini's vision capabilities to extract structured financial data — including revenues, expenditures, staffing, and capital projects — across all funds and departments.

## Project Structure

```
wallingford-budget/
├── data/
│   ├── raw/
│   │   └── budget.pdf          # Source PDF (not committed)
│   └── processed/
│       ├── budget.json         # Combined extracted data (not committed)
│       └── pages/              # Per-page JSON files (not committed)
│           ├── page_001_other.json
│           ├── page_008_revenue.json
│           └── ...
├── tests/
│   └── test_extractor.py       # 63 unit tests for extractor.py
├── extractor.py                # PDF → JSON extraction pipeline
├── insights.py                 # Budget analysis helpers
├── agent.py                    # Conversational agent
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/karthikishorejs/wallingford-budget.git
cd wallingford-budget
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get a free API key at [Google AI Studio](https://aistudio.google.com/).

### 3. Add the budget PDF

Place the Wallingford budget PDF at:
```
data/raw/budget.pdf
```

## Running the Extractor

### Process all pages

```bash
python extractor.py
```

### Process a single page (useful for testing)

```bash
python extractor.py --page 8
```

### Process only the first N pages

```bash
python extractor.py --pages 20
```

### Resume an interrupted run (skip already-processed pages)

```bash
python extractor.py --resume
```

The extractor saves a per-page JSON file under `data/processed/pages/` after each page, so if the run is interrupted (e.g. by a rate limit), you can resume without reprocessing completed pages.

## Output Format

### Per-page JSON (`data/processed/pages/page_NNN_type.json`)

```json
{
  "source_page": 8,
  "page_type": "revenue",
  "fund": "GENERAL FUND",
  "department": "REVENUE SUMMARY",
  "function": null,
  "items_extracted": 14,
  "items": [
    {
      "acct_no": "01-0400",
      "line_item": "CURRENT TAXES",
      "category": "REVENUE",
      "sub_category": "PROPERTY TAXES",
      "fy2024_actual": 98234567,
      "fy2025_actual_ytd": 51234567,
      "budget_2425_original": 99000000,
      "budget_2425_adjusted": 99000000,
      "budget_2526_request": 102000000,
      "budget_2526_mayor": 102000000,
      "budget_2526_final": 102000000
    }
  ]
}
```

### Page types extracted

| Type | Description |
|------|-------------|
| `dept_summary` | Department/Activity Summary — totals by department |
| `revenue` | Revenue Budget — revenue line items with account numbers |
| `expense_detail` | Department Budget Estimate — expenditure line items |
| `staffing` | Department staffing — headcount by category |
| `capital` | Capital / Non-Recurring / Appropriations Reserve |
| `utility` | Utility/Enterprise funds (Electric, Sewer, Water) |
| `other` | Cover pages, narrative, blank pages (no items extracted) |

## Running Tests

```bash
pytest tests/ -v
```

All 63 tests run in under 2 seconds with no real API calls (Gemini client is mocked).

## Model

Uses **Gemini 2.5 Flash** (`gemini-2.5-flash`) via the `google-genai` SDK. The model receives each PDF page as a PNG image and returns structured JSON.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Google Gemini API key |

## License

MIT

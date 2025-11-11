# Biotech Clinical Trial Analysis System

PharmaSight is an automated biotech diligence workflow that gathers scientific evidence, clinical trial activity, and company process documentation before producing an investor-ready PDF briefing. The core engine orchestrates Google Gemini with custom retrieval tooling, while a FastAPI service and React front end make the workflows accessible to product teams.

## Key Capabilities

- PubMed literature discovery with automatic proof-of-work handling for PubMed Central PDF downloads.
- ClinicalTrials.gov landscape summaries, enriched with drug mechanism templates.
- LandingAI ADE parsing for structured extraction from research PDFs, with graceful fallbacks to basic text extraction.
- Gemini-powered metadata inference, full report synthesis, and generated reports.
- PDF generation for stakeholders plus a chat for lookups of specific insight.


## Prerequisites

- Python 3.10 or newer.
- `pip install -r requirements.txt` inside an activated virtual environment.
- Google Gemini API key (`GOOGLE_API_KEY`).
- LandingAI ADE API key (`VISION_AGENT_API_KEY`) for advanced PDF parsing.
- Node.js 18+ and npm for running the React frontend.

## Setup

```bash
git clone <your-repo-url>
cd finAnalysis
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (the repository does not ship with one):

```
GOOGLE_API_KEY=your_google_key_here
GEMINI_MODEL=gemini-1.5-pro
VISION_AGENT_API_KEY=your_landingai_key
LANDINGAI_ADE_MODEL=dpt-2-latest
LANDINGAI_ADE_BASE_URL=https://api.va.landing.ai
```

> Get your Gemini key from [Google AI Studio](https://aistudio.google.com/app/apikey).  
> Get your LandingAI key from [LandingAI](https://landing.ai/).


## Usage

### Command Line Quickstart

Run the built-in Pembrolizumab example and generate a PDF under `reports/`:

```bash
python -m src.main
```

To analyze a different asset, call `BiotechAnalysisSystem().analyze_drug(...)` with your own parameters or duplicate the example block inside `src/main.py`.

### Scenario Scripts

`src/examples.py` bundles a handful of ready-made scenarios and a customization template. From the repo root:

```bash
python src/examples.py
```

Uncomment scenarios inside the file to batch analyses or to point at internal SOP folders.

### FastAPI Service

Launch the REST API (defaults to `http://localhost:8000`):

```bash
uvicorn src.api:app --reload
```


### PharmaSight Frontend

The React + Vite interface lives under `frontend/` and defaults to proxying `/api` to `http://localhost:8000`.

```bash
cd frontend
npm install
npm run dev
```

Override the proxy target for a remote API:

```bash
VITE_API_PROXY_TARGET=http://your-api-host:8000 npm run dev
```

For production builds, set `VITE_API_BASE_URL` before running `npm run build`.

### PDF Output

PDF briefings are saved to `reports/` (or the folder you pass to `save_report`). The filenames follow `<Drug>_<Company>_analysis.pdf`.

## Pipeline Stages

1. **Metadata inference** uses Gemini to enrich missing ticker, phase, or compound aliases before analysis.
2. **Research sweep** leverages `search_and_parse_pubmed_papers` to pull PubMed metadata, resolve open-access PDFs, solve PMC proof-of-work prompts, and parse full-text with LandingAI ADE when configured (PyMuPDF fallback otherwise).
3. **Trial landscape** runs `analyze_clinical_trial_phases` against ClinicalTrials.gov for up-to-date protocol summaries.
4. **Company documentation cross-check** executes `load_company_process_reports` to ingest SOPs, readiness checklists, and other internal documentation.
5. **Gemini synthesis** composes research insights, clinical risk factors, and company docs into a prediction narrative and scenario analysis.
6. **Report generation** formats the findings into a branded PDF and exposes a structured object for downstream clients or the `/query` endpoint.

## Project Layout

```
finAnalysis/
├── src/
│   ├── __init__.py
│   ├── api.py        # FastAPI application with async job queue and report Q&A
│   ├── main.py       # BiotechAnalysisSystem workflow and PDF generator
│   └── tools.py      # PubMed, ClinicalTrials.gov, LandingAI, and PDF helpers
├── frontend/         # PharmaSight React + TypeScript chat interface (Vite)
├── reports/          # Generated PDF deliverables
├── requirements.txt  # Python dependencies
└── README.md         # You are here
```

## License

This project is under the MIT License. See [LICENSE](LICENSE.md) for more information.

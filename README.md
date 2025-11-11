# Biotech Clinical Trial Analysis System

PharmaSight is an automated biotech diligence workflow that gathers scientific evidence, clinical trial activity, and company process documentation before producing an investor-ready PDF briefing. The core engine orchestrates Google Gemini with custom retrieval tooling, while a FastAPI service and React front end make the workflows accessible to product teams.

## Prerequisites

- Python 3.10 or newer.
- `pip install -r requirements.txt` inside an activated virtual environment.
- Google Gemini API key (`GOOGLE_API_KEY`).
- LandingAI ADE API key (`VISION_AGENT_API_KEY`) for advanced PDF parsing.
- Node.js 18+ and npm for running the React frontend.

## Setup

```bash
git clone <your-repo-url>
cd PharmaSight
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

### PharmaSight Frontend

The React + Vite interface lives under `frontend/` and defaults to proxying `/api` to `http://localhost:8000`.

```bash
cd frontend
npm install
npm run dev
```

### Backend

Launch the REST API (defaults to `http://localhost:8000`):

```bash
cd src
uvicorn src.api:app --reload
```

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

"""
FastAPI application that exposes the biotech analysis system as an asynchronous API.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .main import BiotechAnalysisSystem


logger = logging.getLogger(__name__)


class AnalysisRequest(BaseModel):
    """Payload accepted by the analysis endpoint."""

    company: str = Field(..., description="Name of the company producing the drug")
    drug: str = Field(..., description="Drug under investigation")
    indication: Optional[str] = Field(
        None, description="Disease or indication being targeted (optional)"
    )
    ticker: Optional[str] = Field(None, description="Public stock ticker if available")
    current_phase: Optional[str] = Field(
        None, description="Current clinical trial phase for the drug"
    )
    company_reports: Optional[list[str]] = Field(
        None, description="Paths to company process documentation files or directories"
    )


class AnalysisJob(BaseModel):
    """Representation of an analysis job returned to clients."""

    job_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    summary: Optional[str] = None
    report_path: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class ReportQuery(BaseModel):
    """Payload for querying an existing report."""

    question: str = Field(..., description="Question about the generated report")


class _JobStore:
    """Thread-safe in-memory job registry."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Optional[str]]] = {}
        self._lock = Lock()

    def create_job(self) -> AnalysisJob:
        job_id = uuid4().hex
        now = datetime.utcnow()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "summary": None,
                "report_path": None,
                "error": None,
                "result": None,
            }
        return AnalysisJob(**self._jobs[job_id])  # type: ignore[arg-type]

    def update_job(self, job_id: str, **fields) -> AnalysisJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._jobs[job_id].update(fields)
            snapshot = dict(self._jobs[job_id])
        return AnalysisJob(**snapshot)  # type: ignore[arg-type]

    def get_job(self, job_id: str) -> AnalysisJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            snapshot = dict(self._jobs[job_id])
        return AnalysisJob(**snapshot)  # type: ignore[arg-type]


job_store = _JobStore()
analysis_executor = ThreadPoolExecutor(max_workers=2)
_system_lock = Lock()
_system: Optional[BiotechAnalysisSystem] = None


def _get_system() -> BiotechAnalysisSystem:
    global _system
    if _system is not None:
        return _system

    with _system_lock:
        if _system is None:
            logger.info("Initializing BiotechAnalysisSystem for API usage...")
            _system = BiotechAnalysisSystem()
    return _system


app = FastAPI(
    title="Biotech Clinical Trial Analysis API",
    description=(
        "REST API that triggers the biotech analysis agents and provides job status updates."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_analysis(job_id: str, payload: AnalysisRequest) -> None:
    """Worker function executed in a background thread."""
    try:
        job_store.update_job(job_id, status="running", started_at=datetime.utcnow())
    except KeyError:
        logger.error("Job %s disappeared before execution started.", job_id)
        return

    try:
        system = _get_system()
    except Exception as exc:  # noqa: BLE001 - propagate to job record
        logger.exception("Failed to initialise BiotechAnalysisSystem")
        job_store.update_job(
            job_id,
            status="failed",
            completed_at=datetime.utcnow(),
            error=str(exc),
        )
        return

    try:
        ticker_value = (
            payload.ticker.strip() if isinstance(payload.ticker, str) else None
        )
        current_phase_value = (
            payload.current_phase.strip()
            if isinstance(payload.current_phase, str)
            else None
        )
        indication_value = (
            payload.indication.strip()
            if isinstance(payload.indication, str)
            else None
        )

        inferred_metadata = system.infer_metadata_from_gemini(
            drug_name=payload.drug,
            company=payload.company,
            existing_indication=indication_value,
            existing_phase=current_phase_value,
            existing_ticker=ticker_value,
        )

        compound_name = inferred_metadata.get("compound_name")
        company_use = inferred_metadata.get("company_use")

        ticker_value = ticker_value or inferred_metadata.get("ticker")
        indication_value = indication_value or inferred_metadata.get("indication")
        current_phase_value = current_phase_value or inferred_metadata.get(
            "current_phase"
        )

        result = system.analyze_drug(
            drug_name=payload.drug,
            company=payload.company,
            ticker=ticker_value,
            indication=indication_value,
            current_phase=current_phase_value,
            company_reports=payload.company_reports,
            compound_name=compound_name,
            company_use=company_use,
        )
        report_path = system.save_report(result)
        job_store.update_job(
            job_id,
            status="completed",
            completed_at=datetime.utcnow(),
            summary=result.get("prediction"),
            report_path=str(Path(report_path)),
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - capture errors for clients
        logger.exception("Analysis job %s failed", job_id)
        job_store.update_job(
            job_id,
            status="failed",
            completed_at=datetime.utcnow(),
            error=str(exc),
        )


@app.get("/api/health", tags=["Utility"])
async def healthcheck() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/api/analyze",
    response_model=AnalysisJob,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Analysis"],
)
async def queue_analysis(request: AnalysisRequest) -> AnalysisJob:
    """
    Queue an analysis job and return the job status.

    The job runs in the background and can be polled by calling `GET /api/analyze/{job_id}`.
    """
    job = job_store.create_job()
    analysis_executor.submit(_run_analysis, job.job_id, request)
    return job


@app.get(
    "/api/analyze/{job_id}",
    response_model=AnalysisJob,
    tags=["Analysis"],
)
async def get_analysis(job_id: str) -> AnalysisJob:
    """Retrieve the current status of an analysis job."""
    try:
        return job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No job found with id '{job_id}'",
        ) from exc


@app.get(
    "/api/analyze/{job_id}/report",
    response_class=FileResponse,
    tags=["Analysis"],
)
async def download_report(job_id: str) -> FileResponse:
    """Download the generated PDF report for a completed analysis job."""
    try:
        job = job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No job found with id '{job_id}'",
        ) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis job has not completed yet.",
        )

    if not job.report_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No report available for this job.",
        )

    report_path = Path(job.report_path).expanduser().resolve()
    if not report_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found on server.",
        )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=report_path.name,
    )


@app.post(
    "/api/analyze/{job_id}/query",
    tags=["Analysis"],
)
async def query_report(job_id: str, payload: ReportQuery) -> Dict[str, str]:
    """Answer a user question about a completed report."""
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty.",
        )

    try:
        job = job_store.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No job found with id '{job_id}'",
        ) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis job has not completed yet.",
        )

    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No report data available for this job.",
        )

    system = _get_system()
    try:
        answer = system.answer_question_about_report(job.result, payload.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {exc}",
        ) from exc

    return {"answer": answer}


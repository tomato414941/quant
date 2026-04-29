from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Query, status

from app.research_job_store import ResearchJobStore
from app.research_jobs import ResearchJobSpec


RESEARCH_JOB_STORE_DIR_ENV = "QUANT_RESEARCH_JOB_STORE_DIR"
DEFAULT_RESEARCH_JOB_STORE_DIR = Path("backend/data/research_jobs")

router = APIRouter(prefix="/api/research-jobs", tags=["research-jobs"])


def get_research_job_store() -> ResearchJobStore:
    return ResearchJobStore(Path(os.getenv(RESEARCH_JOB_STORE_DIR_ENV, str(DEFAULT_RESEARCH_JOB_STORE_DIR))))


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_research_job(spec: Annotated[ResearchJobSpec, Body(...)]) -> dict:
    record = get_research_job_store().create_or_get(spec)
    return record.to_public_payload()


@router.get("")
async def list_research_jobs(limit: int = Query(50, ge=1, le=500)) -> dict:
    records = get_research_job_store().list(limit=limit)
    return {
        "kind": "research_job_index",
        "jobs": [record.to_public_payload() for record in records],
    }


@router.get("/{job_id}")
async def get_research_job(job_id: str) -> dict:
    record = get_research_job_store().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Research job not found.")
    return record.to_public_payload()


@router.get("/{job_id}/result")
async def get_research_job_result(job_id: str) -> dict:
    store = get_research_job_store()
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Research job not found.")
    if record.status.value != "succeeded":
        raise HTTPException(status_code=409, detail="Research job result is not available.")
    result = store.load_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research job result not found.")
    return result

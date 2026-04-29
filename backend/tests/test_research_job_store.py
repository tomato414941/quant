import json

from app.research_job_store import ResearchJobStore
from app.research_jobs import ResearchJobSpec, ResearchJobStatus


def test_research_job_store_creates_index_and_dedupes_active_jobs(tmp_path) -> None:
    store = ResearchJobStore(tmp_path)
    spec = ResearchJobSpec(jobType="comparison", payload={"a": 1})

    first = store.create_or_get(spec)
    second = store.create_or_get(spec)

    assert second.job_id == first.job_id
    assert first.status == ResearchJobStatus.QUEUED
    index_payload = json.loads((tmp_path / "_index.json").read_text(encoding="utf-8"))
    assert len(index_payload["entries"]) == 1
    assert index_payload["entries"][0]["jobId"] == first.job_id


def test_research_job_store_claim_and_result_lifecycle(tmp_path) -> None:
    store = ResearchJobStore(tmp_path)
    record = store.create_or_get(ResearchJobSpec(jobType="strategy_runs", payload={}))

    claimed = store.claim_next_queued()
    assert claimed is not None
    assert claimed.job_id == record.job_id
    assert claimed.status == ResearchJobStatus.RUNNING

    result_path = store.save_result(record.job_id, {"ok": True})
    succeeded = store.mark_succeeded(record.job_id, result_path=str(result_path))

    assert succeeded.status == ResearchJobStatus.SUCCEEDED
    assert store.load_result(record.job_id) == {"ok": True}


def test_research_job_store_failed_job_does_not_dedupe_new_submission(tmp_path) -> None:
    store = ResearchJobStore(tmp_path)
    spec = ResearchJobSpec(jobType="comparison", payload={})
    first = store.create_or_get(spec)
    store.claim_queued(first.job_id)
    store.mark_failed(first.job_id, error="boom")

    second = store.create_or_get(spec)

    assert second.job_id != first.job_id
    assert second.status == ResearchJobStatus.QUEUED

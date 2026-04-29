from app.research_job_runner import ResearchJobRunner
from app.research_job_store import ResearchJobStore
from app.research_jobs import ResearchJobSpec, ResearchJobStatus, ResearchJobType


def test_research_job_runner_moves_queued_job_to_succeeded(tmp_path) -> None:
    store = ResearchJobStore(tmp_path)
    record = store.create_or_get(ResearchJobSpec(jobType="comparison", payload={"x": 1}))
    runner = ResearchJobRunner(
        store,
        handlers={ResearchJobType.COMPARISON: lambda payload: {"payload": payload}},
    )

    result = runner.run_next_queued()

    assert result is not None
    assert result.job_id == record.job_id
    assert result.status == ResearchJobStatus.SUCCEEDED
    assert store.load_result(record.job_id) == {"payload": {"x": 1}}


def test_research_job_runner_marks_failed_on_handler_error(tmp_path) -> None:
    store = ResearchJobStore(tmp_path)
    record = store.create_or_get(ResearchJobSpec(jobType="comparison", payload={}))

    def fail(_payload):
        raise ValueError("bad job")

    runner = ResearchJobRunner(store, handlers={ResearchJobType.COMPARISON: fail})

    result = runner.run_next_queued()

    assert result is not None
    assert result.job_id == record.job_id
    assert result.status == ResearchJobStatus.FAILED
    assert result.error == "ValueError: bad job"

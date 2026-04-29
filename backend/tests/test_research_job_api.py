import asyncio
import json

import httpx

from app import main as main_module
from app.main import app


class ASGITestClient:
    def __init__(self, app) -> None:
        self._app = app

    def get(self, url: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("GET", url, **kwargs))

    def post(self, url: str, **kwargs) -> httpx.Response:
        return asyncio.run(self._request("POST", url, **kwargs))

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=self._app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)


class ChunkedBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


client = ASGITestClient(app)


def test_create_research_job_does_not_call_heavy_builder(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANT_RESEARCH_JOB_STORE_DIR", str(tmp_path))

    def fail_builder(*_args, **_kwargs):
        raise AssertionError("research job creation must not execute builders")

    monkeypatch.setattr(main_module, "build_comparison_payload", fail_builder)

    response = client.post("/api/research-jobs", json={"jobType": "comparison", "payload": {}})

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["jobType"] == "comparison"
    assert payload["jobId"].startswith("job_")


def test_research_job_status_and_result_endpoints(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUANT_RESEARCH_JOB_STORE_DIR", str(tmp_path))
    created = client.post("/api/research-jobs", json={"jobType": "comparison", "payload": {}}).json()

    status_response = client.get(f"/api/research-jobs/{created['jobId']}")
    result_response = client.get(f"/api/research-jobs/{created['jobId']}/result")

    assert status_response.status_code == 200
    assert status_response.json()["jobId"] == created["jobId"]
    assert result_response.status_code == 409


def test_inline_disabled_heavy_endpoint_does_not_call_builder(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_ALLOW_INLINE_RESEARCH", "false")

    def fail_builder(*_args, **_kwargs):
        raise AssertionError("inline-disabled heavy endpoint must not execute builders")

    monkeypatch.setattr(main_module, "build_strategy_runs_payload", fail_builder)

    response = client.post("/api/strategy-runs")

    assert response.status_code == 409
    assert response.json()["jobEndpoint"] == "/api/research-jobs"


def test_research_job_body_limit_rejects_chunked_body_without_content_length() -> None:
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        body = ChunkedBody([b"{", b'"x":"' + b"a" * (main_module.MAX_RERUN_SPEC_BODY_BYTES + 1) + b'"}'])
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            return await async_client.post(
                "/api/research-jobs",
                content=body,
                headers={"content-type": "application/json"},
            )

    response = asyncio.run(run_request())

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large."

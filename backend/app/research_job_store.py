from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from app.research_jobs import ResearchJobRecord, ResearchJobSpec, ResearchJobStatus, utc_now_iso


RESEARCH_JOB_INDEX_FILENAME = "_index.json"
RESEARCH_JOB_SCHEMA_VERSION = "v1"
ACTIVE_DEDUPE_STATUSES = {
    ResearchJobStatus.QUEUED,
    ResearchJobStatus.RUNNING,
    ResearchJobStatus.SUCCEEDED,
}


class _ResearchJobStoreLock:
    _thread_locks_guard = threading.Lock()
    _thread_locks: dict[Path, threading.RLock] = {}

    def __init__(self, root_dir: Path, lock_path: Path) -> None:
        self._root_dir = root_dir
        self._lock_path = lock_path
        self._thread_lock: threading.RLock | None = None
        self._lock_file = None

    def __enter__(self) -> None:
        with self._thread_locks_guard:
            lock = self._thread_locks.get(self._root_dir)
            if lock is None:
                lock = threading.RLock()
                self._thread_locks[self._root_dir] = lock
        self._thread_lock = lock
        self._thread_lock.acquire()
        try:
            self._lock_file = self._lock_path.open("a+b")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        except BaseException:
            self._thread_lock.release()
            self._thread_lock = None
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._lock_file is not None:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
        finally:
            self._lock_file = None
            if self._thread_lock is not None:
                self._thread_lock.release()
                self._thread_lock = None


class ResearchJobStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root_dir / ".lock"

    def create_or_get(self, spec: ResearchJobSpec) -> ResearchJobRecord:
        fingerprint = build_job_fingerprint(spec)
        with self._locked_root():
            existing = self._find_by_fingerprint_unlocked(fingerprint)
            if existing is not None and existing.status in ACTIVE_DEDUPE_STATUSES:
                return existing
            now = utc_now_iso()
            job_id = build_job_id(now)
            record = ResearchJobRecord(
                jobId=job_id,
                status=ResearchJobStatus.QUEUED,
                jobType=spec.job_type,
                payload=spec.payload,
                jobFingerprint=fingerprint,
                createdAtUtc=now,
                updatedAtUtc=now,
            )
            self._write_record_unlocked(record)
            self._upsert_index_entry_unlocked(record)
            return record

    def get(self, job_id: str) -> ResearchJobRecord | None:
        path = self._record_path(job_id)
        if not path.exists():
            return None
        return ResearchJobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list(self, *, limit: int | None = None) -> list[ResearchJobRecord]:
        entries = self._load_index_entries()
        records = []
        for entry in entries:
            record = self.get(entry["jobId"])
            if record is None:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        return records

    def claim_next_queued(self) -> ResearchJobRecord | None:
        with self._locked_root():
            for entry in self._load_index_entries_unlocked():
                record = self._read_record_unlocked(entry["jobId"])
                if record is None or record.status != ResearchJobStatus.QUEUED:
                    continue
                updated = record.model_copy(
                    update={
                        "status": ResearchJobStatus.RUNNING,
                        "updatedAtUtc": utc_now_iso(),
                    }
                )
                self._write_record_unlocked(updated)
                self._upsert_index_entry_unlocked(updated)
                return updated
        return None

    def claim_queued(self, job_id: str) -> ResearchJobRecord | None:
        with self._locked_root():
            record = self._read_record_unlocked(job_id)
            if record is None or record.status != ResearchJobStatus.QUEUED:
                return record
            updated = record.model_copy(
                update={
                    "status": ResearchJobStatus.RUNNING,
                    "updatedAtUtc": utc_now_iso(),
                }
            )
            self._write_record_unlocked(updated)
            self._upsert_index_entry_unlocked(updated)
            return updated

    def mark_succeeded(self, job_id: str, *, result_path: str) -> ResearchJobRecord:
        return self._update_status(job_id, ResearchJobStatus.SUCCEEDED, result_path=result_path, error=None)

    def mark_failed(self, job_id: str, *, error: str) -> ResearchJobRecord:
        return self._update_status(job_id, ResearchJobStatus.FAILED, result_path=None, error=error)

    def save_result(self, job_id: str, result: dict) -> Path:
        result_path = self.root_dir / job_id / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(result_path, result)
        return result_path

    def load_result(self, job_id: str) -> dict | None:
        record = self.get(job_id)
        if record is None or record.result_path is None:
            return None
        path = Path(record.result_path)
        if not path.is_absolute():
            path = self.root_dir / record.result_path
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _update_status(
        self,
        job_id: str,
        status: ResearchJobStatus,
        *,
        result_path: str | None,
        error: str | None,
    ) -> ResearchJobRecord:
        with self._locked_root():
            record = self._read_record_unlocked(job_id)
            if record is None:
                raise ValueError(f"Unknown research job: {job_id}")
            updated = record.model_copy(
                update={
                    "status": status,
                    "updatedAtUtc": utc_now_iso(),
                    "resultPath": result_path,
                    "error": error,
                }
            )
            self._write_record_unlocked(updated)
            self._upsert_index_entry_unlocked(updated)
            return updated

    def _find_by_fingerprint_unlocked(self, fingerprint: str) -> ResearchJobRecord | None:
        for entry in self._load_index_entries_unlocked():
            if entry.get("jobFingerprint") != fingerprint:
                continue
            return self._read_record_unlocked(entry["jobId"])
        return None

    def _record_path(self, job_id: str) -> Path:
        return self.root_dir / job_id / "job.json"

    def _index_path(self) -> Path:
        return self.root_dir / RESEARCH_JOB_INDEX_FILENAME

    def _read_record_unlocked(self, job_id: str) -> ResearchJobRecord | None:
        path = self._record_path(job_id)
        if not path.exists():
            return None
        return ResearchJobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _write_record_unlocked(self, record: ResearchJobRecord) -> None:
        path = self._record_path(record.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(path, record.model_dump())

    def _load_index_entries(self) -> list[dict]:
        with self._locked_root():
            return self._load_index_entries_unlocked()

    def _load_index_entries_unlocked(self) -> list[dict]:
        path = self._index_path()
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            return []
        return self._sort_index_entries(entries)

    def _upsert_index_entry_unlocked(self, record: ResearchJobRecord) -> None:
        entries = [entry for entry in self._load_index_entries_unlocked() if entry.get("jobId") != record.job_id]
        entries.append(
            {
                "jobId": record.job_id,
                "jobType": record.job_type.value,
                "status": record.status.value,
                "jobFingerprint": record.job_fingerprint,
                "createdAtUtc": record.created_at_utc,
                "updatedAtUtc": record.updated_at_utc,
            }
        )
        payload = {
            "kind": "research_job_index",
            "schemaVersion": RESEARCH_JOB_SCHEMA_VERSION,
            "entries": self._sort_index_entries(entries),
        }
        self._atomic_write_json(self._index_path(), payload)

    def _sort_index_entries(self, entries: list[dict]) -> list[dict]:
        return sorted(entries, key=lambda entry: (entry.get("createdAtUtc") or "", entry.get("jobId") or ""), reverse=True)

    def _atomic_write_json(self, path: Path, payload: dict) -> None:
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        try:
            with temp_path.open("w", encoding="utf-8") as temp_file:
                temp_file.write(serialized)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            temp_path.replace(path)
            self._fsync_directory(path.parent)
        except OSError:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _fsync_directory(self, path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _locked_root(self):
        return _ResearchJobStoreLock(self.root_dir.resolve(), self._lock_path)


def build_job_fingerprint(spec: ResearchJobSpec) -> str:
    payload = {
        "jobType": spec.job_type.value,
        "payload": spec.payload,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_job_id(created_at_utc: str) -> str:
    timestamp = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00")).strftime("%Y%m%d_%H%M%S")
    return f"job_{timestamp}_{uuid.uuid4().hex[:8]}"

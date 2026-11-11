import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sereno_model.data.acquire import ChecksumMismatchError, Source, acquire

NOW = datetime(2026, 11, 3, 15, 30, 0, tzinfo=UTC)
PAYLOAD = b"url,label\nhttps://example.com/,0\n"
PAYLOAD_HASH = hashlib.sha256(PAYLOAD).hexdigest()


def fake_fetch(payload: bytes = PAYLOAD, last_modified: str | None = "Tue, 03 Nov 2026"):
    def fetch(url: str, target: Path) -> str | None:
        target.write_bytes(payload)
        return last_modified

    return fetch


def make_source(**overrides) -> Source:
    values = {
        "name": "demo",
        "url": "https://example.org/demo.csv",
        "filename": "demo.csv",
        "license": "CC BY 4.0",
        "citation": "Demo (2026)",
    }
    return Source(**{**values, **overrides})


def read_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["downloads"]


def test_stores_file_in_timestamped_folder_and_records_it(tmp_path):
    manifest = tmp_path / "manifest.json"

    record = acquire(make_source(), tmp_path / "raw", manifest, fake_fetch(), NOW)

    stored = tmp_path / "raw" / "demo" / "20261103T153000Z" / "demo.csv"
    assert stored.read_bytes() == PAYLOAD
    assert record["sha256"] == PAYLOAD_HASH
    entry = read_manifest(manifest)[0]
    assert entry["source"] == "demo"
    assert entry["url"] == "https://example.org/demo.csv"
    assert entry["sha256"] == PAYLOAD_HASH
    assert entry["bytes"] == len(PAYLOAD)
    assert entry["retrieved_at"] == "2026-11-03T15:30:00Z"
    assert entry["last_modified"] == "Tue, 03 Nov 2026"
    assert entry["license"] == "CC BY 4.0"
    assert entry["citation"] == "Demo (2026)"


def test_rejects_checksum_mismatch_and_leaves_no_trace(tmp_path):
    manifest = tmp_path / "manifest.json"
    source = make_source(expected_sha256="0" * 64)

    with pytest.raises(ChecksumMismatchError):
        acquire(source, tmp_path / "raw", manifest, fake_fetch(), NOW)

    assert not manifest.exists()
    assert [p for p in (tmp_path / "raw").rglob("*") if p.is_file()] == []


def test_new_snapshot_never_overwrites_previous_one(tmp_path):
    manifest = tmp_path / "manifest.json"
    later = datetime(2026, 11, 4, 15, 30, 0, tzinfo=UTC)

    acquire(make_source(), tmp_path / "raw", manifest, fake_fetch(), NOW)
    acquire(make_source(), tmp_path / "raw", manifest, fake_fetch(b"changed"), later)

    first = tmp_path / "raw" / "demo" / "20261103T153000Z" / "demo.csv"
    assert first.read_bytes() == PAYLOAD
    assert len(read_manifest(manifest)) == 2


def test_same_timestamp_refuses_to_overwrite(tmp_path):
    manifest = tmp_path / "manifest.json"

    acquire(make_source(), tmp_path / "raw", manifest, fake_fetch(), NOW)
    with pytest.raises(FileExistsError):
        acquire(make_source(), tmp_path / "raw", manifest, fake_fetch(b"changed"), NOW)

    assert (tmp_path / "raw" / "demo" / "20261103T153000Z" / "demo.csv").read_bytes() == PAYLOAD
    assert len(read_manifest(manifest)) == 1


def test_pinned_source_already_recorded_is_not_downloaded_again(tmp_path):
    manifest = tmp_path / "manifest.json"
    source = make_source(expected_sha256=PAYLOAD_HASH)
    later = datetime(2026, 11, 4, 15, 30, 0, tzinfo=UTC)

    def must_not_fetch(url: str, target: Path) -> str | None:
        raise AssertionError("pinned source was downloaded again")

    first = acquire(source, tmp_path / "raw", manifest, fake_fetch(), NOW)
    second = acquire(source, tmp_path / "raw", manifest, must_not_fetch, later)

    assert second == first
    assert len(read_manifest(manifest)) == 1

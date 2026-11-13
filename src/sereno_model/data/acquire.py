"""Download raw data sources into timestamped snapshots and record them in a manifest.

Sources and snapshots: docs/01-datos.md.
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

USER_AGENT = "sereno-model-research"
TIMEOUT_SECONDS = 120
CHUNK_BYTES = 1024 * 1024

Fetch = Callable[[str, Path], str | None]


class ChecksumMismatchError(Exception):
    """The downloaded bytes differ from the checksum published by the source."""


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    filename: str
    license: str
    citation: str
    expected_sha256: str | None = None


SOURCES = (
    Source(
        name="hannousse_2021",
        url=(
            "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/files/"
            "575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
        ),
        filename="dataset_B_05_2020.csv",
        license="CC BY 4.0",
        citation=(
            "Hannousse, A. & Yahiouche, S. (2021). Web page phishing detection (Version 3). "
            "Mendeley Data. https://doi.org/10.17632/c2gw7fy2j4.3"
        ),
        expected_sha256="21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c",
    ),
    Source(
        name="url_phish_2026",
        url=(
            "https://data.mendeley.com/public-files/datasets/65z9twcx3r/files/"
            "0e9c55e4-9adb-43f5-8403-1bbd143ebdb6/file_downloaded"
        ),
        filename="Dataset.csv",
        license="CC BY 4.0",
        citation=(
            "Dam Minh, L. & Tran Cong, H. (2026). URL-Phish: A Feature-Engineered Dataset for "
            "Phishing Detection (Version 2). Mendeley Data. https://doi.org/10.17632/65z9twcx3r.2"
        ),
        expected_sha256="d68b3cd0648dcf9c775347416ad1a8995e8a025921fbe3871ca6158d4db3c3a1",
    ),
    Source(
        name="phishtank",
        url="https://data.phishtank.com/data/online-valid.csv.gz",
        filename="online-valid.csv.gz",
        license="PhishTank terms; pending review",
        citation="PhishTank. Online valid phishing URLs. https://phishtank.org/",
    ),
    Source(
        name="majestic_million",
        url="https://downloads.majestic.com/majestic_million.csv",
        filename="majestic_million.csv",
        license="CC BY 3.0",
        citation="Majestic. The Majestic Million. https://majestic.com/reports/majestic-million",
    ),
)


def http_fetch(url: str, target: Path) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        with target.open("wb") as output:
            shutil.copyfileobj(response, output, CHUNK_BYTES)
        return response.headers.get("Last-Modified")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_downloads(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))["downloads"]


def save_downloads(manifest_path: Path, downloads: list[dict]) -> None:
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"downloads": downloads}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def acquire(
    source: Source,
    raw_dir: Path,
    manifest_path: Path,
    fetch: Fetch = http_fetch,
    now: datetime | None = None,
) -> dict:
    """Store one snapshot of a source and return its manifest record.

    A source with a published checksum is downloaded once; a feed gets a new snapshot on
    every call.
    """
    downloads = load_downloads(manifest_path)
    if source.expected_sha256:
        for record in downloads:
            if record["source"] == source.name and record["sha256"] == source.expected_sha256:
                return record

    moment = now or datetime.now(UTC)
    snapshot_dir = raw_dir / source.name / moment.strftime("%Y%m%dT%H%M%SZ")
    target = snapshot_dir / source.filename
    if target.exists():
        raise FileExistsError(target)

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        last_modified = fetch(source.url, partial)
        checksum = sha256_of(partial)
        if source.expected_sha256 and checksum != source.expected_sha256:
            raise ChecksumMismatchError(
                f"{source.name}: expected {source.expected_sha256}, got {checksum}"
            )
        size = partial.stat().st_size
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)

    record = {
        "source": source.name,
        "url": source.url,
        "path": Path(os.path.relpath(target, manifest_path.parent)).as_posix(),
        "sha256": checksum,
        "bytes": size,
        "retrieved_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_modified": last_modified,
        "license": source.license,
        "citation": source.citation,
    }
    save_downloads(manifest_path, [*downloads, record])
    return record


def main() -> None:
    by_name = {source.name: source for source in SOURCES}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", help=f"Default: all of {', '.join(by_name)}")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    unknown = set(args.sources) - set(by_name)
    if unknown:
        parser.error(f"unknown sources: {', '.join(sorted(unknown))}")

    for name in args.sources or by_name:
        record = acquire(by_name[name], args.data_dir / "raw", args.data_dir / "manifest.json")
        print(f"{record['source']}: {record['bytes']} bytes, sha256 {record['sha256']}")


if __name__ == "__main__":
    main()

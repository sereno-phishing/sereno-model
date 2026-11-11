import csv
import gzip
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.clean import clean, main

NOW = datetime(2026, 11, 11, 15, 30, 0, tzinfo=UTC)
LATER = datetime(2026, 11, 12, 8, 0, 0, tzinfo=UTC)
STAMP = "20261111T153000Z"

HANNOUSSE = [
    ("http://shop.example.com/cart", "legitimate"),
    ("http://login.bad.example/verify", "phishing"),
]
URL_PHISH = [
    ("https://www.uni.example.edu/", "0"),
    ("https://secure.bad.example/account?id=1", "1"),
]
PHISHTANK = [
    ("https://tank.bad.example/", "2026-11-01T10:00:00+00:00", "yes"),
]
MAJESTIC = [("1", "google.com"), ("2", "example.org")]


def to_csv(header: list[str], rows: list[tuple]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def make_data_dir(
    root: Path,
    hannousse=HANNOUSSE,
    url_phish=URL_PHISH,
    phishtank=PHISHTANK,
    majestic=MAJESTIC,
) -> Path:
    """Write tiny raw snapshots plus a manifest that records their real hashes."""
    files = {
        "hannousse_2021": (
            "dataset_B_05_2020.csv",
            to_csv(["url", "length_url", "status"], [(u, len(u), s) for u, s in hannousse]),
        ),
        "url_phish_2026": (
            "Dataset.csv",
            to_csv(["url", "url_len", "label"], [(u, len(u), lab) for u, lab in url_phish]),
        ),
        "phishtank": (
            "online-valid.csv.gz",
            gzip.compress(
                to_csv(
                    ["phish_id", "url", "submission_time", "verified", "target"],
                    [(i, u, t, v, "Other") for i, (u, t, v) in enumerate(phishtank)],
                ),
                mtime=0,
            ),
        ),
        "majestic_million": (
            "majestic_million.csv",
            to_csv(["GlobalRank", "TldRank", "Domain"], [(r, r, d) for r, d in majestic]),
        ),
    }
    data_dir = root / "data"
    downloads = []
    for source, (filename, payload) in files.items():
        relative = f"raw/{source}/20261104T000000Z/{filename}"
        target = data_dir / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        downloads.append(
            {
                "source": source,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "retrieved_at": "2026-11-04T00:00:00Z",
            }
        )
    (data_dir / "manifest.json").write_text(json.dumps({"downloads": downloads}), encoding="utf-8")
    return data_dir


def read_urls(data_dir: Path, stamp: str = STAMP) -> dict[str, dict]:
    frame = pd.read_parquet(data_dir / "processed" / stamp / "urls.parquet")
    return {row["url"]: row for row in frame.to_dict("records")}


def test_writes_canonical_labels_and_expected_columns(tmp_path):
    data_dir = make_data_dir(tmp_path)

    clean(data_dir, NOW)

    frame = pd.read_parquet(data_dir / "processed" / STAMP / "urls.parquet")
    expected = ["url", "label", "sources", "has_path_or_query", "submission_time"]
    assert list(frame.columns) == expected
    assert list(frame["url"]) == sorted(frame["url"])
    labels = dict(zip(frame["url"], frame["label"], strict=True))
    assert labels == {
        "http://shop.example.com/cart": 0,
        "http://login.bad.example/verify": 1,
        "https://www.uni.example.edu/": 0,
        "https://secure.bad.example/account?id=1": 1,
        "https://tank.bad.example/": 1,
    }
    rows = read_urls(data_dir)
    assert rows["https://tank.bad.example/"]["submission_time"] == "2026-11-01T10:00:00+00:00"
    assert pd.isna(rows["http://shop.example.com/cart"]["submission_time"])
    assert rows["https://tank.bad.example/"]["has_path_or_query"] is False
    assert rows["https://secure.bad.example/account?id=1"]["has_path_or_query"] is True


def test_column_types_do_not_depend_on_the_rows_present(tmp_path):
    data_dir = make_data_dir(tmp_path, phishtank=[])

    clean(data_dir, NOW)

    schema = pq.read_schema(data_dir / "processed" / STAMP / "urls.parquet")
    types = {name: str(schema.field(name).type) for name in schema.names}
    assert types["label"] == "int8"
    assert types["has_path_or_query"] == "bool"
    assert types["submission_time"] in ("string", "large_string")
    assert types["url"] in ("string", "large_string")
    assert types["sources"].startswith(("list<", "large_list<"))


def test_report_records_inputs_with_their_hashes(tmp_path):
    data_dir = make_data_dir(tmp_path)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["downloads"]

    report = clean(data_dir, NOW)

    saved = json.loads(
        (data_dir / "processed" / STAMP / "clean_report.json").read_text(encoding="utf-8")
    )
    assert saved == report
    assert report["output_dir"] == f"processed/{STAMP}"
    assert report["created_at"] == "2026-11-11T15:30:00Z"
    for record in manifest:
        assert report["inputs"][record["source"]]["path"] == record["path"]
        assert report["inputs"][record["source"]]["sha256"] == record["sha256"]
    assert report["label_mapping"]["url_phish_2026"] == {"0": 0, "1": 1}


def test_rejects_raw_file_that_differs_from_manifest_hash(tmp_path):
    data_dir = make_data_dir(tmp_path)
    tampered = data_dir / "raw/url_phish_2026/20261104T000000Z/Dataset.csv"
    tampered.write_bytes(tampered.read_bytes() + b"http://extra.example/,0\n")

    with pytest.raises(ChecksumMismatchError, match="url_phish_2026"):
        clean(data_dir, NOW)

    assert not (data_dir / "processed").exists()


def test_fails_when_a_source_is_missing_from_manifest(tmp_path):
    data_dir = make_data_dir(tmp_path)
    manifest_path = data_dir / "manifest.json"
    downloads = json.loads(manifest_path.read_text(encoding="utf-8"))["downloads"]
    kept = [record for record in downloads if record["source"] != "phishtank"]
    manifest_path.write_text(json.dumps({"downloads": kept}), encoding="utf-8")

    with pytest.raises(LookupError, match="phishtank"):
        clean(data_dir, NOW)


def test_uses_latest_snapshot_of_each_source(tmp_path):
    data_dir = make_data_dir(tmp_path)
    payload = gzip.compress(
        to_csv(
            ["url", "submission_time", "verified"],
            [("https://newer.bad.example/", "2026-11-09T00:00:00+00:00", "yes")],
        ),
        mtime=0,
    )
    relative = "raw/phishtank/20261110T000000Z/online-valid.csv.gz"
    (data_dir / relative).parent.mkdir(parents=True)
    (data_dir / relative).write_bytes(payload)
    manifest_path = data_dir / "manifest.json"
    downloads = json.loads(manifest_path.read_text(encoding="utf-8"))["downloads"]
    downloads.append(
        {
            "source": "phishtank",
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "retrieved_at": "2026-11-10T00:00:00Z",
        }
    )
    manifest_path.write_text(json.dumps({"downloads": downloads}), encoding="utf-8")

    report = clean(data_dir, NOW)

    assert report["inputs"]["phishtank"]["path"] == relative
    assert "https://newer.bad.example/" in read_urls(data_dir)
    assert "https://tank.bad.example/" not in read_urls(data_dir)


def test_counts_and_excludes_invalid_urls_per_source_and_reason(tmp_path):
    data_dir = make_data_dir(
        tmp_path,
        hannousse=[
            *HANNOUSSE,
            ("ftp://files.example.com/a", "legitimate"),
            ("http://bad.example/a b", "phishing"),
            ("http://bad.example/c d", "phishing"),
        ],
        url_phish=[*URL_PHISH, ("   ", "0"), ("http:///nohost", "1")],
        phishtank=[*PHISHTANK, ("https://pending.bad.example/", "2026-11-02T00:00:00+00:00", "no")],
    )

    report = clean(data_dir, NOW)

    sources = report["sources"]
    assert sources["hannousse_2021"]["rows_read"] == 5
    assert sources["hannousse_2021"]["invalid"] == {
        "unsupported_scheme": 1,
        "whitespace_or_control": 2,
    }
    assert sources["hannousse_2021"]["valid_rows"] == 2
    assert sources["url_phish_2026"]["invalid"] == {"empty": 1, "missing_host": 1}
    assert sources["phishtank"]["invalid"] == {"not_verified": 1}
    assert len(read_urls(data_dir)) == 5


def test_removes_duplicates_within_a_source_after_normalization(tmp_path):
    data_dir = make_data_dir(
        tmp_path,
        hannousse=[
            ("http://shop.example.com/cart", "legitimate"),
            ("HTTP://Shop.Example.com:80/cart#top", "legitimate"),
            ("http://shop.example.com/c%61rt", "legitimate"),
            ("http://shop.example.com/cart/", "legitimate"),
        ],
    )

    report = clean(data_dir, NOW)

    stats = report["sources"]["hannousse_2021"]
    assert stats["valid_rows"] == 4
    assert stats["unique_urls"] == 2
    assert stats["duplicates_within_source"] == 2
    rows = read_urls(data_dir)
    assert list(rows["http://shop.example.com/cart"]["sources"]) == ["hannousse_2021"]
    assert "http://shop.example.com/cart/" in rows


def test_merges_same_label_duplicates_across_sources_and_records_both(tmp_path):
    data_dir = make_data_dir(
        tmp_path,
        url_phish=[*URL_PHISH, ("HTTPS://Tank.bad.example:443/", "1")],
        phishtank=[
            ("https://tank.bad.example/", "2026-11-03T10:00:00+00:00", "yes"),
            ("https://tank.bad.example/#again", "2026-11-01T10:00:00+00:00", "yes"),
        ],
    )

    report = clean(data_dir, NOW)

    row = read_urls(data_dir)["https://tank.bad.example/"]
    assert list(row["sources"]) == ["phishtank", "url_phish_2026"]
    assert row["label"] == 1
    assert row["submission_time"] == "2026-11-01T10:00:00+00:00"
    assert report["cross_source"]["urls_in_multiple_sources"] == 1
    assert report["cross_source"]["duplicate_rows_merged"] == 1
    assert report["cross_source"]["overlap_by_source_pair"] == {
        "phishtank+url_phish_2026": {"same_label": 1, "conflicting": 0}
    }
    assert report["sources"]["phishtank"]["duplicates_within_source"] == 1
    assert report["kept"]["total"] == 5
    assert report["kept"]["by_source"]["phishtank"] == {"phishing": 1, "legitimate": 0}
    assert report["kept"]["by_source"]["url_phish_2026"] == {"phishing": 2, "legitimate": 1}


def test_excludes_urls_whose_sources_disagree_on_the_label(tmp_path):
    data_dir = make_data_dir(
        tmp_path,
        hannousse=[*HANNOUSSE, ("https://tank.bad.example/", "legitimate")],
    )

    report = clean(data_dir, NOW)

    assert "https://tank.bad.example/" not in read_urls(data_dir)
    conflicts = pd.read_parquet(data_dir / "processed" / STAMP / "conflicts.parquet")
    assert conflicts.to_dict("records") == [
        {"url": "https://tank.bad.example/", "source": "hannousse_2021", "label": 0},
        {"url": "https://tank.bad.example/", "source": "phishtank", "label": 1},
    ]
    assert report["conflicts"] == {"urls": 1, "by_sources": {"hannousse_2021+phishtank": 1}}
    assert report["cross_source"]["overlap_by_source_pair"] == {
        "hannousse_2021+phishtank": {"same_label": 0, "conflicting": 1}
    }
    assert report["kept"]["total"] == 4
    assert report["kept"]["by_source"]["phishtank"] == {"phishing": 0, "legitimate": 0}


def test_excludes_urls_with_both_labels_inside_one_source(tmp_path):
    data_dir = make_data_dir(
        tmp_path,
        hannousse=[*HANNOUSSE, ("http://shop.example.com/cart", "phishing")],
    )

    report = clean(data_dir, NOW)

    assert "http://shop.example.com/cart" not in read_urls(data_dir)
    assert report["conflicts"] == {"urls": 1, "by_sources": {"hannousse_2021": 1}}


def test_writes_empty_conflicts_file_when_sources_agree(tmp_path):
    data_dir = make_data_dir(tmp_path)

    report = clean(data_dir, NOW)

    conflicts = pd.read_parquet(data_dir / "processed" / STAMP / "conflicts.parquet")
    assert list(conflicts.columns) == ["url", "source", "label"]
    assert len(conflicts) == 0
    assert report["conflicts"] == {"urls": 0, "by_sources": {}}


def test_rejects_unknown_label_values(tmp_path):
    data_dir = make_data_dir(tmp_path, url_phish=[*URL_PHISH, ("http://x.example/", "2")])

    with pytest.raises(ValueError, match="url_phish_2026"):
        clean(data_dir, NOW)


def test_majestic_domains_are_kept_apart_and_never_become_urls(tmp_path):
    data_dir = make_data_dir(
        tmp_path,
        majestic=[
            ("1", "Google.com"),
            ("2", "example.org"),
            ("3", "google.com"),
            ("4", "not a domain"),
            ("5", "xn--80ak6aa92e.com"),
            ("6", ""),
            ("7", "localhost"),
        ],
    )

    report = clean(data_dir, NOW)

    domains = pd.read_parquet(data_dir / "processed" / STAMP / "majestic_domains.parquet")
    assert domains.to_dict("records") == [
        {"domain": "google.com", "global_rank": 1},
        {"domain": "example.org", "global_rank": 2},
        {"domain": "xn--80ak6aa92e.com", "global_rank": 5},
    ]
    assert report["majestic_domains"] == {
        "rows_read": 7,
        "invalid": {"empty": 1, "invalid_domain": 2},
        "duplicates": 1,
        "kept": 3,
    }
    urls = read_urls(data_dir)
    assert not any("google.com" in url for url in urls)
    assert "majestic_million" not in report["sources"]
    assert report["kept"]["total"] == 5


def test_reports_kept_rows_and_path_share_per_class_and_source(tmp_path):
    data_dir = make_data_dir(tmp_path)

    report = clean(data_dir, NOW)

    assert report["kept"]["by_class"] == {"phishing": 3, "legitimate": 2}
    assert report["kept"]["by_source"]["hannousse_2021"] == {"phishing": 1, "legitimate": 1}
    shares = report["path_or_query_share"]
    assert shares["by_class"] == {"phishing": pytest.approx(2 / 3, abs=1e-6), "legitimate": 0.5}
    assert shares["by_source"]["hannousse_2021"] == {"phishing": 1.0, "legitimate": 1.0}
    assert shares["by_source"]["url_phish_2026"] == {"phishing": 1.0, "legitimate": 0.0}
    assert shares["by_source"]["phishtank"] == {"phishing": 0.0, "legitimate": None}


def test_content_hash_covers_sorted_url_label_lines(tmp_path):
    data_dir = make_data_dir(tmp_path)
    lines = [
        "http://login.bad.example/verify\t1\n",
        "http://shop.example.com/cart\t0\n",
        "https://secure.bad.example/account?id=1\t1\n",
        "https://tank.bad.example/\t1\n",
        "https://www.uni.example.edu/\t0\n",
    ]

    report = clean(data_dir, NOW)

    expected = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert report["content_hash"]["value"] == expected
    assert report["content_hash"]["algorithm"] == "sha256"


def test_content_hash_ignores_run_time_and_input_row_order(tmp_path):
    first_dir = make_data_dir(tmp_path / "first")
    second_dir = make_data_dir(
        tmp_path / "second",
        hannousse=list(reversed(HANNOUSSE)),
        url_phish=list(reversed(URL_PHISH)),
    )

    first = clean(first_dir, NOW)
    again = clean(first_dir, LATER)
    reordered = clean(second_dir, NOW)

    assert first["content_hash"]["value"] == again["content_hash"]["value"]
    assert first["content_hash"]["value"] == reordered["content_hash"]["value"]
    assert again["output_dir"] == "processed/20261112T080000Z"
    assert read_urls(first_dir).keys() == read_urls(first_dir, "20261112T080000Z").keys()


def test_content_hash_changes_when_a_label_changes(tmp_path):
    first_dir = make_data_dir(tmp_path / "first")
    second_dir = make_data_dir(
        tmp_path / "second",
        hannousse=[("http://shop.example.com/cart", "phishing"), HANNOUSSE[1]],
    )

    first = clean(first_dir, NOW)
    second = clean(second_dir, NOW)

    assert first["content_hash"]["value"] != second["content_hash"]["value"]


def test_refuses_to_overwrite_an_existing_output_folder(tmp_path):
    data_dir = make_data_dir(tmp_path)

    clean(data_dir, NOW)
    with pytest.raises(FileExistsError):
        clean(data_dir, NOW)


def test_cli_cleans_the_given_data_dir_and_prints_a_summary(tmp_path, capsys):
    data_dir = make_data_dir(tmp_path)

    main(["--data-dir", str(data_dir)])

    output = capsys.readouterr().out
    assert "kept 5 URLs" in output
    folders = [path for path in (data_dir / "processed").iterdir() if path.is_dir()]
    assert len(folders) == 1
    assert (folders[0] / "clean_report.json").exists()

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.assemble import DATASET_SCHEMA, dataset_hash
from sereno_model.data.clean import write_parquet
from sereno_model.data.domains import suffix_list_snapshot_id
from sereno_model.data.normalize import parse_url
from sereno_model.data.split import SPLIT_SCHEMA, split
from sereno_model.features.build import (
    AUDIT_COLUMNS,
    METADATA_COLUMNS,
    build,
    git_state,
    main,
    matrix_hash,
)
from sereno_model.features.extract import FEATURE_NAMES, FEATURE_SET_VERSION, extract_features

ASSEMBLE_STAMP = "20261112T090000Z"
SPLIT_NOW = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)
SPLIT_STAMP = "20261113T090000Z"
NOW = datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 11, 14, 10, 0, 0, tzinfo=UTC)
STAMP = "20261114T090000Z"
LATER_STAMP = "20261114T100000Z"

PHISHING, LEGITIMATE = 1, 0
FILES = {"train": "features_train.parquet", "test": "features_test.parquet"}
REPORT = "features_report.json"


def row(url: str, label: int, sources: tuple[str, ...], synthetic: bool = False) -> dict:
    parts = parse_url(url)
    return {
        "url": url,
        "label": label,
        "sources": list(sources),
        "has_path_or_query": parts.has_path_or_query,
        "scheme": parts.scheme,
        "has_www": parts.host.startswith("www."),
        "synthetic": synthetic,
        "submission_time": None,
    }


def sample_rows(domains: int = 20) -> list[dict]:
    """Two classes over many small domains, with every scheme and ``www.`` combination."""
    rows = []
    for index in range(domains):
        rows += [
            row(f"https://bad{index}.com/login?id={index}&next=//x", PHISHING, ("phishtank",)),
            row(f"http://www.secure-bad{index}.com/a/b", PHISHING, ("hannousse_2021",)),
            row(f"https://www.good{index}.org/about", LEGITIMATE, ("url_phish_2026",)),
            row(f"http://good{index}.org", LEGITIMATE, ("hannousse_2021", "url_phish_2026")),
            row(f"https://top{index}.net", LEGITIMATE, ("majestic_million",), synthetic=True),
        ]
    rows += [row(f"http://203.0.113.9/panel/{index}", PHISHING, ("phishtank",)) for index in (1, 2)]
    return rows


def make_split_run(root: Path, rows: list[dict] | None = None) -> Path:
    """Write a tiny assembled dataset, split it with the real splitter and return the data dir."""
    rows = sample_rows() if rows is None else rows
    frame = pd.DataFrame(sorted(rows, key=lambda item: item["url"]), columns=DATASET_SCHEMA.names)
    frame = frame.astype(
        {"label": "int8", "has_path_or_query": "bool", "has_www": "bool", "synthetic": "bool"}
    )
    data_dir = root / "data"
    run_dir = data_dir / "processed" / ASSEMBLE_STAMP
    run_dir.mkdir(parents=True)
    write_parquet(frame, DATASET_SCHEMA, run_dir / "dataset.parquet")
    report = {
        "output_dir": f"processed/{ASSEMBLE_STAMP}",
        "content_hash": {"value": dataset_hash(frame)},
    }
    (run_dir / "assemble_report.json").write_text(json.dumps(report), encoding="utf-8")
    split(data_dir, seed=7, now=SPLIT_NOW)
    return data_dir


def read_split(data_dir: Path, stamp: str = SPLIT_STAMP) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / stamp / "split.parquet")


def read_matrix(data_dir: Path, name: str, stamp: str = STAMP) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / stamp / FILES[name])


def read_report(data_dir: Path, stamp: str = STAMP) -> dict:
    return json.loads((data_dir / "processed" / stamp / REPORT).read_text(encoding="utf-8"))


def test_writes_one_matrix_per_split_with_every_row_of_the_partition(tmp_path):
    data_dir = make_split_run(tmp_path)
    partition = read_split(data_dir)

    build(data_dir, now=NOW)

    for name in ("train", "test"):
        matrix = read_matrix(data_dir, name)
        expected = partition[partition["split"] == name]
        assert list(matrix["url"]) == list(expected["url"])
        assert set(matrix["split"]) == {name}
        assert list(matrix["label"]) == list(expected["label"])
        assert list(matrix["group"]) == list(expected["group"])
        assert list(matrix["synthetic"]) == list(expected["synthetic"])


def test_feature_columns_are_the_extractor_output_in_order(tmp_path):
    data_dir = make_split_run(tmp_path)

    build(data_dir, now=NOW)

    for name in ("train", "test"):
        matrix = read_matrix(data_dir, name)
        assert list(matrix.columns) == [*METADATA_COLUMNS, *AUDIT_COLUMNS, *FEATURE_NAMES]
        for record in matrix.to_dict("records"):
            found = tuple(record[feature] for feature in FEATURE_NAMES)
            assert found == pytest.approx(extract_features(record["url"]))


def test_metadata_and_audit_columns_stay_outside_the_feature_set(tmp_path):
    rows = sample_rows()
    data_dir = make_split_run(tmp_path, rows)

    build(data_dir, now=NOW)

    assert METADATA_COLUMNS == ("url", "label", "split", "group", "synthetic", "sources")
    assert AUDIT_COLUMNS == ("audit_scheme", "audit_has_www")
    assert not set(FEATURE_NAMES) & {*METADATA_COLUMNS, *AUDIT_COLUMNS}
    sources = {item["url"]: item["sources"] for item in rows}
    matrix = pd.concat([read_matrix(data_dir, "train"), read_matrix(data_dir, "test")])
    for record in matrix.to_dict("records"):
        parts = parse_url(record["url"])
        assert list(record["sources"]) == sources[record["url"]]
        assert record["audit_scheme"] == parts.scheme
        assert record["audit_has_www"] == parts.host.startswith("www.")
    assert set(matrix["audit_scheme"]) == {"http", "https"}
    assert set(matrix["audit_has_www"]) == {True, False}


def test_report_traces_the_inputs_the_code_and_the_feature_set(tmp_path):
    data_dir = make_split_run(tmp_path)
    split_report = json.loads(
        (data_dir / "processed" / SPLIT_STAMP / "split_report.json").read_text(encoding="utf-8")
    )

    returned = build(data_dir, now=NOW, command="python -m sereno_model.features.build")

    report = read_report(data_dir)
    assert report == returned
    assert report["created_at"] == "2026-11-14T09:00:00Z"
    assert report["output_dir"] == f"processed/{STAMP}"
    assert report["dry_run"] is False
    assert report["command"] == "python -m sereno_model.features.build"
    assert report["split_run"] == {
        "dir": f"processed/{SPLIT_STAMP}",
        "partition_hash": split_report["partition_hash"]["value"],
        "seed": 7,
    }
    assert report["assemble_run"] == split_report["assemble_run"]
    assert report["feature_set"]["version"] == FEATURE_SET_VERSION
    assert report["feature_set"]["names"] == list(FEATURE_NAMES)
    assert report["columns"] == {
        "metadata": list(METADATA_COLUMNS),
        "audit": list(AUDIT_COLUMNS),
        "features": list(FEATURE_NAMES),
        "note": report["columns"]["note"],
    }
    assert "source-bias control" in report["columns"]["note"]
    assert set(report["git"]) == {"commit", "clean"}
    assert set(report["machine"]) == {"platform", "processor", "logical_cpus"}
    assert report["duration_seconds"] >= 0
    versions = report["versions"]
    assert set(versions) == {
        "python",
        "pandas",
        "pyarrow",
        "numpy",
        "tldextract",
        "public_suffix_list_snapshot_sha256",
    }
    assert versions["pandas"] == pd.__version__
    assert versions["public_suffix_list_snapshot_sha256"] == suffix_list_snapshot_id()


def test_report_counts_rows_per_split_and_class_and_hashes_each_matrix(tmp_path):
    data_dir = make_split_run(tmp_path)
    partition = read_split(data_dir)

    report = build(data_dir, now=NOW)

    for name in ("train", "test"):
        part = partition[partition["split"] == name]
        found = report["splits"][name]
        assert found["rows"] == len(part)
        assert found["phishing"] == int((part["label"] == PHISHING).sum())
        assert found["legitimate"] == int((part["label"] == LEGITIMATE).sum())
        assert found["file"] == FILES[name]
        assert found["content_hash"]["algorithm"] == "sha256"
        assert found["content_hash"]["value"] == matrix_hash(read_matrix(data_dir, name))
    hashes = {report["splits"][name]["content_hash"]["value"] for name in ("train", "test")}
    assert len(hashes) == 2


def test_matrix_hash_changes_with_a_value_and_ignores_the_row_order(tmp_path):
    data_dir = make_split_run(tmp_path)
    build(data_dir, now=NOW)
    matrix = read_matrix(data_dir, "train")

    shuffled = matrix.iloc[::-1].reset_index(drop=True)
    changed = matrix.copy()
    changed.loc[0, "url_length"] += 1

    assert matrix_hash(shuffled) == matrix_hash(matrix)
    assert matrix_hash(changed) != matrix_hash(matrix)


def test_statistics_describe_each_feature_per_class_on_train_only(tmp_path):
    data_dir = make_split_run(tmp_path)

    report = build(data_dir, now=NOW)

    train = read_matrix(data_dir, "train")
    statistics = report["train_statistics"]
    assert statistics["split"] == "train"
    assert set(statistics["by_class"]) == {"phishing", "legitimate"}
    for kind, label in (("phishing", PHISHING), ("legitimate", LEGITIMATE)):
        part = train[train["label"] == label]
        found = statistics["by_class"][kind]
        assert found["rows"] == len(part)
        assert list(found["features"]) == list(FEATURE_NAMES)
        for feature in FEATURE_NAMES:
            assert found["features"][feature] == {
                "min": pytest.approx(part[feature].min(), abs=1e-6),
                "median": pytest.approx(part[feature].median(), abs=1e-6),
                "max": pytest.approx(part[feature].max(), abs=1e-6),
                "nonzero_share": pytest.approx((part[feature] != 0).mean(), abs=1e-6),
            }
    phishing = statistics["by_class"]["phishing"]["features"]
    assert phishing["url_length"]["nonzero_share"] == 1.0
    assert statistics["by_class"]["legitimate"]["features"]["host_is_ip"]["max"] == 0


def test_report_measures_the_extraction_time(tmp_path):
    data_dir = make_split_run(tmp_path)

    report = build(data_dir, now=NOW)

    extraction = report["extraction"]
    assert extraction["rows"] == len(read_split(data_dir))
    assert extraction["seconds"] > 0
    assert extraction["rows_per_second"] > 0
    assert extraction["mean_ms_per_url"] == pytest.approx(
        1000 * extraction["seconds"] / extraction["rows"], rel=1e-3
    )


def test_refuses_a_partition_that_does_not_match_its_recorded_hash(tmp_path):
    data_dir = make_split_run(tmp_path)
    path = data_dir / "processed" / SPLIT_STAMP / "split.parquet"
    tampered = pd.read_parquet(path)
    tampered.loc[0, "split"] = "test" if tampered.loc[0, "split"] == "train" else "train"
    write_parquet(tampered, SPLIT_SCHEMA, path)

    with pytest.raises(ChecksumMismatchError, match="partition hash"):
        build(data_dir, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def tamper_split(data_dir: Path, column: str, value: object) -> None:
    """Change one cell of the split file that the partition hash does not cover."""
    path = data_dir / "processed" / SPLIT_STAMP / "split.parquet"
    tampered = pd.read_parquet(path)
    tampered.loc[0, column] = value(tampered.loc[0, column]) if callable(value) else value
    write_parquet(tampered, SPLIT_SCHEMA, path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("label", lambda label: 1 - label),
        ("synthetic", lambda synthetic: not synthetic),
        ("group", "another-domain.example"),
    ],
)
def test_refuses_a_partition_whose_labels_or_groups_were_altered(tmp_path, column, value):
    # The partition hash only covers the URL and its split, so these columns need their own check.
    data_dir = make_split_run(tmp_path)
    tamper_split(data_dir, column, value)

    with pytest.raises(ChecksumMismatchError, match=column):
        build(data_dir, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def test_refuses_an_assembled_dataset_that_does_not_match_its_recorded_hash(tmp_path):
    data_dir = make_split_run(tmp_path)
    path = data_dir / "processed" / ASSEMBLE_STAMP / "dataset.parquet"
    tampered = pd.read_parquet(path)
    tampered.loc[0, "label"] = 1 - tampered.loc[0, "label"]
    write_parquet(tampered, DATASET_SCHEMA, path)

    with pytest.raises(ChecksumMismatchError, match="content hash"):
        build(data_dir, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def test_never_overwrites_an_existing_run(tmp_path):
    data_dir = make_split_run(tmp_path)
    build(data_dir, now=NOW)
    before = (data_dir / "processed" / STAMP / REPORT).read_bytes()

    with pytest.raises(FileExistsError):
        build(data_dir, now=NOW)

    assert (data_dir / "processed" / STAMP / REPORT).read_bytes() == before


def test_same_partition_gives_the_same_matrices(tmp_path):
    data_dir = make_split_run(tmp_path)

    first = build(data_dir, split_run=SPLIT_STAMP, now=NOW)
    again = build(data_dir, split_run=SPLIT_STAMP, now=LATER)

    for name in ("train", "test"):
        assert first["splits"][name]["content_hash"] == again["splits"][name]["content_hash"]
        assert read_matrix(data_dir, name).equals(read_matrix(data_dir, name, LATER_STAMP))
    assert first["train_statistics"] == again["train_statistics"]


def test_uses_the_newest_split_run_unless_one_is_named(tmp_path):
    data_dir = make_split_run(tmp_path)
    newer = datetime(2026, 11, 13, 12, 0, 0, tzinfo=UTC)
    split(data_dir, assemble_run=ASSEMBLE_STAMP, seed=8, now=newer)

    latest = build(data_dir, now=NOW)
    named = build(data_dir, split_run=SPLIT_STAMP, now=LATER)

    assert latest["split_run"]["dir"] == "processed/20261113T120000Z"
    assert latest["split_run"]["seed"] == 8
    assert named["split_run"]["dir"] == f"processed/{SPLIT_STAMP}"


def test_fails_clearly_without_a_split_run(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)

    with pytest.raises(LookupError, match="no split run"):
        build(tmp_path / "data", now=NOW)


def test_dry_run_reports_without_writing(tmp_path):
    data_dir = make_split_run(tmp_path)
    before = sorted(path.name for path in (data_dir / "processed").iterdir())

    report = build(data_dir, now=NOW, dry_run=True)

    assert sorted(path.name for path in (data_dir / "processed").iterdir()) == before
    assert report["dry_run"] is True
    assert report["output_dir"] is None
    assert report["splits"]["train"]["rows"] + report["splits"]["test"]["rows"] == len(
        read_split(data_dir)
    )
    written = build(data_dir, now=NOW)
    for name in ("train", "test"):
        assert report["splits"][name]["content_hash"] == written["splits"][name]["content_hash"]


def test_cli_builds_the_matrices_and_prints_a_summary(tmp_path, capsys):
    data_dir = make_split_run(tmp_path)

    main(["--data-dir", str(data_dir), "--split-run", SPLIT_STAMP])

    reports = list((data_dir / "processed").glob(f"*/{REPORT}"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["command"].endswith(f"--split-run {SPLIT_STAMP}")
    output = capsys.readouterr().out
    assert report["splits"]["train"]["content_hash"]["value"] in output
    assert report["splits"]["test"]["content_hash"]["value"] in output
    assert "train" in output and "test" in output
    assert "ms per URL" in output
    assert REPORT in output
    assert "http://" not in output and "https://" not in output


def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    data_dir = make_split_run(tmp_path)

    main(["--data-dir", str(data_dir), "--dry-run"])

    assert not list((data_dir / "processed").glob(f"*/{REPORT}"))
    assert "dry run" in capsys.readouterr().out


def test_git_state_is_unknown_outside_a_repository(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "clean": None}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_state_reports_the_commit_and_whether_the_tree_is_clean(tmp_path):
    def git(*arguments: str) -> None:
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        options = [*identity, "-c", "commit.gpgsign=false", "-c", "core.hooksPath=none"]
        command = ["git", "-C", str(tmp_path), *options, *arguments]
        subprocess.run(command, check=True, capture_output=True)

    git("init")
    (tmp_path / "tracked.txt").write_text("one\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "chore: first")

    clean = git_state(tmp_path)
    (tmp_path / "tracked.txt").write_text("two\n", encoding="utf-8")
    dirty = git_state(tmp_path)

    assert len(clean["commit"]) == 40 and clean["clean"] is True
    assert dirty == {"commit": clean["commit"], "clean": False}

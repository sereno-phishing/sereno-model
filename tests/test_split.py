import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from sereno_model.data import split as split_module
from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.assemble import DATASET_SCHEMA, dataset_hash
from sereno_model.data.clean import write_parquet
from sereno_model.data.domains import suffix_list_snapshot_id
from sereno_model.data.normalize import parse_url
from sereno_model.data.split import LeakageError, leakage_controls, main, split

NOW = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 11, 13, 10, 0, 0, tzinfo=UTC)
STAMP = "20261113T090000Z"
LATER_STAMP = "20261113T100000Z"
ASSEMBLE_STAMP = "20261112T090000Z"

PHISHING, LEGITIMATE = 1, 0
CONTROLS = [
    "duplicates",
    "domains",
    "form_bias",
    "source_bias",
    "temporal",
    "importance",
    "stability",
]


def row(
    url: str,
    label: int,
    sources: tuple[str, ...] | None = None,
    synthetic: bool = False,
    submission_time: str | None = None,
) -> dict:
    parts = parse_url(url)
    default = ("phishtank",) if label == PHISHING else ("url_phish_2026",)
    return {
        "url": url,
        "label": label,
        "sources": list(sources or default),
        "has_path_or_query": parts.has_path_or_query,
        "scheme": parts.scheme,
        "has_www": parts.host.startswith("www."),
        "synthetic": synthetic,
        "submission_time": submission_time,
    }


def sample_rows(domains: int = 20) -> list[dict]:
    """Two classes over many small domains, one shared platform, one IP and synthetic rows."""
    rows = []
    for index in range(domains):
        moment = f"2026-{index % 12 + 1:02d}-15T08:00:00+00:00"
        rows += [
            row(f"https://bad{index}.com/login", PHISHING, submission_time=moment),
            row(f"https://secure.bad{index}.com/verify?id=1", PHISHING, submission_time=moment),
            row(f"http://www.bad{index}.com", PHISHING, ("hannousse_2021",)),
            row(f"https://www.good{index}.org/about", LEGITIMATE),
            row(f"http://good{index}.org", LEGITIMATE, ("hannousse_2021", "url_phish_2026")),
            row(f"https://top{index}.net", LEGITIMATE, ("majestic_million",), synthetic=True),
        ]
    rows += [row(f"https://victim{index}.pages.dev/pay", PHISHING) for index in range(7)]
    rows += [row("https://docs.pages.dev/guide", LEGITIMATE)]
    rows += [row(f"http://203.0.113.9/panel/{index}", PHISHING) for index in range(3)]
    return rows


def make_assemble_run(root: Path, rows: list[dict], stamp: str = ASSEMBLE_STAMP) -> Path:
    """Write a tiny assembled dataset with its report and return the data dir."""
    frame = pd.DataFrame(sorted(rows, key=lambda item: item["url"]), columns=DATASET_SCHEMA.names)
    frame = frame.astype(
        {"label": "int8", "has_path_or_query": "bool", "has_www": "bool", "synthetic": "bool"}
    )
    data_dir = root / "data"
    run_dir = data_dir / "processed" / stamp
    run_dir.mkdir(parents=True)
    write_parquet(frame, DATASET_SCHEMA, run_dir / "dataset.parquet")
    report = {"output_dir": f"processed/{stamp}", "content_hash": {"value": dataset_hash(frame)}}
    (run_dir / "assemble_report.json").write_text(json.dumps(report), encoding="utf-8")
    return data_dir


def read_split(data_dir: Path, stamp: str = STAMP) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / stamp / "split.parquet")


def controls(report: dict) -> dict[str, dict]:
    return {entry["control"]: entry for entry in report["leakage_controls"]}


def leaky_frame() -> pd.DataFrame:
    """One domain on both sides, including the same normalized URL written in two ways."""
    rows = [
        {**row("https://bad.com/a", PHISHING), "split": "train", "group": "bad.com"},
        {**row("https://BAD.com/a", PHISHING), "split": "test", "group": "bad.com"},
        {**row("https://good.org/", LEGITIMATE), "split": "train", "group": "good.org"},
        {**row("https://fine.org/", LEGITIMATE), "split": "test", "group": "fine.org"},
    ]
    return pd.DataFrame(rows)


def test_assigns_every_row_exactly_once(tmp_path):
    rows = sample_rows()
    data_dir = make_assemble_run(tmp_path, rows)

    split(data_dir, now=NOW)

    frame = read_split(data_dir)
    assert len(frame) == len(rows)
    assert sorted(frame["url"]) == sorted(item["url"] for item in rows)
    assert set(frame["split"]) == {"train", "test"}
    labels = {item["url"]: item["label"] for item in rows}
    assert all(labels[url] == label for url, label in zip(frame["url"], frame["label"]))


def test_no_group_and_no_url_appears_in_both_splits(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    split(data_dir, now=NOW)

    frame = read_split(data_dir)
    train, test = frame[frame["split"] == "train"], frame[frame["split"] == "test"]
    assert not set(train["group"]) & set(test["group"])
    assert not set(train["url"]) & set(test["url"])


def test_groups_rows_by_registered_domain(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    split(data_dir, now=NOW)

    groups = dict(zip(*(read_split(data_dir)[name] for name in ("url", "group"))))
    assert groups["https://bad3.com/login"] == "bad3.com"
    assert groups["https://secure.bad3.com/verify?id=1"] == "bad3.com"
    assert groups["http://www.bad3.com"] == "bad3.com"
    assert groups["https://victim0.pages.dev/pay"] == "pages.dev"
    assert groups["https://docs.pages.dev/guide"] == "pages.dev"
    assert groups["http://203.0.113.9/panel/0"] == "203.0.113.9"


def test_test_set_is_about_one_fold_and_keeps_the_class_proportions(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows(domains=60))

    report = split(data_dir, now=NOW)

    train, test = report["splits"]["train"], report["splits"]["test"]
    assert 0.15 <= test["share_of_rows"] <= 0.25
    assert train["rows"] + test["rows"] == 60 * 6 + 11
    assert abs(train["phishing_share"] - test["phishing_share"]) < 0.05
    assert report["parameters"]["n_splits"] == 5


def test_n_splits_sets_the_size_of_the_test_set(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows(domains=60))

    report = split(data_dir, n_splits=2, now=NOW)

    assert report["parameters"]["n_splits"] == 2
    assert 0.4 <= report["splits"]["test"]["share_of_rows"] <= 0.6


def test_rejects_fewer_than_two_splits(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    with pytest.raises(ValueError, match="n_splits"):
        split(data_dir, n_splits=1, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def test_same_inputs_and_seed_give_the_same_partition(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    first = split(data_dir, seed=7, now=NOW)
    second = split(data_dir, seed=7, now=LATER)

    assert first["partition_hash"]["value"] == second["partition_hash"]["value"]
    assert read_split(data_dir).equals(read_split(data_dir, LATER_STAMP))


def test_another_seed_gives_another_partition(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    first = split(data_dir, seed=7, now=NOW)
    second = split(data_dir, seed=8, now=LATER)

    assert first["partition_hash"]["value"] != second["partition_hash"]["value"]
    assert (first["parameters"]["seed"], second["parameters"]["seed"]) == (7, 8)


def test_partition_hash_depends_on_the_content_not_on_the_parquet_encoding(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    report = split(data_dir, now=NOW)

    frame = read_split(data_dir)
    lines = sorted(f"{url}\t{name}\n" for url, name in zip(frame["url"], frame["split"]))
    expected = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert report["partition_hash"]["value"] == expected
    assert report["partition_hash"]["algorithm"] == "sha256"


def test_stops_when_the_dataset_does_not_match_its_recorded_hash(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())
    run_dir = data_dir / "processed" / ASSEMBLE_STAMP
    recorded = json.loads((run_dir / "assemble_report.json").read_text(encoding="utf-8"))
    recorded["content_hash"]["value"] = "0" * 64
    (run_dir / "assemble_report.json").write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(ChecksumMismatchError, match="dataset.parquet"):
        split(data_dir, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def test_uses_the_newest_assemble_run_unless_one_is_named(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows(domains=10), stamp="20261110T000000Z")
    make_assemble_run(tmp_path, sample_rows(domains=12), stamp=ASSEMBLE_STAMP)

    newest = split(data_dir, now=NOW)
    named = split(data_dir, assemble_run="20261110T000000Z", now=LATER)

    assert newest["assemble_run"]["dir"] == f"processed/{ASSEMBLE_STAMP}"
    assert named["assemble_run"]["dir"] == "processed/20261110T000000Z"
    assert len(read_split(data_dir)) == 12 * 6 + 11
    assert len(read_split(data_dir, LATER_STAMP)) == 10 * 6 + 11


def test_fails_without_an_assemble_run(tmp_path):
    with pytest.raises(LookupError, match="no assemble run"):
        split(tmp_path / "data", now=NOW)


def test_never_overwrites_an_existing_output_folder(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())
    split(data_dir, now=NOW)

    with pytest.raises(FileExistsError):
        split(data_dir, now=NOW)


def test_writes_expected_columns_types_and_order(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    split(data_dir, now=NOW)

    frame = read_split(data_dir)
    assert list(frame.columns) == ["url", "label", "split", "group", "synthetic"]
    assert list(frame["url"]) == sorted(frame["url"])
    schema = pq.read_schema(data_dir / "processed" / STAMP / "split.parquet")
    types = {name: str(schema.field(name).type) for name in schema.names}
    assert types == {
        "url": "string",
        "label": "int8",
        "split": "string",
        "group": "string",
        "synthetic": "bool",
    }
    assert int(frame["synthetic"].sum()) == 20


def test_report_describes_the_run_and_each_split(tmp_path):
    rows = sample_rows()
    data_dir = make_assemble_run(tmp_path, rows)

    report = split(data_dir, now=NOW)

    saved = data_dir / "processed" / STAMP / "split_report.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == report
    assert report["created_at"] == "2026-11-13T09:00:00Z"
    assert report["output_dir"] == f"processed/{STAMP}"
    frame = pd.DataFrame(rows, columns=DATASET_SCHEMA.names)
    assert report["assemble_run"] == {
        "dir": f"processed/{ASSEMBLE_STAMP}",
        "content_hash": dataset_hash(frame),
    }
    parameters = report["parameters"]
    assert (parameters["seed"], parameters["n_splits"]) == (20261111, 5)
    assert parameters["test_fold"] == 0
    assert "StratifiedGroupKFold" in parameters["splitter"]

    written = read_split(data_dir)
    for name in ("train", "test"):
        part = written[written["split"] == name]
        summary = report["splits"][name]
        assert summary["rows"] == len(part)
        assert summary["phishing"] == int((part["label"] == PHISHING).sum())
        assert summary["legitimate"] == int((part["label"] == LEGITIMATE).sum())
        assert summary["groups"] == part["group"].nunique()
        assert summary["synthetic"] == int(part["synthetic"].sum())
        assert summary["synthetic_share"] == round(summary["synthetic"] / len(part), 6)
    assert report["groups"]["total"] == written["group"].nunique()
    assert report["groups"]["total"] == sum(
        report["splits"][name]["groups"] for name in ("train", "test")
    )

    versions = report["versions"]
    assert {"python", "pandas", "pyarrow", "scikit_learn", "tldextract"} <= set(versions)
    assert versions["public_suffix_list_snapshot_sha256"] == suffix_list_snapshot_id()


def test_report_lists_the_largest_groups_with_their_split(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    report = split(data_dir, now=NOW)

    largest = report["groups"]["largest"]
    assert len(largest) == 10
    written = read_split(data_dir)
    side = written.loc[written["group"] == "pages.dev", "split"].iloc[0]
    assert largest[0] == {
        "group": "pages.dev",
        "rows": 8,
        "phishing": 7,
        "legitimate": 1,
        "split": side,
    }
    assert [entry["rows"] for entry in largest] == sorted(
        (entry["rows"] for entry in largest), reverse=True
    )
    assert report["groups"]["with_both_classes"] == 1


def test_report_counts_sources_and_form_strata_per_split_and_class(tmp_path):
    rows = sample_rows()
    data_dir = make_assemble_run(tmp_path, rows)

    report = split(data_dir, now=NOW)

    sources, strata = report["sources"], report["form_strata"]
    total = {
        name: sum(sources[part][kind].get(name, 0) for part in sources for kind in sources[part])
        for name in ("phishtank", "hannousse_2021", "url_phish_2026", "majestic_million")
    }
    # A URL listed by two sources counts once in each of them.
    assert total == {
        "phishtank": 50,
        "hannousse_2021": 40,
        "url_phish_2026": 41,
        "majestic_million": 20,
    }
    assert set(sources["train"]) == set(strata["test"]) == {"phishing", "legitimate"}
    assert len(strata["train"]["phishing"]) == 8
    for name in ("train", "test"):
        for kind in ("phishing", "legitimate"):
            assert sum(strata[name][kind].values()) == report["splits"][name][kind]
    no_path_http_www = sum(strata[name]["phishing"]["no_path/http/www"] for name in strata)
    assert no_path_http_www == 20


def test_report_gives_the_phishing_date_range_and_says_legitimate_has_none(tmp_path):
    rows = [
        row(f"https://bad{index}.com/a", PHISHING, submission_time=f"2026-0{index + 1}-01T00:00:00+00:00")
        for index in range(6)
    ]
    rows += [row(f"https://old{index}.com/a", PHISHING) for index in range(4)]
    rows += [row(f"https://good{index}.org/a", LEGITIMATE) for index in range(10)]
    data_dir = make_assemble_run(tmp_path, rows)

    report = split(data_dir, n_splits=2, now=NOW)

    written = read_split(data_dir)
    dates = {item["url"]: item["submission_time"] for item in rows}
    times = report["submission_time"]
    assert "legitimate" in times["note"] and "no" in times["note"]
    for name in ("train", "test"):
        urls = written.loc[(written["split"] == name) & (written["label"] == PHISHING), "url"]
        known = sorted(dates[url] for url in urls if dates[url])
        found = times["phishing"][name]
        assert found["with_date"] == len(known)
        assert found["without_date"] == len(urls) - len(known)
        assert found["min"] == (known[0] if known else None)
        assert found["max"] == (known[-1] if known else None)
        assert found["median"] == (known[(len(known) - 1) // 2] if known else None)


def test_leakage_controls_follow_the_decision_gates_of_the_skill(tmp_path):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    report = split(data_dir, now=NOW)

    assert [entry["control"] for entry in report["leakage_controls"]] == CONTROLS
    found = controls(report)
    assert {name: entry["status"] for name, entry in found.items()} == {
        "duplicates": "pass",
        "domains": "pass",
        "form_bias": "reported",
        "source_bias": "reported",
        "temporal": "reported",
        "importance": "pending_model",
        "stability": "pending_model",
    }
    assert found["duplicates"]["evidence"] == {"normalized_urls_in_both_splits": 0}
    assert found["domains"]["evidence"] == {"groups_in_both_splits": 0}
    for name in ("train", "test"):
        distances = found["form_bias"]["evidence"][name]
        assert set(distances) == {"all_strata", "with_path", "without_path"}
        assert 0 <= distances["with_path"] <= 1
    assert found["temporal"]["evidence"] == report["submission_time"]
    assert all(entry["fails_if"] for entry in report["leakage_controls"])


def test_leakage_controls_fail_when_a_url_or_a_domain_is_on_both_sides():
    found = {entry["control"]: entry for entry in leakage_controls(leaky_frame())}

    assert found["duplicates"]["status"] == "fail"
    assert found["duplicates"]["evidence"] == {"normalized_urls_in_both_splits": 1}
    assert found["domains"]["status"] == "fail"
    assert found["domains"]["evidence"] == {"groups_in_both_splits": 1}


def test_writes_nothing_when_a_leakage_control_fails(tmp_path, monkeypatch):
    data_dir = make_assemble_run(tmp_path, sample_rows())
    alternate = lambda labels, groups, seed, n_splits: [i % 2 == 0 for i in range(len(labels))]
    monkeypatch.setattr(split_module, "held_out_mask", alternate)

    with pytest.raises(LeakageError, match="domains"):
        split(data_dir, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def test_cli_runs_the_split_and_prints_a_summary(tmp_path, capsys):
    data_dir = make_assemble_run(tmp_path, sample_rows())

    main(
        [
            "--data-dir",
            str(data_dir),
            "--assemble-run",
            ASSEMBLE_STAMP,
            "--seed",
            "7",
            "--n-splits",
            "4",
        ]
    )

    reports = list((data_dir / "processed").glob("*/split_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert (report["parameters"]["seed"], report["parameters"]["n_splits"]) == (7, 4)
    output = capsys.readouterr().out
    assert report["partition_hash"]["value"] in output
    assert "train" in output and "test" in output
    assert "duplicates" in output and "pass" in output

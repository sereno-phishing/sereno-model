import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_features_build import make_split_run, row

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.clean import write_parquet
from sereno_model.features.build import MATRIX_SCHEMA, build, matrix_hash
from sereno_model.features.extract import FEATURE_NAMES
from sereno_model.features.source_bias import (
    DEFAULT_SEED,
    N_ESTIMATORS,
    REPORT_NAME,
    grouped_folds,
    main,
    probe_target,
    run_control,
    verdict_for,
)

FEATURES_NOW = datetime(2026, 11, 14, 9, 0, 0, tzinfo=UTC)
FEATURES_STAMP = "20261114T090000Z"
NOW = datetime(2026, 11, 15, 9, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 11, 15, 10, 0, 0, tzinfo=UTC)
STAMP = "20261115T090000Z"

PHISHING, LEGITIMATE = 1, 0
# Few trees keep the tests fast; the command line keeps the documented 200.
TREES = 15
SEED = 11


def fixture_rows() -> list[dict]:
    """Four strata: one that leaks, one with no relation and two too small to evaluate.

    Phishing with a path: every https URL carries a tilde and comes from PhishTank, every http
    one has no tilde and comes from another source, so one feature gives away both the scheme
    and the source. Legitimate with a path: the scheme alternates over URLs of the same shape.
    The ``www.`` prefix follows its own pattern in both, unrelated to anything measured.
    """
    rows = []
    for index in range(100, 500):
        secure, www = index % 2 == 0, "www." if index % 4 < 2 else ""
        if secure:
            rows.append(row(f"https://{www}ph{index}.com/~u/page", PHISHING, ("phishtank",)))
        else:
            rows.append(row(f"http://{www}ph{index}.com/xu/page", PHISHING, ("hannousse_2021",)))
        scheme = "https" if secure else "http"
        rows.append(
            row(f"{scheme}://{www}lg{index}.org/about/team", LEGITIMATE, ("url_phish_2026",))
        )
    for index in range(100, 112):
        rows.append(row(f"https://top{index}.net", LEGITIMATE, ("majestic_million",)))
    for index in range(100, 106):
        rows.append(row(f"http://bare{index}.com", PHISHING, ("phishtank",)))
    return rows


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Path:
    """A features run built once with the real splitter and the real matrix builder."""
    root = tmp_path_factory.mktemp("source_bias")
    data_dir = make_split_run(root, fixture_rows())
    build(data_dir, now=FEATURES_NOW)
    return data_dir


@pytest.fixture
def data_dir(template, tmp_path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(template, target)
    return target


@pytest.fixture(scope="module")
def report(template) -> dict:
    """One dry run over the untouched template, shared by the read-only tests."""
    return run_control(template, seed=SEED, n_estimators=TREES, now=NOW, dry_run=True)


def stratum(report: dict, name: str) -> dict:
    return next(entry for entry in report["strata"] if entry["stratum"] == name)


def target(report: dict, stratum_name: str, target_name: str) -> dict:
    targets = stratum(report, stratum_name)["targets"]
    return next(entry for entry in targets if entry["target"] == target_name)


def train_path(data_dir: Path) -> Path:
    return data_dir / "processed" / FEATURES_STAMP / "features_train.parquet"


def held_out_matrix_path(data_dir: Path) -> Path:
    return data_dir / "processed" / FEATURES_STAMP / "features_test.parquet"


def test_a_feature_that_leaks_the_scheme_fails_and_is_named_first(report):
    found = target(report, "phishing/path", "scheme")

    assert found["control"] == "A"
    assert found["status"] == "evaluated"
    assert found["oof_auc"] > 0.95
    assert found["verdict"] == "fail"
    assert len(found["fold_aucs"]) == 5
    ranking = found["feature_separability"]
    assert ranking[0] == {"feature": "tilde_count", "separability": 1.0}
    assert sorted(entry["feature"] for entry in ranking) == sorted(FEATURE_NAMES)
    values = [entry["separability"] for entry in ranking]
    assert values == sorted(values, reverse=True)
    assert all(0.5 <= value <= 1.0 for value in values)
    assert report["controls"]["A"]["verdict"] == "fail"


def test_a_feature_that_leaks_the_source_fails_the_source_control(report):
    names = [entry["target"] for entry in stratum(report, "phishing/path")["targets"]]
    assert names == ["scheme", "www", "source:hannousse_2021", "source:phishtank"]

    found = target(report, "phishing/path", "source:phishtank")
    assert found["control"] == "C"
    assert found["verdict"] == "fail"
    assert found["feature_separability"][0]["feature"] == "tilde_count"
    assert report["controls"]["C"]["verdict"] == "fail"


def test_no_relation_between_the_vector_and_the_target_passes(report):
    scheme = target(report, "legitimate/path", "scheme")
    www = target(report, "legitimate/path", "www")

    for found in (scheme, www, target(report, "phishing/path", "www")):
        assert found["status"] == "evaluated"
        assert found["oof_auc"] < 0.65
        assert found["verdict"] == "pass"
    # B has no failing target in any stratum, so its overall verdict is the worst: pass.
    assert report["controls"]["B"]["verdict"] == "pass"
    assert report["controls"]["B"]["evaluated_targets"] == 2


def test_counts_rows_positives_and_groups_per_stratum_and_target(report, template):
    train = pd.read_parquet(train_path(template))
    part = train[(train["label"] == PHISHING) & train["url"].str.contains("/page")]

    found = stratum(report, "phishing/path")
    assert [entry["stratum"] for entry in report["strata"]] == [
        "legitimate/bare",
        "legitimate/path",
        "phishing/bare",
        "phishing/path",
    ]
    assert (found["class"], found["shape"]) == ("phishing", "path")
    assert found["rows"] == len(part)
    assert found["groups"] == part["group"].nunique()
    scheme = target(report, "phishing/path", "scheme")
    secure = part[part["audit_scheme"] == "https"]
    assert scheme["rows"] == len(part)
    assert scheme["positives"] == len(secure)
    assert scheme["negatives"] == len(part) - len(secure)
    assert scheme["positive_groups"] == secure["group"].nunique()
    assert scheme["negative_groups"] == part["group"].nunique() - secure["group"].nunique()


def test_small_or_one_sided_targets_are_declared_not_evaluable(report):
    few_rows = target(report, "legitimate/bare", "scheme")
    one_sided = target(report, "legitimate/path", "source:url_phish_2026")

    for found in (few_rows, one_sided):
        assert found["status"] == "not_evaluable"
        assert found["verdict"] == "not_evaluable"
        assert "fewer than 100 rows" in found["reason"]
        assert found["oof_auc"] is None
        assert found["fold_aucs"] is None
        assert found["feature_separability"] is None
        assert found["positives"] + found["negatives"] == found["rows"]
    assert few_rows["negatives"] == 0
    assert one_sided["negatives"] == 0


def test_a_target_held_by_too_few_groups_is_not_evaluable():
    # 120 positives packed in 3 domains: enough rows, not enough independent groups.
    labels = np.array([True] * 120 + [False] * 120)
    groups = np.array([f"big{index % 3}" for index in range(120)] + [f"g{i}" for i in range(120)])
    features = np.random.default_rng(0).normal(size=(240, 3))

    found = probe_target(features, labels, groups, ("a", "b", "c"), seed=SEED, n_estimators=TREES)

    assert found["status"] == "not_evaluable"
    assert "fewer than 5 distinct groups" in found["reason"]
    assert (found["positive_groups"], found["negative_groups"]) == (3, 120)
    assert found["oof_auc"] is None


def test_a_control_without_evaluable_targets_is_not_evaluable(report):
    # Every source target of the fixture that can be evaluated lies in one stratum; B and C
    # ignore the targets that cannot when taking the worst verdict.
    control = report["controls"]["C"]
    assert control["evaluated_targets"] == 2
    assert control["not_evaluable_targets"] >= 1
    assert verdict_for(None) == "not_evaluable"


@pytest.mark.parametrize(
    ("auc", "expected"),
    [
        (0.5, "pass"),
        (0.6499, "pass"),
        (0.65, "review"),
        (0.80, "review"),
        (0.8001, "fail"),
        (1.0, "fail"),
    ],
)
def test_verdict_bands_are_fixed(auc, expected, report):
    assert verdict_for(auc) == expected
    assert report["parameters"]["verdict_bands"] == {
        "pass": "oof_auc < 0.65",
        "review": "0.65 <= oof_auc <= 0.80",
        "fail": "oof_auc > 0.80",
        "pass_below": 0.65,
        "fail_above": 0.80,
        "control_verdict": "the worst verdict of its targets, ignoring not_evaluable",
    }


def test_folds_never_put_a_group_on_both_sides():
    rng = np.random.default_rng(3)
    groups = np.array([f"g{index // 4}" for index in range(400)])
    labels = rng.random(400) < 0.4

    folds = grouped_folds(labels, groups, seed=SEED)

    assert len(folds) == 5
    held_out = np.concatenate([test for _, test in folds])
    assert sorted(held_out) == list(range(400))
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def test_the_probe_cannot_score_by_memorizing_a_group():
    # The feature identifies the domain and the target is constant inside each domain. A probe
    # validated by rows would recall the domain and score near 1; grouped by domain it cannot.
    rng = np.random.default_rng(5)
    group_ids = np.repeat(np.arange(200), 5)
    by_group = rng.random(200) < 0.5
    features = group_ids.reshape(-1, 1).astype(float)
    groups = np.array([f"g{index}" for index in group_ids])

    found = probe_target(
        features, by_group[group_ids], groups, ("group_id",), seed=SEED, n_estimators=TREES
    )

    assert found["status"] == "evaluated"
    assert found["oof_auc"] < 0.65


def test_refuses_a_train_matrix_that_does_not_match_its_recorded_hash(data_dir):
    path = train_path(data_dir)
    tampered = pd.read_parquet(path)
    tampered.loc[0, "tilde_count"] += 1
    write_parquet(tampered, MATRIX_SCHEMA, path)

    with pytest.raises(ChecksumMismatchError, match="content hash"):
        run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("audit_scheme", lambda scheme: "http" if scheme == "https" else "https"),
        ("audit_has_www", lambda flag: not flag),
        ("group", "another-domain.example"),
    ],
)
def test_refuses_columns_the_matrix_hash_does_not_cover(data_dir, column, value):
    # The matrix hash covers the URL, the label and the features; these are derived again.
    path = train_path(data_dir)
    tampered = pd.read_parquet(path)
    tampered.loc[0, column] = value(tampered.loc[0, column]) if callable(value) else value
    write_parquet(tampered, MATRIX_SCHEMA, path)

    with pytest.raises(ChecksumMismatchError, match=column):
        run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW)


@pytest.mark.parametrize("state", ["absent", "corrupt"])
def test_the_test_matrix_is_never_read(data_dir, report, state, monkeypatch):
    if state == "absent":
        held_out_matrix_path(data_dir).unlink()
    else:
        held_out_matrix_path(data_dir).write_bytes(b"not a parquet file")
    opened = []
    read_parquet = pd.read_parquet

    def recording_read(path, *args, **kwargs):
        opened.append(Path(path).name)
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)

    found = run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW, dry_run=True)

    assert opened == ["features_train.parquet"]
    assert found["strata"] == report["strata"]
    assert found["features_run"]["split"] == "train"


def test_same_seed_gives_the_same_report_content(data_dir, report):
    again = run_control(data_dir, seed=SEED, n_estimators=TREES, now=LATER, dry_run=True)
    other = run_control(data_dir, seed=SEED + 1, n_estimators=TREES, now=LATER, dry_run=True)

    for key in ("parameters", "features_run", "strata", "controls"):
        assert again[key] == report[key]
    assert other["strata"] != report["strata"]


def test_report_records_what_a_reproducible_run_needs(data_dir):
    features_report = json.loads(
        (data_dir / "processed" / FEATURES_STAMP / "features_report.json").read_text("utf-8")
    )

    returned = run_control(
        data_dir, seed=SEED, n_estimators=TREES, now=NOW, command="python -m x --data-dir data"
    )

    written = json.loads((data_dir / "processed" / STAMP / REPORT_NAME).read_text("utf-8"))
    assert written == returned
    assert written["created_at"] == "2026-11-15T09:00:00Z"
    assert written["output_dir"] == f"processed/{STAMP}"
    assert written["dry_run"] is False
    assert written["command"] == "python -m x --data-dir data"
    assert "exploration" in written["run_type"] and "not a model result" in written["run_type"]
    assert "source" in written["hypothesis"]
    assert set(written["git"]) == {"commit", "clean"}
    parameters = written["parameters"]
    assert (parameters["seed"], parameters["jobs"]) == (SEED, 1)
    assert parameters["probe"]["n_estimators"] == TREES
    assert parameters["probe"]["min_samples_leaf"] == 5
    assert parameters["probe"]["n_splits"] == 5
    assert parameters["evaluability"] == {"min_minority_rows": 100, "min_groups_per_side": 5}
    train = pd.read_parquet(train_path(data_dir))
    assert written["features_run"] == {
        "dir": f"processed/{FEATURES_STAMP}",
        "split": "train",
        "file": "features_train.parquet",
        "rows": len(train),
        "matrix_hash": matrix_hash(train),
        "partition_hash": features_report["split_run"]["partition_hash"],
        "feature_set_version": features_report["feature_set"]["version"],
        "test_matrix": "never opened",
    }
    assert written["feature_names"] == list(FEATURE_NAMES)
    assert set(written["machine"]) == {"platform", "processor", "logical_cpus"}
    assert {"python", "scikit-learn", "numpy", "pandas", "pyarrow"} <= set(written["versions"])
    assert written["duration_seconds"] >= 0
    assert "http://" not in json.dumps(written) and "https://" not in json.dumps(written)


def test_uses_the_newest_features_run_unless_one_is_named(data_dir):
    newer = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
    build(data_dir, now=newer)

    latest = run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW, dry_run=True)
    named = run_control(
        data_dir, FEATURES_STAMP, seed=SEED, n_estimators=TREES, now=NOW, dry_run=True
    )

    assert latest["features_run"]["dir"] == "processed/20261114T120000Z"
    assert named["features_run"]["dir"] == f"processed/{FEATURES_STAMP}"


def test_fails_clearly_without_a_features_run(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)

    with pytest.raises(LookupError, match="no features run"):
        run_control(tmp_path / "data", now=NOW)


def test_dry_run_writes_nothing(data_dir):
    before = sorted(path.name for path in (data_dir / "processed").iterdir())

    found = run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW, dry_run=True)

    assert sorted(path.name for path in (data_dir / "processed").iterdir()) == before
    assert found["dry_run"] is True
    assert found["output_dir"] is None


def test_never_overwrites_an_existing_run(data_dir):
    run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW)
    before = (data_dir / "processed" / STAMP / REPORT_NAME).read_bytes()

    with pytest.raises(FileExistsError):
        run_control(data_dir, seed=SEED, n_estimators=TREES, now=NOW)
    with pytest.raises(FileExistsError):
        # The folder of the features run itself is just as untouchable.
        run_control(data_dir, seed=SEED, n_estimators=TREES, now=FEATURES_NOW)

    assert (data_dir / "processed" / STAMP / REPORT_NAME).read_bytes() == before
    assert not (data_dir / "processed" / FEATURES_STAMP / REPORT_NAME).exists()


def test_defaults_are_the_documented_seed_and_probe_size():
    assert DEFAULT_SEED == 20261111
    assert N_ESTIMATORS == 200


def test_cli_dry_run_prints_a_table_and_writes_nothing(data_dir, capsys, monkeypatch):
    # The command line always grows the documented 200 trees; shrink them for the test only.
    monkeypatch.setattr("sereno_model.features.source_bias.N_ESTIMATORS", TREES)

    main(["--data-dir", str(data_dir), "--run", FEATURES_STAMP, "--seed", str(SEED), "--dry-run"])

    assert not list((data_dir / "processed").glob(f"*/{REPORT_NAME}"))
    output = capsys.readouterr().out
    assert "phishing/path" in output and "scheme" in output and "source:phishtank" in output
    assert "tilde_count" in output
    assert "fail" in output and "pass" in output and "not_evaluable" in output
    assert "A scheme: fail" in output
    assert "dry run" in output
    assert "http://" not in output and "https://" not in output


def test_cli_writes_the_report_and_records_the_command(data_dir, capsys, monkeypatch):
    monkeypatch.setattr("sereno_model.features.source_bias.N_ESTIMATORS", TREES)

    main(["--data-dir", str(data_dir), "--seed", str(SEED), "--jobs", "2"])

    reports = list((data_dir / "processed").glob(f"*/{REPORT_NAME}"))
    assert len(reports) == 1
    written = json.loads(reports[0].read_text(encoding="utf-8"))
    assert written["command"].startswith("python -m sereno_model.features.source_bias")
    assert written["command"].endswith("--jobs 2")
    assert (written["parameters"]["seed"], written["parameters"]["jobs"]) == (SEED, 2)
    assert REPORT_NAME in capsys.readouterr().out

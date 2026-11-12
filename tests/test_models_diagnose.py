import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from test_models_stacking import synthetic_matrix
from test_models_train import make_features_run

from sereno_model.models import diagnose as diagnose_module
from sereno_model.models.diagnose import (
    base_model_correlation,
    diagnose,
    make_folds,
    subgroup_breakdown,
    top_domains,
)

NOW = datetime(2026, 11, 16, 10, 0, 0, tzinfo=UTC)
FAST = {
    "n_iter": 2,
    "n_iter_wide": 1,
    "n_splits": 3,
    "n_jobs": 1,
    "threshold_grid": [0.3, 0.5, 0.7],
}
CAPPED = ["site1.com", "site3.com"]


def write_assemble_report(data_dir, capped=CAPPED):
    run_dir = data_dir / "processed" / "20261111T222725Z"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phishing_domain_cap": {
            "max_per_domain": 500,
            "capped_domains": [
                {"domain": domain, "rows_before": 900, "rows_kept": 500} for domain in capped
            ],
        }
    }
    (run_dir / "assemble_report.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnose")
    data_dir = make_features_run(root)
    write_assemble_report(data_dir)
    out_dir = root / "artifacts" / "diagnostics"
    report = diagnose(data_dir, out_dir, now=NOW, repo_dir=root, **FAST)
    return root, out_dir, report


def test_ungrouped_folds_ignore_the_groups_and_grouped_folds_keep_them_whole():
    frame = synthetic_matrix(rows=300, groups=30)
    labels, groups = frame["label"].to_numpy(), frame["group"].to_numpy()
    for grouped in (True, False):
        folds = make_folds(labels, groups, n_splits=5, seed=7, grouped=grouped)
        assert len(folds) == 5
        covered = np.concatenate([val for _, val in folds])
        assert sorted(covered) == list(range(300))
        for train, val in folds:
            assert abs(labels[val].mean() - labels.mean()) < 0.1
            split_groups = set(groups[train]) & set(groups[val])
            if grouped:
                assert not split_groups
        if not grouped:
            assert any(set(groups[train]) & set(groups[val]) for train, val in folds)


def test_ungrouped_folds_are_reproducible_with_the_seed():
    labels = np.array([0, 1] * 50)
    groups = np.arange(100)
    first = make_folds(labels, groups, n_splits=5, seed=3, grouped=False)
    second = make_folds(labels, groups, n_splits=5, seed=3, grouped=False)
    for (_, a), (_, b) in zip(first, second, strict=True):
        assert list(a) == list(b)


def test_subgroup_breakdown_on_a_hand_made_case():
    y = np.array([1, 1, 0, 0, 1, 0])
    predicted = np.array([1, 0, 1, 0, 1, 0])
    keys = [["a"], ["a", "b"], ["b"], ["a"], ["b"], ["b"]]
    table = subgroup_breakdown(y, predicted, keys)
    a, b = table["a"], table["b"]
    assert (a["rows"], a["tp"], a["fp"], a["tn"], a["fn"]) == (3, 1, 0, 1, 1)
    assert a["precision"] == 1.0
    assert a["recall"] == 0.5
    assert a["f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert a["fpr"] == 0.0
    assert (b["rows"], b["tp"], b["fp"], b["tn"], b["fn"]) == (4, 1, 1, 1, 1)
    assert (b["precision"], b["recall"], b["f1"], b["fpr"]) == (0.5, 0.5, 0.5, 0.5)


def test_subgroup_rates_are_none_when_undefined():
    table = subgroup_breakdown(np.array([0, 0]), np.array([0, 1]), [["legit"], ["legit"]])
    legit = table["legit"]
    assert legit["recall"] is None
    assert legit["fpr"] == 0.5
    assert legit["precision"] == 0.0


def test_top_domains_counts_only_the_masked_rows():
    groups = np.array(["x.com", "y.com", "x.com", "z.com", "x.com"])
    mask = np.array([True, True, True, False, False])
    assert top_domains(groups, mask, k=10) == [
        {"domain": "x.com", "count": 2},
        {"domain": "y.com", "count": 1},
    ]


def test_base_model_correlation_reports_pairs_and_shared_errors():
    y = np.array([1, 1, 0, 0])
    oof = {
        "random_forest": np.array([0.9, 0.2, 0.1, 0.6]),
        "xgboost": np.array([0.8, 0.3, 0.2, 0.7]),
        "lightgbm": np.array([0.7, 0.6, 0.3, 0.4]),
    }
    result = base_model_correlation(y, oof, threshold=0.5)
    pair = result["probability_pearson"]["random_forest|xgboost"]
    assert pair == pytest.approx(np.corrcoef(oof["random_forest"], oof["xgboost"])[0, 1], 1e-6)
    # Row 1 is wrong for RF and XGB, row 3 for RF and XGB; LightGBM gets row 1 right.
    assert result["rows_wrong_by_n_models"] == {"0": 2, "1": 0, "2": 2, "3": 0}


def test_report_measures_grouping_ranges_and_errors(finished):
    _, out_dir, report = finished
    saved = json.loads((out_dir / "20261116T100000Z" / "report.json").read_text(encoding="utf-8"))
    assert saved == report
    assert report["reportable"] is False
    assert "exploration" in report["run_type"]
    grouping = report["measurements"]["grouping"]
    for variant in ("grouped", "ungrouped"):
        entry = grouping[variant]
        for name in ("random_forest", "xgboost", "lightgbm", "stacking"):
            assert set(entry["cv_mean"][name]) == {"f1", "precision", "recall", "accuracy"}
    wide = report["measurements"]["ranges"]["wide"]
    assert wide["n_iter"] == 1
    assert "own" in wide["spaces_source"]
    assert set(wide["best_params"]["random_forest"]) == {
        "n_estimators", "max_depth", "min_samples_leaf"
    }
    errors = report["error_analysis"]
    assert set(errors["confusion"]) == {"tp", "fp", "tn", "fn"}
    assert set(errors["by_source"]) == {"phishtank", "hannousse_2021"}
    assert set(errors["by_synthetic"]) == {"false"}
    assert set(errors["by_capped_platform_domain"]["subgroups"]) <= {"capped", "not_capped"}
    assert errors["by_capped_platform_domain"]["domains"] == CAPPED
    assert set(report["base_model_correlation"]["probability_pearson"]) == {
        "random_forest|xgboost", "random_forest|lightgbm", "xgboost|lightgbm"
    }
    assert report["input"]["test_split"] == "not read"
    assert report["git"]["clean"] is not True or report["reportable"] is False


def test_report_holds_no_full_url(finished):
    _, out_dir, _ = finished
    text = (out_dir / "20261116T100000Z" / "report.json").read_text(encoding="utf-8")
    assert "://" not in text
    assert "/page/" not in text


def test_never_opens_the_blind_test_matrix(tmp_path, monkeypatch):
    data_dir = make_features_run(tmp_path)
    opened = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    diagnose(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, **FAST)
    assert opened
    assert not any("features_test" in path for path in opened)


def test_capped_subgroup_is_skipped_when_the_assemble_report_is_missing(tmp_path):
    data_dir = make_features_run(tmp_path)
    report = diagnose(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, **FAST)
    capped = report["error_analysis"]["by_capped_platform_domain"]
    assert capped["subgroups"] is None
    assert "not available" in capped["note"]


def test_refuses_to_overwrite_an_existing_diagnostic(tmp_path):
    data_dir = make_features_run(tmp_path)
    existing = tmp_path / "out" / "20261116T100000Z"
    existing.mkdir(parents=True)
    (existing / "report.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnose(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, **FAST)
    assert (existing / "report.json").read_text(encoding="utf-8") == "keep"


def test_cli_writes_a_report(tmp_path, monkeypatch, capsys):
    data_dir = make_features_run(tmp_path)
    monkeypatch.setattr(diagnose_module, "REPOSITORY_DIR", tmp_path)
    diagnose_module.main(
        [
            "--data-dir", str(data_dir),
            "--out-dir", str(tmp_path / "out"),
            "--n-iter", "1", "--n-iter-wide", "1", "--n-splits", "3", "--n-jobs", "1",
            "--threshold-step", "0.25",
        ]
    )
    reports = list((tmp_path / "out").glob("*/report.json"))
    assert len(reports) == 1
    assert "stacking" in capsys.readouterr().out

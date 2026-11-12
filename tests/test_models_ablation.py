import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from test_models_diagnose import CAPPED, write_assemble_report
from test_models_stacking import synthetic_matrix
from test_models_train import make_features_run

from sereno_model.features.candidates import CANDIDATE_NAMES, extract_candidates
from sereno_model.models import ablation as ablation_module
from sereno_model.models.ablation import VARIANTS, ablation, candidate_columns, prevalence

NOW = datetime(2026, 11, 16, 11, 0, 0, tzinfo=UTC)
FAST = {"n_iter": 1, "n_splits": 3, "n_jobs": 1, "threshold_grid": [0.3, 0.5, 0.7]}


def matrix_with_list_signals():
    """The synthetic matrix with URLs that switch the candidate features on for some rows."""
    frame = synthetic_matrix()
    urls = []
    for index, (url, label) in enumerate(zip(frame["url"], frame["label"], strict=True)):
        if label and index % 3 == 0:
            url = url.replace("/page/", "/login/paypal.com/")
        elif not label and index % 5 == 0:
            url = url.replace("/page/", "/account/")
        urls.append(url)
    frame["url"] = urls
    return frame


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("ablation")
    data_dir = make_features_run(root, matrix_with_list_signals())
    write_assemble_report(data_dir)
    out_dir = root / "artifacts" / "diagnostics"
    report = ablation(data_dir, out_dir, now=NOW, repo_dir=root, **FAST)
    return root, out_dir, report


def test_variants_are_v2_each_group_alone_and_all_groups():
    assert VARIANTS == {
        "v2": (),
        "v2+G1": ("G1",),
        "v2+G2": ("G2",),
        "v2+G3": ("G3",),
        "v2+G1+G2+G3": ("G1", "G2", "G3"),
    }


def test_candidate_columns_use_the_extractor_on_each_url():
    frame = matrix_with_list_signals()
    columns = candidate_columns(frame["url"])
    assert list(columns.columns) == list(CANDIDATE_NAMES)
    assert list(columns.index) == list(frame.index)
    for position in (0, 3, 5, 10):
        expected = extract_candidates(frame["url"].iloc[position])
        assert columns.iloc[position].to_dict() == expected


def test_prevalence_on_a_hand_made_case():
    columns = pd.DataFrame(
        {"is_shortener": [1, 0, 0, 1, 0], "sensitive_word_count": [2, 0, 1, 0, 0]}
    )
    labels = [1, 1, 0, 0, 0]
    sources = [["a"], ["a", "b"], ["b"], ["b"], ["c"]]
    table = prevalence(columns, labels, sources)
    short = table["is_shortener"]
    assert short["by_class"] == {"phishing": 0.5, "legitimate": pytest.approx(1 / 3, abs=1e-6)}
    assert short["by_source"]["b"]["legitimate"] == {"rows": 2, "nonzero_share": 0.5}
    assert short["by_source"]["b"]["phishing"] == {"rows": 1, "nonzero_share": 0.0}
    assert short["by_source"]["c"]["phishing"] == {"rows": 0, "nonzero_share": None}
    assert table["sensitive_word_count"]["mean_by_class"]["phishing"] == 1.0


def test_report_compares_every_variant_with_v2(finished):
    _, out_dir, report = finished
    saved = json.loads((out_dir / "20261116T110000Z" / "report.json").read_text(encoding="utf-8"))
    assert saved == report
    assert report["reportable"] is False
    assert "exploration" in report["run_type"]
    assert report["input"]["test_split"] == "not read"
    assert set(report["variants"]) == set(VARIANTS)
    assert report["variants"]["v2"]["added_features"] == []
    assert report["variants"]["v2+G1"]["added_features"] == ["tld_in_subdomain", "tld_in_path"]
    for entry in report["variants"].values():
        for name in ("random_forest", "xgboost", "lightgbm", "stacking"):
            assert set(entry["cv_mean"][name]) == {"f1", "precision", "recall", "accuracy"}
        assert len(entry["stacking_fold_f1"]) == 3
        assert "by_source" in entry["error_analysis"]
        assert set(entry["error_analysis"]["by_capped_platform_domain"]["subgroups"]) <= {
            "capped", "not_capped"
        }
        assert entry["error_analysis"]["by_capped_platform_domain"]["domains"] == CAPPED
    assert report["variants"]["v2"]["delta_vs_v2"]["stacking_f1"] == 0.0
    assert set(report["prevalence"]) == set(CANDIDATE_NAMES)
    assert report["lists"]["shortener_domains"][0]["domain"] == "bit.ly"


def test_report_holds_no_full_url(finished):
    _, out_dir, _ = finished
    text = (out_dir / "20261116T110000Z" / "report.json").read_text(encoding="utf-8")
    assert "://" not in text
    assert "/page/" not in text
    assert "/login/" not in text


def test_never_opens_the_blind_test_matrix(tmp_path, monkeypatch):
    data_dir = make_features_run(tmp_path)
    opened = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    ablation(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, variants=("v2", "v2+G3"),
             **FAST)
    assert opened
    assert not any("features_test" in path for path in opened)


def test_refuses_to_overwrite_an_existing_report(tmp_path):
    data_dir = make_features_run(tmp_path)
    existing = tmp_path / "out" / "20261116T110000Z"
    existing.mkdir(parents=True)
    (existing / "report.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ablation(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, **FAST)
    assert (existing / "report.json").read_text(encoding="utf-8") == "keep"


def test_cli_writes_a_report(tmp_path, monkeypatch, capsys):
    data_dir = make_features_run(tmp_path)
    monkeypatch.setattr(ablation_module, "REPOSITORY_DIR", tmp_path)
    ablation_module.main(
        [
            "--data-dir", str(data_dir),
            "--out-dir", str(tmp_path / "out"),
            "--n-iter", "1", "--n-splits", "3", "--n-jobs", "1", "--threshold-step", "0.25",
            "--variants", "v2", "v2+G2",
        ]
    )
    reports = list((tmp_path / "out").glob("*/report.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert set(saved["variants"]) == {"v2", "v2+G2"}
    assert "v2+G2" in capsys.readouterr().out

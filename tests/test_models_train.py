import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import inspect

import pytest
from test_models_stacking import synthetic_matrix

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.clean import write_parquet
from sereno_model.features.build import MATRIX_SCHEMA, matrix_hash
from sereno_model.features.extract import FEATURE_NAMES, FEATURE_SET_VERSION
from sereno_model.models.stacking import BASE_MODEL_NAMES, StackingModel
from sereno_model.models.train import (
    DirtyRepositoryError,
    main,
    train,
)

FEATURES_RUN = "20261112T025607Z"
NOW = datetime(2026, 11, 15, 9, 0, 0, tzinfo=UTC)
FAST = {"n_iter": 2, "n_splits": 3, "n_jobs": 1, "threshold_grid": [0.3, 0.5, 0.7],
        "latency_rows": 10, "latency_warmup": 2}


def make_features_run(
    root: Path, frame: pd.DataFrame | None = None, split_strategy: str | None = "random"
) -> Path:
    """Write a train matrix and its features report; the test file is deliberately unreadable."""
    frame = synthetic_matrix() if frame is None else frame
    data_dir = root / "data"
    run_dir = data_dir / "processed" / FEATURES_RUN
    run_dir.mkdir(parents=True)
    write_parquet(frame, MATRIX_SCHEMA, run_dir / "features_train.parquet")
    (run_dir / "features_test.parquet").write_bytes(b"blind test: must never be opened in T1")
    split_run = {"dir": "processed/20261111T222736Z", "partition_hash": "p" * 64,
                 "seed": 20261111}
    if split_strategy is not None:
        split_run["strategy"] = split_strategy
    report = {
        "split_run": split_run,
        "assemble_run": {"dir": "processed/20261111T222725Z", "content_hash": "a" * 64},
        "feature_set": {"version": FEATURE_SET_VERSION, "names": list(FEATURE_NAMES)},
        "splits": {
            "train": {"rows": len(frame), "file": "features_train.parquet",
                      "content_hash": {"algorithm": "sha256", "value": matrix_hash(frame)}},
            "test": {"rows": 0, "file": "features_test.parquet",
                     "content_hash": {"algorithm": "sha256", "value": "t" * 64}},
        },
    }
    (run_dir / "features_report.json").write_text(json.dumps(report), encoding="utf-8")
    return data_dir


def run(root: Path, **overrides) -> dict:
    arguments = {
        "data_dir": make_features_run(root) if "data_dir" not in overrides else None,
        "artifacts_dir": root / "artifacts",
        "version": "0.1.0",
        "now": NOW,
        "allow_dirty": True,
        "repo_dir": root,
        **FAST,
    }
    arguments.update(overrides)
    return train(**arguments)


def test_writes_the_model_and_its_report_into_a_new_version_folder(tmp_path):
    report = run(tmp_path)
    version_dir = tmp_path / "artifacts" / "0.1.0"
    model = joblib.load(version_dir / "model.joblib")
    assert isinstance(model, StackingModel)
    assert model.model_version == "0.1.0"
    assert model.feature_set_version == FEATURE_SET_VERSION
    assert model.feature_names == list(FEATURE_NAMES)
    assert model.threshold == report["threshold"]["value"]
    saved = json.loads((version_dir / "report.json").read_text(encoding="utf-8"))
    assert saved == report


def test_report_holds_every_required_key(tmp_path):
    report = run(tmp_path)
    for key in (
        "git", "command", "seed", "input", "versions", "machine", "cv", "search",
        "base_models", "meta_model", "stacking", "threshold", "durations", "model",
    ):
        assert key in report, key
    assert set(report["git"]) == {"commit", "clean"}
    assert report["seed"] == 20261111
    assert report["input"]["train_content_hash"] == matrix_hash(synthetic_matrix())
    assert report["input"]["partition_hash"] == "p" * 64
    assert report["input"]["test_split"] == "not read"
    for library in ("python", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib"):
        assert library in report["versions"]
    assert report["search"]["n_iter"] == 2
    assert "own decision" in report["search"]["n_iter_source"]
    assert report["threshold"]["grid"] == [0.3, 0.5, 0.7]
    for name in BASE_MODEL_NAMES:
        entry = report["base_models"][name]
        assert entry["best_params"]
        assert set(entry["cv"]["mean"]) == {"f1", "precision", "recall", "accuracy"}
        assert len(entry["cv"]["per_fold"]) == 3
    assert len(report["stacking"]["cv"]["per_fold"]) == 3
    assert {"total_seconds", "search_seconds", "oof_seconds", "refit_seconds"} <= set(
        report["durations"]
    )


def test_never_opens_the_test_matrix(tmp_path, monkeypatch):
    opened = []
    original = pd.read_parquet

    def spy(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    run(tmp_path)
    assert opened == ["features_train.parquet"]


def test_refuses_to_overwrite_an_existing_version(tmp_path):
    data_dir = make_features_run(tmp_path)
    (tmp_path / "artifacts" / "0.1.0").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run(tmp_path, data_dir=data_dir)
    assert not any((tmp_path / "artifacts" / "0.1.0").iterdir())


def test_refuses_an_official_run_from_a_repository_not_known_to_be_clean(tmp_path):
    data_dir = make_features_run(tmp_path)
    with pytest.raises(DirtyRepositoryError):
        run(tmp_path, data_dir=data_dir, allow_dirty=False)
    assert not (tmp_path / "artifacts").exists()


def test_exploratory_run_is_marked_as_not_reportable(tmp_path):
    report = run(tmp_path)
    assert report["reportable"] is False
    assert "exploration" in report["run_type"]


def test_refuses_a_train_matrix_that_does_not_match_its_recorded_hash(tmp_path):
    data_dir = make_features_run(tmp_path)
    path = data_dir / "processed" / FEATURES_RUN / "features_train.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "url_length"] = frame.loc[0, "url_length"] + 1
    write_parquet(frame, MATRIX_SCHEMA, path)
    with pytest.raises(ChecksumMismatchError):
        run(tmp_path, data_dir=data_dir)


def test_refuses_a_matrix_with_rows_from_another_split(tmp_path):
    frame = synthetic_matrix()
    frame.loc[0, "split"] = "test"
    data_dir = make_features_run(tmp_path, frame)
    with pytest.raises(ValueError, match="test"):
        run(tmp_path, data_dir=data_dir)


def test_two_runs_with_the_same_seed_give_the_same_threshold_and_scores(tmp_path):
    data_dir = make_features_run(tmp_path)
    first = run(tmp_path, data_dir=data_dir, version="a")
    second = run(tmp_path, data_dir=data_dir, version="b")
    assert first["threshold"] == second["threshold"]
    assert first["stacking"] == second["stacking"]
    frame = synthetic_matrix()
    model_a = joblib.load(tmp_path / "artifacts" / "a" / "model.joblib")
    model_b = joblib.load(tmp_path / "artifacts" / "b" / "model.joblib")
    assert np.array_equal(model_a.score(frame), model_b.score(frame))


def test_cli_trains_and_prints_a_summary(tmp_path, capsys):
    data_dir = make_features_run(tmp_path)
    main([
        "--data-dir", str(data_dir), "--artifacts-dir", str(tmp_path / "artifacts"),
        "--version", "cli", "--n-iter", "2", "--n-splits", "3", "--n-jobs", "1",
        "--threshold-step", "0.25", "--allow-dirty",
    ])
    out = capsys.readouterr().out
    assert "threshold" in out
    report = json.loads((tmp_path / "artifacts" / "cli" / "report.json").read_text("utf-8"))
    assert report["command"].startswith("python -m sereno_model.models.train")
    assert report["threshold"]["grid"] == [0.25, 0.5, 0.75]


def test_report_records_the_tfidf_feature_set_folds_latency_and_size(tmp_path):
    report = run(tmp_path)
    feature_set = report["feature_set"]
    assert feature_set["version"] == FEATURE_SET_VERSION
    assert feature_set["names"] == list(FEATURE_NAMES)
    assert feature_set["tfidf"] == {
        "analyzer": "char",
        "ngram_range": [1, 3],
        "max_features": 5000,
        "lowercase": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": False,
        "max_features_source": "P02",
        "transformer_params_source": "P32 (also the scikit-learn defaults)",
        "selection_source": "variant C of artifacts/diagnostics/20261112T192824Z",
        "text": feature_set["tfidf"]["text"],
        "fitted_on": feature_set["tfidf"]["fitted_on"],
    }
    assert report["cv"]["strategy"] == "random"
    assert report["cv"]["strategy_source"] == "split run"
    assert report["cv"]["method"].startswith("StratifiedKFold")
    latency = report["latency_ms"]
    assert latency["rows"] == 10 and latency["p50"] > 0 and latency["p95"] >= latency["p50"]
    model_file = tmp_path / "artifacts" / "0.1.0" / "model.joblib"
    assert report["model_size_bytes"] == model_file.stat().st_size


def test_cv_folds_follow_the_split_strategy_unless_one_is_given(tmp_path):
    grouped_dir = make_features_run(tmp_path / "g", split_strategy="grouped")
    legacy_dir = make_features_run(tmp_path / "l", split_strategy=None)
    grouped = run(tmp_path / "g", data_dir=grouped_dir)
    legacy = run(tmp_path / "l", data_dir=legacy_dir)
    forced = run(tmp_path / "l", data_dir=legacy_dir, version="forced", cv_strategy="random")
    assert (grouped["cv"]["strategy"], grouped["cv"]["method"][:20]) == (
        "grouped", "StratifiedGroupKFold"
    )
    assert legacy["cv"]["strategy"] == "grouped"
    assert (forced["cv"]["strategy"], forced["cv"]["strategy_source"]) == ("random", "argument")


def test_saved_model_scores_raw_urls(tmp_path):
    run(tmp_path)
    model = joblib.load(tmp_path / "artifacts" / "0.1.0" / "model.joblib")
    urls = ["https://login.example-bank.com/verify?id=7", "https://www.wikipedia.org/"]
    scores = model.score(urls)
    assert scores.shape == (2,)
    assert np.all((scores >= 0) & (scores <= 1))
    assert set(model.predict(urls)) <= {0, 1}
    assert model.score_url(urls[0]) == pytest.approx(scores[0])


def test_default_search_budget_is_twenty_candidates():
    assert inspect.signature(train).parameters["n_iter"].default == 20

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from test_models_diagnose import write_assemble_report
from test_models_stacking import synthetic_matrix
from test_models_train import make_features_run

from sereno_model.data.normalize import parse_url
from sereno_model.features.extract import FEATURE_NAMES, features_from_parts, measured_text
from sereno_model.models import tfidf_ablation as module
from sereno_model.data.split import make_folds
from sereno_model.models.stacking import BASE_MODEL_NAMES, StackingModel, cross_validate
from sereno_model.models.tfidf_ablation import (
    TEXT_COLUMN,
    VARIANTS,
    model_inputs,
    tfidf_ablation,
    tfidf_estimator,
    tfidf_vectorizer,
    url_text,
)

NOW = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)
FAST = {
    "n_iter": 1,
    "n_iter_tfidf": 1,
    "n_splits": 3,
    "n_jobs": 1,
    "threshold_grid": [0.3, 0.5, 0.7],
    "latency_rows": 15,
    "latency_warmup": 3,
}
TINY_SPACES = {
    "random_forest": {"model__n_estimators": [5], "model__max_depth": [3]},
    "xgboost": {"model__n_estimators": [5], "model__max_depth": [3],
                "model__learning_rate": [0.1]},
    "lightgbm": {"model__n_estimators": [5], "model__max_depth": [3],
                 "model__learning_rate": [0.1]},
}


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("tfidf")
    data_dir = make_features_run(root, synthetic_matrix())
    write_assemble_report(data_dir)
    out_dir = root / "artifacts" / "diagnostics"
    report = tfidf_ablation(data_dir, out_dir, now=NOW, repo_dir=root, **FAST)
    return root, out_dir, report


def test_url_text_drops_scheme_and_www_as_the_v2_vector_does():
    urls = ["http://www.Example.com/a/b/?q=1", "https://example.com/a/b/?q=1",
            "https://www.www.shop.example.com:443/x/"]
    texts = url_text(urls)
    assert texts[0] == texts[1]
    for url, text in zip(urls, texts, strict=True):
        assert text == measured_text(parse_url(url))
        assert "://" not in text
        assert not text.startswith("www.")
        assert "http" not in text


def test_vectorizer_settings_follow_p02_and_the_sklearn_defaults_of_p32():
    vectorizer = tfidf_vectorizer((1, 3))
    params = vectorizer.get_params()
    assert params["analyzer"] == "char"
    assert params["ngram_range"] == (1, 3)
    assert params["max_features"] == 5000
    assert params["lowercase"] is True
    assert params["norm"] == "l2"
    assert params["use_idf"] is True
    assert params["smooth_idf"] is True
    assert params["sublinear_tf"] is False


def _text_frame():
    frame = synthetic_matrix(rows=60, groups=12)
    frame["url"] = [
        f"https://site{index}.com/{'qzx' if index >= 40 else 'abc'}/{index}"
        for index in range(len(frame))
    ]
    return frame


def test_vectorizer_learns_its_vocabulary_from_training_rows_only():
    frame = _text_frame()
    X = model_inputs(frame, with_vector=True)
    y = frame["label"].to_numpy()
    train = np.arange(40)
    pipeline = tfidf_estimator("lightgbm", 0, (1, 3), with_vector=True)
    pipeline.set_params(model__n_estimators=5)
    pipeline.fit(X.iloc[train], y[train])
    vocabulary = pipeline.named_steps["features"].named_transformers_["tfidf"].vocabulary_
    assert "abc" in vocabulary
    assert "qzx" not in vocabulary
    assert "q" not in vocabulary


def test_cross_validation_refits_the_vectorizer_inside_each_training_fold(monkeypatch):
    frame = _text_frame()
    X = model_inputs(frame, with_vector=True)
    y = frame["label"].to_numpy().astype(int)
    folds = make_folds(y, frame["group"], n_splits=3, seed=0, strategy="random")
    seen = []
    original = TfidfVectorizer.fit_transform

    def spy(self, raw_documents, y=None):
        seen.append(set(raw_documents))
        return original(self, raw_documents, y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", spy)
    cross_validate(
        X, y, folds, search_spaces=TINY_SPACES, n_iter=1, seed=0, n_jobs=1,
        threshold_grid=[0.5],
        estimator_factory=lambda name, seed: tfidf_estimator(name, seed, (1, 3), True),
    )
    texts = X[TEXT_COLUMN].to_numpy()
    train_sets = [set(texts[train]) for train, _ in folds]
    assert seen
    for fitted in seen:
        assert fitted in train_sets
    for (_, val), train_set in zip(folds, train_sets, strict=True):
        assert not set(texts[val]) & train_set


@pytest.mark.parametrize("name", BASE_MODEL_NAMES)
def test_sparse_pipeline_works_with_every_base_model(name):
    frame = _text_frame()
    X = model_inputs(frame, with_vector=True)
    y = frame["label"].to_numpy().astype(int)
    pipeline = tfidf_estimator(name, 0, (3, 5), with_vector=True)
    pipeline.set_params(model__n_estimators=5)
    pipeline.fit(X, y)
    matrix = pipeline.named_steps["features"].transform(X)
    assert sparse.issparse(matrix)
    vocabulary = pipeline.named_steps["features"].named_transformers_["tfidf"].vocabulary_
    assert matrix.shape[1] == len(vocabulary) + len(FEATURE_NAMES)
    scores = pipeline.predict_proba(X)[:, 1]
    assert scores.shape == (len(frame),)
    assert np.all((scores >= 0) & (scores <= 1))


@pytest.mark.parametrize("with_vector", [True, False])
def test_refit_best_builds_the_saved_model_class(with_vector):
    frame = _text_frame()
    y = frame["label"].to_numpy().astype(int)
    folds = make_folds(y, frame["group"], n_splits=3, seed=0, strategy="random")
    cv = cross_validate(
        model_inputs(frame, with_vector), y, folds, search_spaces=TINY_SPACES, n_iter=1, seed=0,
        n_jobs=1, threshold_grid=[0.5],
        estimator_factory=lambda name, seed: tfidf_estimator(name, seed, (1, 3), with_vector),
    )
    spec = {"with_vector": with_vector, "ngram_range": (1, 3), "grouped": False}

    model, _ = module.refit_best(frame, cv, spec, seed=0)

    assert type(model) is StackingModel
    assert model.ngram_range == (1, 3)
    assert model.threshold == cv.threshold["threshold"]
    for url in frame["url"][:5]:
        parts = parse_url(url)
        row = {TEXT_COLUMN: measured_text(parts)}
        row.update(zip(FEATURE_NAMES, features_from_parts(parts), strict=True))
        matrix = model.features.transform(pd.DataFrame([row]))
        meta = np.array([[m.predict_proba(matrix)[0, 1] for m in model.base_models.values()]])
        assert model.score_url(url) == float(model.meta_model.predict_proba(meta)[0, 1])


def test_the_ablation_has_no_model_class_of_its_own():
    assert not hasattr(module, "TfidfStacking")


def test_text_only_inputs_hold_no_vector_feature():
    X = model_inputs(_text_frame(), with_vector=False)
    assert list(X.columns) == [TEXT_COLUMN]


def test_report_has_every_variant_latency_and_size(finished):
    _, out_dir, report = finished
    saved = json.loads((out_dir / "20261116T120000Z" / "report.json").read_text(encoding="utf-8"))
    assert saved == report
    assert report["reportable"] is False
    assert "exploration" in report["run_type"]
    assert report["input"]["test_split"] == "not read"
    assert set(report["variants"]) == set(VARIANTS)
    assert report["variants"]["E"]["folds"].startswith("StratifiedGroupKFold")
    assert report["variants"]["E"]["ngram_range"] in ([1, 3], [3, 5])
    assert report["variants"]["E"]["source_variant"] in ("C", "D")
    for name in ("A", "B", "C", "D"):
        assert report["variants"][name]["folds"].startswith("StratifiedKFold")
    for entry in report["variants"].values():
        for model in (*BASE_MODEL_NAMES, "stacking"):
            assert set(entry["cv_mean"][model]) == {"f1", "precision", "recall", "accuracy"}
        assert len(entry["stacking_fold_f1"]) == 3
        assert "by_source" in entry["error_analysis"]
        assert "delta_vs_A" in entry
        for params in entry["best_params"].values():
            assert not any(key.startswith("model__") for key in params)
    best = report["best_variant"]
    assert best["name"] in ("C", "D")
    latency = best["latency_ms"]
    assert latency["rows"] == 15
    assert 0 < latency["p50"] <= latency["p95"]
    assert best["model_size_bytes"] > 0
    assert report["variants"]["A"]["delta_vs_A"]["stacking_f1"] == 0.0


def test_report_holds_no_full_url(finished):
    _, out_dir, _ = finished
    text = (out_dir / "20261116T120000Z" / "report.json").read_text(encoding="utf-8")
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
    tfidf_ablation(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, **FAST)
    assert opened
    assert not any("features_test" in path for path in opened)


def test_refuses_to_overwrite_an_existing_report(tmp_path):
    data_dir = make_features_run(tmp_path)
    existing = tmp_path / "out" / "20261116T120000Z"
    existing.mkdir(parents=True)
    (existing / "report.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        tfidf_ablation(data_dir, tmp_path / "out", now=NOW, repo_dir=tmp_path, **FAST)
    assert (existing / "report.json").read_text(encoding="utf-8") == "keep"


def test_cli_writes_a_report(tmp_path, monkeypatch, capsys):
    data_dir = make_features_run(tmp_path)
    monkeypatch.setattr(module, "REPOSITORY_DIR", tmp_path)
    module.main(
        [
            "--data-dir", str(data_dir),
            "--out-dir", str(tmp_path / "out"),
            "--n-iter", "1", "--n-iter-tfidf", "1", "--n-splits", "3", "--n-jobs", "1",
            "--threshold-step", "0.25", "--latency-rows", "5", "--latency-warmup", "1",
        ]
    )
    reports = list((tmp_path / "out").glob("*/report.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert set(saved["variants"]) == set(VARIANTS)
    assert "latency" in capsys.readouterr().out

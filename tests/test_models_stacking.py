import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from sereno_model.features.build import AUDIT_COLUMNS, METADATA_COLUMNS
from sereno_model.features.extract import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    FLOAT_FEATURES,
)
from sereno_model.models.stacking import (
    BASE_MODEL_NAMES,
    StackingModel,
    feature_columns,
    fit_stacking,
    grouped_folds,
    oof_probabilities,
    select_threshold,
)

SEED = 20261111


def synthetic_matrix(rows: int = 300, groups: int = 60, seed: int = 0) -> pd.DataFrame:
    """A feature matrix with the real column schema; the label leaks into two features."""
    rng = np.random.default_rng(seed)
    group_ids = np.arange(rows) % groups
    labels = (group_ids % 2).astype("int8")
    frame = pd.DataFrame(
        {
            "url": [f"https://site{g}.com/page/{index}" for index, g in enumerate(group_ids)],
            "label": labels,
            "split": "train",
            "group": [f"site{g}.com" for g in group_ids],
            "synthetic": False,
            "sources": [["phishtank"] if label else ["hannousse_2021"] for label in labels],
            "audit_scheme": "https",
            "audit_has_www": False,
        }
    )
    for name in FEATURE_NAMES:
        if name in FLOAT_FEATURES:
            frame[name] = rng.random(rows)
        else:
            frame[name] = rng.integers(0, 30, rows).astype("int32")
    frame["url_length"] = (frame["url_length"] + 25 * labels).astype("int32")
    frame["url_entropy"] = frame["url_entropy"] + labels
    return frame


def test_feature_columns_are_the_vector_without_metadata_or_audit_columns():
    frame = synthetic_matrix()
    columns = feature_columns(frame)
    assert columns == list(FEATURE_NAMES)
    assert len(columns) == 24
    for name in (*METADATA_COLUMNS, *AUDIT_COLUMNS):
        assert name not in columns


def test_feature_columns_fail_when_a_feature_is_missing():
    frame = synthetic_matrix().drop(columns=["url_entropy"])
    with pytest.raises(ValueError, match="url_entropy"):
        feature_columns(frame)


def test_feature_columns_fail_on_an_unknown_column():
    frame = synthetic_matrix()
    frame["is_shortener"] = 0
    with pytest.raises(ValueError, match="is_shortener"):
        feature_columns(frame)


def test_folds_never_put_a_group_on_both_sides_and_cover_every_row_once():
    frame = synthetic_matrix()
    folds = grouped_folds(frame["label"], frame["group"], n_splits=5, seed=SEED)
    assert len(folds) == 5
    groups = frame["group"].to_numpy()
    validated = np.concatenate([val for _, val in folds])
    assert sorted(validated) == list(range(len(frame)))
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
        assert len(set(train) | set(val)) == len(frame)


def test_folds_are_fixed_by_the_seed():
    frame = synthetic_matrix()
    first = grouped_folds(frame["label"], frame["group"], n_splits=5, seed=SEED)
    second = grouped_folds(frame["label"], frame["group"], n_splits=5, seed=SEED)
    for (train_a, val_a), (train_b, val_b) in zip(first, second, strict=True):
        assert np.array_equal(train_a, train_b) and np.array_equal(val_a, val_b)


class Memorizer(ClassifierMixin, BaseEstimator):
    """Scores 1 or 0 for a row it was trained on, 0.5 for any row it has never seen."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.seen_ = {tuple(row): label for row, label in zip(np.asarray(X), y, strict=True)}
        return self

    def predict_proba(self, X):
        scores = [float(self.seen_.get(tuple(row), 0.5)) for row in np.asarray(X)]
        return np.column_stack([1 - np.array(scores), scores])


def test_oof_probabilities_exist_for_every_row_and_come_from_models_that_never_saw_it():
    frame = synthetic_matrix()
    frame["url_length"] = np.arange(len(frame), dtype="int32")  # every row unique
    X, y = frame[list(FEATURE_NAMES)], frame["label"].to_numpy()
    folds = grouped_folds(y, frame["group"], n_splits=5, seed=SEED)
    scores = oof_probabilities(Memorizer(), X, y, folds)
    assert scores.shape == (len(frame),)
    assert not np.isnan(scores).any()
    assert np.all(scores == 0.5)


def test_threshold_maximizes_the_mean_f1_over_the_folds():
    # A score counts as phishing when it reaches the threshold. Fold 0 is perfect for any
    # threshold in (0.3, 0.6]; fold 1 only in (0.45, 0.5], so only 0.5 is perfect on both.
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.6, 0.9, 0.2, 0.45, 0.5, 0.8])
    folds = [
        (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
    ]
    choice = select_threshold(y, scores, folds, grid=[0.2, 0.4, 0.5, 0.7])
    assert choice["threshold"] == 0.5
    assert choice["mean_f1"] == pytest.approx(1.0)
    assert len(choice["fold_f1"]) == 2
    assert choice["curve"][0] == {"threshold": 0.2, "mean_f1": pytest.approx((0.8 + 2 / 3) / 2)}


def test_threshold_ties_go_to_the_value_closest_to_one_half():
    y = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    folds = [(np.array([0]), np.array([0, 1]))]
    choice = select_threshold(y, scores, folds, grid=[0.2, 0.45, 0.6, 0.8])
    assert choice["threshold"] == 0.45


def small_search() -> dict:
    return {"n_iter": 2, "n_splits": 3, "seed": SEED, "n_jobs": 1}


def test_fit_stacking_returns_a_scoring_model_and_its_cv_evidence():
    frame = synthetic_matrix()
    result = fit_stacking(frame, threshold_grid=[0.3, 0.5, 0.7], **small_search())
    model = result.model
    assert isinstance(model, StackingModel)
    assert model.feature_names == list(FEATURE_NAMES)
    assert model.feature_set_version == FEATURE_SET_VERSION
    assert model.threshold == result.threshold["threshold"]
    assert set(result.oof_base) == set(BASE_MODEL_NAMES)
    assert result.oof_meta.shape == (len(frame),)

    scores = model.score(frame)
    assert scores.shape == (len(frame),)
    assert np.all((scores >= 0) & (scores <= 1))
    probabilities = model.predict_proba(frame)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(frame))) <= {0, 1}
    for name in (*BASE_MODEL_NAMES, "stacking"):
        metrics = result.cv_metrics[name]
        assert len(metrics["per_fold"]) == 3
        assert set(metrics["mean"]) == {"f1", "precision", "recall", "accuracy"}


def test_model_scores_ignore_column_order_and_refuse_missing_features():
    frame = synthetic_matrix()
    model = fit_stacking(frame, threshold_grid=[0.5], **small_search()).model
    shuffled = frame[list(reversed(frame.columns))]
    assert np.array_equal(model.score(frame), model.score(shuffled))
    with pytest.raises(ValueError, match="tilde_count"):
        model.score(frame.drop(columns=["tilde_count"]))


def test_same_seed_gives_the_same_oof_scores_and_threshold():
    frame = synthetic_matrix()
    first = fit_stacking(frame, threshold_grid=[0.3, 0.5, 0.7], **small_search())
    second = fit_stacking(frame, threshold_grid=[0.3, 0.5, 0.7], **small_search())
    assert np.array_equal(first.oof_meta, second.oof_meta)
    for name in BASE_MODEL_NAMES:
        assert np.array_equal(first.oof_base[name], second.oof_base[name])
        assert first.best_params[name] == second.best_params[name]
    assert first.threshold["threshold"] == second.threshold["threshold"]
    assert np.array_equal(first.model.score(frame), second.model.score(frame))


def test_search_ranges_follow_p09():
    from sereno_model.models.stacking import SEARCH_SPACES

    rf, xgb, lgbm = (SEARCH_SPACES[name] for name in BASE_MODEL_NAMES)
    assert (min(rf["n_estimators"]), max(rf["n_estimators"])) == (50, 100)
    assert (min(rf["max_depth"]), max(rf["max_depth"])) == (1, 10)
    assert (min(xgb["n_estimators"]), max(xgb["n_estimators"])) == (25, 100)
    assert (min(lgbm["n_estimators"]), max(lgbm["n_estimators"])) == (20, 100)
    for space in (xgb, lgbm):
        assert (min(space["max_depth"]), max(space["max_depth"])) == (6, 10)
        assert min(space["learning_rate"]) == pytest.approx(0.05)
        assert max(space["learning_rate"]) == pytest.approx(0.2)

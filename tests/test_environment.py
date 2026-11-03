"""Smoke test: the designed Stacking ensemble trains, repeats and survives a save."""
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SEED = 42


def build_ensemble():
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=20, random_state=SEED)),
            ("xgb", XGBClassifier(n_estimators=20, random_state=SEED, n_jobs=1)),
            ("lgbm", LGBMClassifier(n_estimators=20, random_state=SEED, n_jobs=1, verbose=-1)),
        ],
        final_estimator=LogisticRegression(),
        cv=3,
        stack_method="predict_proba",
    )


def sample_data():
    return make_classification(n_samples=400, n_features=12, random_state=SEED)


def test_same_seed_gives_same_probabilities():
    X, y = sample_data()
    first = build_ensemble().fit(X, y).predict_proba(X)
    second = build_ensemble().fit(X, y).predict_proba(X)
    assert np.array_equal(first, second)


def test_saved_model_predicts_the_same(tmp_path):
    X, y = sample_data()
    model = build_ensemble().fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert np.array_equal(joblib.load(path).predict_proba(X), model.predict_proba(X))

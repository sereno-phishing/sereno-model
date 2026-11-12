"""The Stacking method approved for phase 3 and the model object saved as the artifact.

Method (``docs/03-modelos.md``, "Propuesta de método", approved on 2026-11-12):

1. Only the 24 features of the vector reach a model; metadata and audit columns never do.
2. ``StratifiedGroupKFold`` over the training split, grouped by registered domain. The same
   folds serve the hyperparameter search, the out-of-fold predictions and the threshold.
3. Random Forest, XGBoost and LightGBM are tuned with ``RandomizedSearchCV`` scored by F1, with
   the ranges of P09. The search budget ``n_iter`` is not in the corpus: it is an own decision.
4. The meta-features are the out-of-fold phishing probabilities of the tuned base models (P15);
   the meta-model is a logistic regression (P27) fitted on them. The base models are then
   refitted on the whole training split.
5. The threshold maximizes the mean F1 of the meta-model's out-of-fold scores over the folds
   (P17). The candidate grid is an own decision.
6. No calibration and no resampling.

Every estimator runs single-threaded with a fixed seed, and parallelism only spreads whole fits
over workers, so the result does not depend on the number of workers.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

from sereno_model.features.build import AUDIT_COLUMNS, METADATA_COLUMNS
from sereno_model.features.extract import FEATURE_NAMES, FEATURE_SET_VERSION

DEFAULT_SEED = 20261111
DEFAULT_N_SPLITS = 5
# Own decision: the corpus gives the ranges of the search, not how many candidates to draw.
DEFAULT_N_ITER = 20
# Own decision: candidate thresholds every 0.01 between 0.01 and 0.99.
DEFAULT_THRESHOLD_STEP = 0.01
# Threshold of the base models' CV metrics: the default decision rule of predict.
BASE_THRESHOLD = 0.5

BASE_MODEL_NAMES = ("random_forest", "xgboost", "lightgbm")

# Learning rates of P09 (0.05 to 0.2) drawn from a 0.01 grid; scipy is not a declared dependency.
_LEARNING_RATES = [round(0.05 + 0.01 * step, 2) for step in range(16)]

# Ranges of P09; both ends included.
SEARCH_SPACES = {
    "random_forest": {
        "n_estimators": list(range(50, 101)),
        "max_depth": list(range(1, 11)),
    },
    "xgboost": {
        "n_estimators": list(range(25, 101)),
        "max_depth": list(range(6, 11)),
        "learning_rate": _LEARNING_RATES,
    },
    "lightgbm": {
        "n_estimators": list(range(20, 101)),
        "max_depth": list(range(6, 11)),
        "learning_rate": _LEARNING_RATES,
    },
}

METRICS = ("f1", "precision", "recall", "accuracy")


def make_threshold_grid(step: float = DEFAULT_THRESHOLD_STEP) -> list[float]:
    """Candidate thresholds ``step, 2*step, ...`` strictly between 0 and 1."""
    if not 0 < step < 1:
        raise ValueError(f"the threshold step must be between 0 and 1, got {step}")
    grid = []
    index = 1
    while index * step < 1 - 1e-9:
        grid.append(round(index * step, 6))
        index += 1
    return grid


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The model inputs: the feature vector, after checking the matrix holds nothing else."""
    columns = set(frame.columns)
    missing = [name for name in FEATURE_NAMES if name not in columns]
    if missing:
        raise ValueError(f"the matrix lacks the features {missing}")
    known = {*METADATA_COLUMNS, *AUDIT_COLUMNS, *FEATURE_NAMES}
    unknown = sorted(columns - known)
    if unknown:
        raise ValueError(f"the matrix holds columns outside the feature set {unknown}")
    return list(FEATURE_NAMES)


def base_estimator(name: str, seed: int):
    """An untuned base model, single-threaded and seeded."""
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(
            random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss"
        )
    if name == "lightgbm":
        return LGBMClassifier(
            random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1
        )
    raise ValueError(f"unknown base model {name}")


def meta_estimator(seed: int) -> LogisticRegression:
    return LogisticRegression(random_state=seed)


def grouped_folds(labels, groups, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds where every row of a group lands in the same validation fold."""
    labels = np.asarray(labels)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (train, val)
        for train, val in splitter.split(np.zeros(len(labels)), labels, np.asarray(groups))
    ]


def _rows(X, index: np.ndarray):
    return X.iloc[index] if isinstance(X, pd.DataFrame) else np.asarray(X)[index]


def _fold_scores(estimator, X, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> np.ndarray:
    fitted = clone(estimator).fit(_rows(X, train), y[train])
    return fitted.predict_proba(_rows(X, val))[:, 1]


def oof_probabilities(estimator, X, y, folds, n_jobs: int = 1) -> np.ndarray:
    """Phishing probability of each row, from a copy of ``estimator`` fitted without its fold."""
    y = np.asarray(y)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_fold_scores)(estimator, X, y, train, val) for train, val in folds
    )
    scores = np.full(len(y), np.nan)
    for (_, val), part in zip(folds, parts, strict=True):
        scores[val] = part
    if np.isnan(scores).any():
        raise ValueError("the folds do not cover every row")
    return scores


def _metrics(y: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
    }


def fold_metrics(y, scores, folds, threshold: float) -> dict:
    """F1, precision, recall and accuracy on each validation fold and their mean."""
    y, scores = np.asarray(y), np.asarray(scores)
    per_fold = [_metrics(y[val], (scores[val] >= threshold).astype(int)) for _, val in folds]
    mean = {name: float(np.mean([fold[name] for fold in per_fold])) for name in METRICS}
    return {"threshold": float(threshold), "mean": mean, "per_fold": per_fold}


def select_threshold(y, scores, folds, grid) -> dict:
    """The grid value with the highest mean F1 over the folds (P17).

    A score counts as phishing when it reaches the threshold. Ties go to the value closest to
    0.5, then to the lower one.
    """
    y, scores = np.asarray(y), np.asarray(scores)
    curve, per_fold = [], {}
    for threshold in grid:
        f1s = [
            float(f1_score(y[val], (scores[val] >= threshold).astype(int), zero_division=0))
            for _, val in folds
        ]
        per_fold[threshold] = f1s
        curve.append({"threshold": float(threshold), "mean_f1": float(np.mean(f1s))})
    best = max(point["mean_f1"] for point in curve)
    tied = [point["threshold"] for point in curve if best - point["mean_f1"] <= 1e-12]
    chosen = min(tied, key=lambda value: (abs(value - 0.5), value))
    return {
        "threshold": chosen,
        "mean_f1": best,
        "fold_f1": per_fold[chosen],
        "curve": curve,
    }


class StackingModel:
    """The saved artifact: three base models, the meta-model and the decision threshold.

    ``score`` returns the meta-model's phishing probability in [0, 1]; ``predict`` turns it into
    1 (phishing) or 0 (legitimate) with ``threshold``. The input must hold the feature columns by
    name, in any order; other columns are ignored.
    """

    def __init__(
        self,
        base_models: dict,
        meta_model,
        threshold: float,
        feature_names: list[str],
        feature_set_version: str,
        model_version: str | None = None,
        created_at: str | None = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)
        self.feature_set_version = feature_set_version
        self.model_version = model_version
        self.created_at = created_at

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"the input lacks the features {missing}")
        return X[self.feature_names]

    def base_scores(self, X: pd.DataFrame) -> np.ndarray:
        """One column per base model, in the order of ``base_models``: the meta-features."""
        features = self._features(X)
        return np.column_stack(
            [model.predict_proba(features)[:, 1] for model in self.base_models.values()]
        )

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_model.predict_proba(self.base_scores(X))[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self.score(X)
        return np.column_stack([1 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.score(X) >= self.threshold).astype(int)


@dataclass
class FitResult:
    model: StackingModel
    folds: list
    best_params: dict
    search_best_f1: dict
    oof_base: dict
    oof_meta: np.ndarray
    threshold: dict
    cv_metrics: dict
    durations: dict = field(default_factory=dict)


def _plain(value):
    """Search results hold numpy scalars; the report needs plain Python numbers."""
    return value.item() if isinstance(value, np.generic) else value


@dataclass
class CrossValidation:
    """Everything the grouped (or any given) folds produce, before the final refit."""

    folds: list
    best_params: dict
    search_best_f1: dict
    tuned: dict
    oof_base: dict
    meta_features: np.ndarray
    oof_meta: np.ndarray
    threshold: dict
    cv_metrics: dict
    durations: dict = field(default_factory=dict)


def _progress(message: str) -> None:
    """Print a timestamped progress line so long runs show they are alive."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list,
    search_spaces: dict | None = None,
    n_iter: int = DEFAULT_N_ITER,
    seed: int = DEFAULT_SEED,
    n_jobs: int = 1,
    threshold_grid: list[float] | None = None,
    estimator_factory=None,
) -> CrossValidation:
    """Search, out-of-fold predictions, meta-model scores and threshold over ``folds``.

    ``estimator_factory(name, seed)`` builds each untuned base model; it defaults to
    ``base_estimator``. A factory that returns a pipeline lets every transformer in it be fitted
    inside each training fold, and its ``search_spaces`` keys then carry the step prefix.
    """
    spaces = SEARCH_SPACES if search_spaces is None else search_spaces
    make = base_estimator if estimator_factory is None else estimator_factory
    grid = make_threshold_grid() if threshold_grid is None else list(threshold_grid)
    y = np.asarray(y).astype(int)

    started = time.perf_counter()
    best_params, search_best_f1 = {}, {}
    for name in BASE_MODEL_NAMES:
        _progress(f"search {name}: {n_iter} candidates x {len(folds)} folds")
        search = RandomizedSearchCV(
            make(name, seed),
            spaces[name],
            n_iter=n_iter,
            scoring="f1",
            cv=folds,
            refit=False,
            random_state=seed,
            n_jobs=n_jobs,
        )
        search.fit(X, y)
        best_params[name] = {key: _plain(value) for key, value in search.best_params_.items()}
        search_best_f1[name] = float(search.best_score_)
        _progress(f"search {name} done: best CV F1 {search_best_f1[name]:.4f}")
    search_seconds = time.perf_counter() - started

    _progress("out-of-fold predictions and meta-model")
    started = time.perf_counter()
    tuned = {
        name: make(name, seed).set_params(**best_params[name]) for name in BASE_MODEL_NAMES
    }
    oof_base = {
        name: oof_probabilities(estimator, X, y, folds, n_jobs=n_jobs)
        for name, estimator in tuned.items()
    }
    meta_features = np.column_stack([oof_base[name] for name in BASE_MODEL_NAMES])
    oof_meta = oof_probabilities(meta_estimator(seed), meta_features, y, folds, n_jobs=n_jobs)
    oof_seconds = time.perf_counter() - started

    threshold = select_threshold(y, oof_meta, folds, grid)
    cv_metrics = {
        name: fold_metrics(y, oof_base[name], folds, BASE_THRESHOLD) for name in BASE_MODEL_NAMES
    }
    cv_metrics["stacking"] = fold_metrics(y, oof_meta, folds, threshold["threshold"])
    return CrossValidation(
        folds=folds,
        best_params=best_params,
        search_best_f1=search_best_f1,
        tuned=tuned,
        oof_base=oof_base,
        meta_features=meta_features,
        oof_meta=oof_meta,
        threshold=threshold,
        cv_metrics=cv_metrics,
        durations={
            "search_seconds": round(search_seconds, 3),
            "oof_seconds": round(oof_seconds, 3),
        },
    )


def fit_stacking(
    frame: pd.DataFrame,
    n_iter: int = DEFAULT_N_ITER,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    n_jobs: int = 1,
    threshold_grid: list[float] | None = None,
    model_version: str | None = None,
    created_at: str | None = None,
) -> FitResult:
    """Tune, stack and refit on ``frame``, the training split only."""
    columns = feature_columns(frame)
    X = frame[columns]
    y = frame["label"].to_numpy().astype(int)
    folds = grouped_folds(y, frame["group"], n_splits=n_splits, seed=seed)
    cv = cross_validate(
        X, y, folds, n_iter=n_iter, seed=seed, n_jobs=n_jobs, threshold_grid=threshold_grid
    )

    started = time.perf_counter()
    base_models = {name: estimator.fit(X, y) for name, estimator in cv.tuned.items()}
    meta_model = meta_estimator(seed).fit(cv.meta_features, y)
    refit_seconds = time.perf_counter() - started

    model = StackingModel(
        base_models,
        meta_model,
        cv.threshold["threshold"],
        columns,
        FEATURE_SET_VERSION,
        model_version=model_version,
        created_at=created_at,
    )
    return FitResult(
        model=model,
        folds=folds,
        best_params=cv.best_params,
        search_best_f1=cv.search_best_f1,
        oof_base=cv.oof_base,
        oof_meta=cv.oof_meta,
        threshold=cv.threshold,
        cv_metrics=cv.cv_metrics,
        durations={**cv.durations, "refit_seconds": round(refit_seconds, 3)},
    )

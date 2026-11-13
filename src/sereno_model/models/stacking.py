"""The Stacking method and the model object saved as the artifact.

Method, papers and own decisions: docs/03-modelos.md. Estimators are single-threaded and
seeded, so results do not depend on the number of workers.
"""

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from sereno_model.data.normalize import parse_url
from sereno_model.data.split import grouped_folds, make_folds  # noqa: F401 (re-exported)
from sereno_model.features.build import AUDIT_COLUMNS, METADATA_COLUMNS
from sereno_model.features.extract import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    features_from_parts,
    measured_text,
)

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

FOLD_RANDOM, FOLD_GROUPED = "random", "grouped"
FOLD_STRATEGIES = (FOLD_RANDOM, FOLD_GROUPED)
DEFAULT_FOLD_STRATEGY = FOLD_RANDOM
FOLD_METHODS = {
    FOLD_RANDOM: "StratifiedKFold(n_splits, shuffle=True, random_state=seed); groups ignored, "
    "as in the papers",
    FOLD_GROUPED: "StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed), grouped by "
    "registered domain",
}

TEXT_COLUMN = "url_text"
# P02 (AntiPhishStack) caps the character TF-IDF vocabulary at 5000 terms.
MAX_FEATURES = 5000
MODEL_STEP = "model"
# Variant C of the TF-IDF ablation (artifacts/diagnostics/20261112T192824Z); the corpus does not
# give the n-gram range, so it is an own decision measured there.
OFFICIAL_NGRAM_RANGE = (1, 3)
ABLATION_REPORT = "artifacts/diagnostics/20261112T192824Z"

DEFAULT_LATENCY_ROWS = 1000
DEFAULT_LATENCY_WARMUP = 50
# Own decision: the latency sample is drawn from the training rows with this offset on the seed.
LATENCY_SEED_OFFSET = 1


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
    return LogisticRegression(random_state=seed)  # P27


def url_text(urls) -> list[str]:
    """The measured text of every URL, as vector version 2 measures it."""
    return [measured_text(parse_url(url)) for url in urls]


def tfidf_vectorizer(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    """Character TF-IDF of the URL text."""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=tuple(ngram_range),
        max_features=MAX_FEATURES,  # P02
        lowercase=True,
        norm="l2",  # norm to sublinear_tf: P32, also the scikit-learn defaults
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )


def tfidf_settings(ngram_range: tuple[int, int]) -> dict:
    """The TF-IDF parameters as the reports record them."""
    return {
        "analyzer": "char",
        "ngram_range": list(ngram_range),
        "max_features": MAX_FEATURES,
        "lowercase": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": False,
        "max_features_source": "P02",
        "transformer_params_source": "P32 (also the scikit-learn defaults)",
        "selection_source": f"variant C of {ABLATION_REPORT}",
        "text": "measured text of vector v2: normalized URL without scheme and separator, "
        "leading 'www.' labels, default port and trailing slashes",
        "fitted_on": "training rows of each fold only (pipeline inside the search and the "
        "out-of-fold predictions); the final vectorizer on the whole training split",
    }


def feature_union(ngram_range: tuple[int, int], with_vector: bool) -> ColumnTransformer:
    """TF-IDF of the text column, optionally followed by the vector columns, as one sparse matrix.

    The vector features are non-negative, so XGBoost reading sparse zeros as missing sends
    them to the side where zero already sits.
    """
    transformers = [("tfidf", tfidf_vectorizer(ngram_range), TEXT_COLUMN)]
    if with_vector:
        transformers.append(("v2", "passthrough", list(FEATURE_NAMES)))
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def tfidf_estimator(name: str, seed: int, ngram_range, with_vector: bool) -> Pipeline:
    """A base model behind the TF-IDF features, so both are fitted on the same training rows."""
    return Pipeline(
        [
            ("features", feature_union(ngram_range, with_vector)),
            (MODEL_STEP, base_estimator(name, seed)),
        ]
    )


def prefixed_spaces(spaces: dict) -> dict:
    """The P09 ranges with the pipeline step prefix that ``RandomizedSearchCV`` needs."""
    return {
        name: {f"{MODEL_STEP}__{key}": values for key, values in space.items()}
        for name, space in spaces.items()
    }


def unprefixed(params: dict) -> dict:
    prefix = f"{MODEL_STEP}__"
    return {key.removeprefix(prefix): value for key, value in params.items()}


def model_inputs(frame: pd.DataFrame, with_vector: bool) -> pd.DataFrame:
    inputs = pd.DataFrame({TEXT_COLUMN: url_text(frame["url"])}, index=frame.index)
    if with_vector:
        inputs = pd.concat([inputs, frame[feature_columns(frame)]], axis=1)
    return inputs


def measure_latency(model, urls, warmup: int) -> dict:
    """Milliseconds per URL of ``model.score_url``, one URL at a time on one thread."""
    urls = list(urls)
    timings = []
    with threadpool_limits(limits=1):
        for url in urls[: max(warmup, 0)]:
            model.score_url(url)
        for url in urls:
            clock = time.perf_counter()
            model.score_url(url)
            timings.append((time.perf_counter() - clock) * 1000)
    values = np.array(timings)
    return {
        "rows": len(values),
        "warmup": warmup,
        "threads": 1,
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "mean": round(float(values.mean()), 3),
        "max": round(float(values.max()), 3),
        "scope": "vector extraction, measured text, TF-IDF and vector union, three base "
        "models and meta-model; no network, cache or server",
    }


def serialized_size(model) -> int:
    """Bytes of ``joblib.dump(model)``, written to a temporary directory and deleted."""
    directory = tempfile.mkdtemp(prefix="sereno-tfidf-size-")
    try:
        path = Path(directory) / "model.joblib"
        joblib.dump(model, path)
        return path.stat().st_size
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _rows(X, index: np.ndarray):
    return X.iloc[index] if isinstance(X, pd.DataFrame) else np.asarray(X)[index]


def _fold_scores(estimator, X, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> np.ndarray:
    fitted = clone(estimator).fit(_rows(X, train), y[train])
    return fitted.predict_proba(_rows(X, val))[:, 1]


def oof_probabilities(estimator, X, y, folds, n_jobs: int = 1) -> np.ndarray:
    """Phishing probability of each row, from ``estimator`` fitted without its fold (P15)."""
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

    A score reaching the threshold is phishing. Ties go to the value closest to 0.5, then to
    the lower one.
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
    """The saved artifact: feature union, three base models, meta-model and threshold.

    Inputs are a URL, a sequence of URLs or a frame with a ``url`` column; a frame holding
    every vector column uses those values. ``features`` is ``None`` for a vector-only model.
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
        features=None,
        ngram_range: tuple[int, int] | None = None,
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)
        self.feature_set_version = feature_set_version
        self.model_version = model_version
        self.created_at = created_at
        self.features = features
        self.ngram_range = tuple(ngram_range) if ngram_range else None

    def _inputs(self, X) -> pd.DataFrame:
        """The columns the union or the base models were fitted on, in their order."""
        if isinstance(X, str):
            X = [X]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame({"url": list(X)})
        present = [name for name in self.feature_names if name in X.columns]
        if len(present) == len(self.feature_names):
            inputs = X[self.feature_names].reset_index(drop=True)
        elif not present and "url" in X.columns:
            vectors = [features_from_parts(parse_url(url)) for url in X["url"]]
            inputs = pd.DataFrame(vectors, columns=list(FEATURE_NAMES))[self.feature_names]
        else:
            missing = [name for name in self.feature_names if name not in X.columns]
            raise ValueError(f"the input lacks the features {missing}")
        if getattr(self, "features", None) is not None:
            if "url" not in X.columns:
                raise ValueError("the input lacks the url column that the TF-IDF needs")
            inputs.insert(0, TEXT_COLUMN, url_text(X["url"]))
        return inputs

    def _matrix(self, X):
        inputs = self._inputs(X)
        features = getattr(self, "features", None)
        return inputs if features is None else features.transform(inputs)

    def base_scores(self, X) -> np.ndarray:
        """One column per base model, in the order of ``base_models``: the meta-features."""
        matrix = self._matrix(X)
        return np.column_stack(
            [model.predict_proba(matrix)[:, 1] for model in self.base_models.values()]
        )

    def score(self, X) -> np.ndarray:
        """Meta-model phishing probability, in [0, 1], of each input."""
        return self.meta_model.predict_proba(self.base_scores(X))[:, 1]

    def score_url(self, url: str) -> float:
        """Phishing probability of one URL."""
        return float(self.score([url])[0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Columns: legitimate, phishing."""
        score = self.score(X)
        return np.column_stack([1 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 (phishing) when the score reaches ``threshold``, else 0."""
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
    fold_strategy: str = DEFAULT_FOLD_STRATEGY
    ngram_range: tuple[int, int] | None = OFFICIAL_NGRAM_RANGE


def _plain(value):
    """Search results hold numpy scalars; the report needs plain Python numbers."""
    return value.item() if isinstance(value, np.generic) else value


@dataclass
class CrossValidation:
    """Everything the folds produce, before the final refit."""

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

    ``estimator_factory(name, seed)`` defaults to ``base_estimator``. A pipeline factory fits
    every transformer inside each training fold; its ``search_spaces`` keys carry the step
    prefix.
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
    fold_strategy: str = DEFAULT_FOLD_STRATEGY,
    ngram_range: tuple[int, int] | None = OFFICIAL_NGRAM_RANGE,
) -> FitResult:
    """Tune, stack and refit on ``frame``, the training split only.

    With ``ngram_range`` the refit fits one union on every training row; the vectorizer is
    deterministic, so it equals the union each pipeline would fit.
    """
    columns = feature_columns(frame)
    y = frame["label"].to_numpy().astype(int)
    folds = make_folds(y, frame["group"], n_splits=n_splits, seed=seed, strategy=fold_strategy)
    common = {"n_iter": n_iter, "seed": seed, "n_jobs": n_jobs, "threshold_grid": threshold_grid}
    if ngram_range is None:
        X = frame[columns]
        cv = cross_validate(X, y, folds, **common)
    else:
        ngram_range = tuple(ngram_range)
        X = model_inputs(frame, with_vector=True)
        cv = cross_validate(
            X, y, folds, search_spaces=prefixed_spaces(SEARCH_SPACES), **common,
            estimator_factory=lambda name, model_seed: tfidf_estimator(
                name, model_seed, ngram_range, True
            ),
        )
    best_params = {name: unprefixed(params) for name, params in cv.best_params.items()}

    started = time.perf_counter()
    if ngram_range is None:
        features = None
        base_models = {name: estimator.fit(X, y) for name, estimator in cv.tuned.items()}
    else:
        features = feature_union(ngram_range, with_vector=True)
        matrix = features.fit_transform(X)
        base_models = {
            name: base_estimator(name, seed).set_params(**best_params[name]).fit(matrix, y)
            for name in BASE_MODEL_NAMES
        }
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
        features=features,
        ngram_range=ngram_range,
    )
    return FitResult(
        model=model,
        folds=folds,
        best_params=best_params,
        search_best_f1=cv.search_best_f1,
        oof_base=cv.oof_base,
        oof_meta=cv.oof_meta,
        threshold=cv.threshold,
        cv_metrics=cv.cv_metrics,
        durations={**cv.durations, "refit_seconds": round(refit_seconds, 3)},
        fold_strategy=fold_strategy,
        ngram_range=ngram_range,
    )

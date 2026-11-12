"""Train the Stacking ensemble on the training matrix and save it as a versioned artifact.

Only ``features_train.parquet`` of a feature run is read, after it reproduces the content hash
recorded in its ``features_report.json``. The blind test matrix is left untouched: it belongs to
the final evaluation. The model and ``report.json`` go to a new ``artifacts/<version>/`` folder;
an existing version is never overwritten.

The official model is variant C of the TF-IDF ablation: vector version 2 plus a character TF-IDF
of the URL (n-grams 1 to 3). The cross-validation folds follow the strategy of the split run
(random ``StratifiedKFold`` or ``StratifiedGroupKFold`` by registered domain) unless
``--cv-strategy`` names one. The saved model scores raw URLs.

An official run needs a clean Git working tree. ``--allow-dirty`` runs anyway and marks the
report as an exploration that cannot be cited.
"""

import argparse
import json
import os
import platform
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.split import TRAIN
from sereno_model.features.build import REPORT_NAME as FEATURES_REPORT
from sereno_model.features.build import git_state, matrix_file, matrix_hash
from sereno_model.features.extract import FEATURE_NAMES, FEATURE_SET_VERSION
from sereno_model.models.stacking import (
    BASE_MODEL_NAMES,
    BASE_THRESHOLD,
    DEFAULT_LATENCY_ROWS,
    DEFAULT_LATENCY_WARMUP,
    DEFAULT_N_ITER,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_STEP,
    FOLD_GROUPED,
    FOLD_METHODS,
    FOLD_STRATEGIES,
    LATENCY_SEED_OFFSET,
    OFFICIAL_NGRAM_RANGE,
    SEARCH_SPACES,
    fit_stacking,
    make_threshold_grid,
    measure_latency,
    tfidf_settings,
)

MODEL_NAME = "model.joblib"
REPORT_NAME = "report.json"
MODULE_COMMAND = "python -m sereno_model.models.train"
# Workers that run whole fits in parallel; each estimator is single-threaded, so the result does
# not depend on this number. 16 is the thread count of the official training machine.
DEFAULT_N_JOBS = 16
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

OFFICIAL_RUN = "model selection and training: cross-validation on the training split only"
EXPLORATORY_RUN = "exploration: working tree not known to be clean; not reportable"


class DirtyRepositoryError(RuntimeError):
    """An official run was asked for from a working tree that is not known to be clean."""


def latest_features_run(data_dir: Path) -> str:
    """Name of the newest folder under ``processed/`` that holds a features report."""
    processed = data_dir / "processed"
    runs = [path.parent.name for path in processed.glob(f"*/{FEATURES_REPORT}")]
    if not runs:
        raise LookupError(f"no feature run found under {processed.as_posix()}")
    return max(runs)


def load_training_matrix(data_dir: Path, run: str) -> tuple[pd.DataFrame, dict]:
    """Read the training matrix of a feature run after checking its recorded hash."""
    run_dir = data_dir / "processed" / run
    recorded = json.loads((run_dir / FEATURES_REPORT).read_text(encoding="utf-8"))
    version = recorded["feature_set"]["version"]
    if version != FEATURE_SET_VERSION:
        raise ValueError(
            f"processed/{run} holds feature set version {version}; this code expects "
            f"{FEATURE_SET_VERSION}"
        )
    frame = pd.read_parquet(run_dir / matrix_file(TRAIN))
    expected = recorded["splits"][TRAIN]["content_hash"]["value"]
    found = matrix_hash(frame)
    if found != expected:
        raise ChecksumMismatchError(
            f"processed/{run}/{matrix_file(TRAIN)}: expected content hash {expected}, got {found}"
        )
    other = sorted(set(frame["split"]) - {TRAIN})
    if other:
        raise ValueError(f"processed/{run}/{matrix_file(TRAIN)} holds rows of the splits {other}")
    return frame, recorded


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
    }


def fold_strategy(recorded: dict, requested: str | None) -> tuple[str, str]:
    """The CV fold strategy and where it came from: the argument, else the split run."""
    if requested is not None:
        if requested not in FOLD_STRATEGIES:
            raise ValueError(f"unknown fold strategy {requested!r}")
        return requested, "argument"
    # Split runs older than the strategy option were all grouped by registered domain.
    return recorded["split_run"].get("strategy", FOLD_GROUPED), "split run"


def _round(metrics: dict) -> dict:
    return {
        "threshold": metrics["threshold"],
        "mean": {name: round(value, 6) for name, value in metrics["mean"].items()},
        "per_fold": [
            {name: round(value, 6) for name, value in fold.items()} for fold in metrics["per_fold"]
        ],
    }


def train(
    data_dir: Path,
    artifacts_dir: Path,
    version: str | None = None,
    features_run: str | None = None,
    n_iter: int = DEFAULT_N_ITER,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
    threshold_grid: list[float] | None = None,
    allow_dirty: bool = False,
    now: datetime | None = None,
    repo_dir: Path | None = None,
    command: str | None = None,
    cv_strategy: str | None = None,
    latency_rows: int = DEFAULT_LATENCY_ROWS,
    latency_warmup: int = DEFAULT_LATENCY_WARMUP,
) -> dict:
    """Train on the training matrix and write ``model.joblib`` and ``report.json``.

    The newest feature run is used unless ``features_run`` names one; the version defaults to
    the UTC timestamp of the run.
    """
    started = time.perf_counter()
    git = git_state(repo_dir or REPOSITORY_DIR)
    if git["clean"] is not True and not allow_dirty:
        raise DirtyRepositoryError(
            "the working tree is not known to be clean; commit first or pass --allow-dirty "
            "for a run that cannot be reported"
        )
    moment = now or datetime.now(UTC)
    version = version or moment.strftime("%Y%m%dT%H%M%SZ")
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"invalid model version {version!r}")
    version_dir = artifacts_dir / version
    if version_dir.exists():
        raise FileExistsError(f"{version_dir.as_posix()} already exists; pick a new version")

    grid = make_threshold_grid() if threshold_grid is None else list(threshold_grid)
    run = features_run or latest_features_run(data_dir)
    frame, recorded = load_training_matrix(data_dir, run)
    strategy, strategy_source = fold_strategy(recorded, cv_strategy)
    created_at = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    result = fit_stacking(
        frame,
        n_iter=n_iter,
        n_splits=n_splits,
        seed=seed,
        n_jobs=n_jobs,
        threshold_grid=grid,
        model_version=version,
        created_at=created_at,
        fold_strategy=strategy,
        ngram_range=OFFICIAL_NGRAM_RANGE,
    )
    sample = np.random.default_rng(seed + LATENCY_SEED_OFFSET).permutation(len(frame))
    latency_urls = frame["url"].to_numpy()[sample[: min(latency_rows, len(frame))]]
    clock = time.perf_counter()
    latency = measure_latency(result.model, latency_urls, latency_warmup)
    latency_seconds = round(time.perf_counter() - clock, 3)
    vocabulary = result.model.features.named_transformers_["tfidf"].vocabulary_
    groups = frame["group"].to_numpy()
    labels = frame["label"].to_numpy()

    report = {
        "created_at": created_at,
        "run_type": OFFICIAL_RUN if git["clean"] is True else EXPLORATORY_RUN,
        "reportable": git["clean"] is True,
        "command": command,
        "git": git,
        "seed": seed,
        "model": {
            "version": version,
            "file": MODEL_NAME,
            "class": "sereno_model.models.stacking.StackingModel",
            "input": "a URL, a list of URLs or a frame with a url column; the measured text "
            "and vector v2 are computed from the URL",
            "output": "score(X) is the meta-model phishing probability in [0, 1]; predict(X) "
            "is 1 (phishing) when the score reaches the threshold, else 0 (legitimate); "
            "score_url(url) scores one URL",
            "loading": "joblib.load; needs sereno_model and the library versions below",
        },
        "input": {
            "features_run": f"processed/{run}",
            "train_file": matrix_file(TRAIN),
            "train_rows": len(frame),
            "train_phishing": int((labels == 1).sum()),
            "train_legitimate": int((labels == 0).sum()),
            "train_content_hash": recorded["splits"][TRAIN]["content_hash"]["value"],
            "train_content_hash_verified": True,
            "partition_hash": recorded["split_run"]["partition_hash"],
            "split_run": recorded["split_run"]["dir"],
            "split_seed": recorded["split_run"]["seed"],
            "split_strategy": recorded["split_run"].get("strategy", FOLD_GROUPED),
            "assemble_run": recorded["assemble_run"],
            "test_split": "not read",
        },
        "feature_set": {
            "version": FEATURE_SET_VERSION,
            "names": list(FEATURE_NAMES),
            "tfidf": tfidf_settings(OFFICIAL_NGRAM_RANGE),
            "tfidf_vocabulary_size": len(vocabulary),
            "matrix": "scipy sparse CSR: TF-IDF columns followed by the 24 vector columns",
        },
        "cv": {
            "strategy": strategy,
            "strategy_source": strategy_source,
            "method": FOLD_METHODS[strategy],
            "n_splits": n_splits,
            "seed": seed,
            "strategy_note": "random folds as in the papers (P02 and others); grouped folds by "
            "registered domain are the robustness analysis",
            "reuse": "the same folds serve the search, the out-of-fold predictions and the "
            "threshold",
            "folds": [
                {
                    "validation_rows": len(val),
                    "validation_phishing": int(labels[val].sum()),
                    "validation_groups": len(set(groups[val])),
                }
                for _, val in result.folds
            ],
        },
        "search": {
            "method": "RandomizedSearchCV, scoring F1, over the CV folds; each candidate is a "
            "pipeline whose TF-IDF is fitted on the training rows of the fold",
            "method_source": "P09",
            "n_iter": n_iter,
            "n_iter_source": "own decision; the corpus does not give a search budget",
            "spaces": {
                name: {key: [min(values), max(values)] for key, values in space.items()}
                for name, space in SEARCH_SPACES.items()
            },
            "spaces_source": "P09 ranges, both ends included; learning rates on a 0.01 grid "
            "(own decision)",
        },
        "base_models": {
            name: {
                "best_params": result.best_params[name],
                "search_best_f1": round(result.search_best_f1[name], 6),
                "cv": _round(result.cv_metrics[name]),
                "cv_note": f"out-of-fold probabilities at threshold {BASE_THRESHOLD}",
            }
            for name in BASE_MODEL_NAMES
        },
        "meta_model": {
            "estimator": "LogisticRegression (scikit-learn defaults)",
            "source": "P27",
            "inputs": [f"oof_probability_{name}" for name in BASE_MODEL_NAMES],
            "inputs_source": "out-of-fold predictions on the same folds (P15)",
            "coefficients": [round(float(value), 6) for value in result.model.meta_model.coef_[0]],
            "intercept": round(float(result.model.meta_model.intercept_[0]), 6),
        },
        "stacking": {
            "cv": _round(result.cv_metrics["stacking"]),
            "cv_note": "out-of-fold meta-model scores at the chosen threshold; the threshold "
            "was chosen on these same folds, so these figures are slightly optimistic",
        },
        "threshold": {
            "value": result.threshold["threshold"],
            "mean_f1": round(result.threshold["mean_f1"], 6),
            "fold_f1": [round(value, 6) for value in result.threshold["fold_f1"]],
            "rule": "highest mean F1 over the folds of the meta-model's out-of-fold scores; "
            "ties go to the value closest to 0.5; a score equal to the threshold is phishing",
            "rule_source": "P17",
            "grid": grid,
            "grid_source": "own decision",
            "test_data_used": False,
        },
        "not_applied": {
            "calibration": "not applied; the corpus does not cover it",
            "resampling": "not applied; the classes are balanced since phase 1",
        },
        "latency_ms": latency,
        "latency_sample": f"{len(latency_urls)} training URLs drawn with seed "
        f"{seed + LATENCY_SEED_OFFSET}",
        "latency_requirement": "RNF-01: mean <= 200 ms end to end (this measures the model only)",
        "parallelism": {
            "n_jobs": n_jobs,
            "estimator_threads": 1,
            "note": "workers run whole fits; results do not depend on n_jobs",
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": _versions(),
    }

    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(result.model, version_dir / MODEL_NAME)
    report["model_size_bytes"] = (version_dir / MODEL_NAME).stat().st_size
    report["durations"] = {
        **result.durations,
        "latency_seconds": latency_seconds,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    (version_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _print_report(report: dict, artifacts_dir: Path) -> None:
    source = report["input"]
    print(
        f"train matrix {source['features_run']}/{source['train_file']}: {source['train_rows']} "
        f"rows, sha256 {source['train_content_hash']}"
    )
    for name, entry in report["base_models"].items():
        mean = entry["cv"]["mean"]
        print(f"{name:<14} CV F1 {mean['f1']:.4f} params {entry['best_params']}")
    mean = report["stacking"]["cv"]["mean"]
    print(
        f"stacking       CV F1 {mean['f1']:.4f} precision {mean['precision']:.4f} "
        f"recall {mean['recall']:.4f} accuracy {mean['accuracy']:.4f}"
    )
    print(f"threshold {report['threshold']['value']} (mean CV F1 {report['threshold']['mean_f1']})")
    print(f"folds {report['cv']['strategy']} ({report['cv']['strategy_source']}); latency p50 "
          f"{report['latency_ms']['p50']} ms, p95 {report['latency_ms']['p95']} ms; size "
          f"{report['model_size_bytes']} bytes")
    print(f"git commit {report['git']['commit']}, clean {report['git']['clean']}")
    if not report["reportable"]:
        print("warning: exploratory run; it cannot be reported")
    print(f"artifact: {(artifacts_dir / report['model']['version']).as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--features-run", help="timestamp of the feature run (default: newest)")
    parser.add_argument("--version", help="model version and folder name (default: timestamp)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument(
        "--n-iter", type=int, default=DEFAULT_N_ITER, help="candidates per base model (own decision)"
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=DEFAULT_THRESHOLD_STEP,
        help="spacing of the candidate thresholds in (0, 1) (own decision)",
    )
    parser.add_argument(
        "--allow-dirty", action="store_true", help="run from an uncommitted tree; not reportable"
    )
    parser.add_argument(
        "--cv-strategy",
        choices=FOLD_STRATEGIES,
        help="CV folds: random (StratifiedKFold) or grouped by registered domain (default: the "
        "strategy of the split run)",
    )
    parser.add_argument("--latency-rows", type=int, default=DEFAULT_LATENCY_ROWS)
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_LATENCY_WARMUP)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)

    command = " ".join([MODULE_COMMAND, *arguments])
    report = train(
        args.data_dir,
        args.artifacts_dir,
        version=args.version,
        features_run=args.features_run,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        threshold_grid=make_threshold_grid(args.threshold_step),
        allow_dirty=args.allow_dirty,
        command=command,
        cv_strategy=args.cv_strategy,
        latency_rows=args.latency_rows,
        latency_warmup=args.latency_warmup,
    )
    _print_report(report, args.artifacts_dir)


if __name__ == "__main__":
    main()

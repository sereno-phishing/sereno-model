"""Measure the paper-style setup: plain stratified folds and a character TF-IDF of the URL.

Exploration run, not citable. Variants A to E and results: docs/03-modelos.md.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sereno_model.data.split import TRAIN
from sereno_model.features.build import git_state, matrix_file
from sereno_model.models.diagnose import (
    _official_reference,
    capped_domains,
    error_analysis,
)
from sereno_model.models.stacking import (  # noqa: F401 (re-exported building blocks)
    BASE_MODEL_NAMES,
    DEFAULT_LATENCY_ROWS,
    DEFAULT_LATENCY_WARMUP,
    DEFAULT_N_ITER,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_STEP,
    FEATURE_SET_VERSION,
    FOLD_GROUPED,
    FOLD_RANDOM,
    LATENCY_SEED_OFFSET,
    MAX_FEATURES,
    MODEL_STEP,
    SEARCH_SPACES,
    TEXT_COLUMN,
    StackingModel,
    base_estimator,
    cross_validate,
    feature_columns,
    feature_union,
    make_folds,
    make_threshold_grid,
    measure_latency,
    meta_estimator,
    model_inputs,
    prefixed_spaces,
    serialized_size,
    tfidf_estimator,
    tfidf_vectorizer,
    url_text,
)
from sereno_model.models.stacking import unprefixed as _unprefixed
from sereno_model.models.train import _versions, latest_features_run, load_training_matrix

REPORT_NAME = "report.json"
MODULE_COMMAND = "python -m sereno_model.models.tfidf_ablation"
DEFAULT_N_JOBS = 16
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
RUN_TYPE = (
    "exploration: paper-style evaluation (plain stratified folds, character TF-IDF of the URL "
    "as in P02) on the training split only; not citable as a final result"
)
BASELINE = "A"
GROUPED_FOLDS = "StratifiedGroupKFold by registered domain (as in 1.0.0)"
PLAIN_FOLDS = "StratifiedKFold, shuffled, groups ignored (paper-style)"
# Own decision: the corpus does not give the n-gram ranges of a character TF-IDF of the URL.
VARIANTS = {
    "A": {"with_vector": True, "ngram_range": None, "grouped": False},
    "B": {"with_vector": False, "ngram_range": (1, 3), "grouped": False},
    "C": {"with_vector": True, "ngram_range": (1, 3), "grouped": False},
    "D": {"with_vector": True, "ngram_range": (3, 5), "grouped": False},
    "E": {"with_vector": True, "ngram_range": "best of C and D", "grouped": True},
}


def _variant_entry(cv, spec: dict, folds_label: str, n_iter: int, frame, capped) -> dict:
    analysis = error_analysis(frame, cv, capped)
    analysis["scores"] = "out-of-fold meta-model scores of this variant's folds"
    ngram = spec["ngram_range"]
    return {
        "inputs": ("vector v2 (24)" if spec["with_vector"] else "")
        + (" + " if spec["with_vector"] and ngram else "")
        + (f"char TF-IDF {ngram[0]}-{ngram[1]} (max {MAX_FEATURES})" if ngram else ""),
        "ngram_range": list(ngram) if ngram else None,
        "folds": folds_label,
        "n_iter": n_iter,
        "best_params": {name: _unprefixed(params) for name, params in cv.best_params.items()},
        "search_best_f1": {name: round(value, 6) for name, value in cv.search_best_f1.items()},
        "cv_mean": {
            name: {metric: round(value, 6) for metric, value in entry["mean"].items()}
            for name, entry in cv.cv_metrics.items()
        },
        "fold_f1": {
            name: [round(fold["f1"], 6) for fold in entry["per_fold"]]
            for name, entry in cv.cv_metrics.items()
        },
        "stacking_fold_f1": [
            round(fold["f1"], 6) for fold in cv.cv_metrics["stacking"]["per_fold"]
        ],
        "threshold": cv.threshold["threshold"],
        "error_analysis": analysis,
        "durations": cv.durations,
    }


def _delta(entry: dict, reference: dict) -> dict:
    delta = {
        f"{name}_f1": round(entry["cv_mean"][name]["f1"] - reference["cv_mean"][name]["f1"], 6)
        for name in (*BASE_MODEL_NAMES, "stacking")
    }
    stacking, base = entry["cv_mean"]["stacking"], reference["cv_mean"]["stacking"]
    for metric in ("precision", "recall", "accuracy"):
        delta[f"stacking_{metric}"] = round(stacking[metric] - base[metric], 6)
    return delta


def refit_best(frame: pd.DataFrame, cv, spec: dict, seed: int) -> tuple[StackingModel, float]:
    """Refit one feature union and the tuned base models on every training row.

    The vectorizer is deterministic, so this union equals the one each pipeline would fit.
    """
    started = time.perf_counter()
    y = frame["label"].to_numpy().astype(int)
    features = feature_union(spec["ngram_range"], spec["with_vector"])
    matrix = features.fit_transform(model_inputs(frame, spec["with_vector"]))
    base_models = {
        name: base_estimator(name, seed).set_params(**_unprefixed(cv.best_params[name])).fit(
            matrix, y
        )
        for name in BASE_MODEL_NAMES
    }
    meta_model = meta_estimator(seed).fit(cv.meta_features, y)
    model = StackingModel(
        base_models,
        meta_model,
        cv.threshold["threshold"],
        feature_columns(frame) if spec["with_vector"] else [],
        FEATURE_SET_VERSION,
        features=features,
        ngram_range=spec["ngram_range"],
    )
    return model, round(time.perf_counter() - started, 3)


def tfidf_ablation(
    data_dir: Path,
    out_dir: Path,
    features_run: str | None = None,
    n_iter: int = DEFAULT_N_ITER,
    n_iter_tfidf: int = DEFAULT_N_ITER,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
    threshold_grid: list[float] | None = None,
    latency_rows: int = DEFAULT_LATENCY_ROWS,
    latency_warmup: int = DEFAULT_LATENCY_WARMUP,
    now: datetime | None = None,
    repo_dir: Path | None = None,
    command: str | None = None,
) -> dict:
    """Cross-validate the five variants on the training matrix and write ``report.json``."""
    started = time.perf_counter()
    repo_dir = repo_dir or REPOSITORY_DIR
    git = git_state(repo_dir)
    moment = now or datetime.now(UTC)
    run_dir = out_dir / moment.strftime("%Y%m%dT%H%M%SZ")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir.as_posix()} already exists; diagnostics never overwrite")

    grid = make_threshold_grid() if threshold_grid is None else list(threshold_grid)
    run = features_run or latest_features_run(data_dir)
    frame, recorded = load_training_matrix(data_dir, run)
    y = frame["label"].to_numpy().astype(int)
    groups = frame["group"].to_numpy()
    plain = make_folds(y, groups, n_splits=n_splits, seed=seed, strategy=FOLD_RANDOM)
    grouped = make_folds(y, groups, n_splits=n_splits, seed=seed, strategy=FOLD_GROUPED)
    capped = capped_domains(data_dir, recorded)
    inputs = {flag: model_inputs(frame, flag) for flag in (True, False)}

    results, fits = {}, {}

    def run_variant(name: str, spec: dict) -> None:
        folds, label = (grouped, GROUPED_FOLDS) if spec["grouped"] else (plain, PLAIN_FOLDS)
        clock = time.perf_counter()
        if spec["ngram_range"] is None:
            iterations = n_iter
            cv = cross_validate(
                frame[feature_columns(frame)], y, folds, search_spaces=SEARCH_SPACES,
                n_iter=iterations, seed=seed, n_jobs=n_jobs, threshold_grid=grid,
            )
        else:
            iterations = n_iter_tfidf
            ngram, with_vector = spec["ngram_range"], spec["with_vector"]
            cv = cross_validate(
                inputs[with_vector], y, folds, search_spaces=prefixed_spaces(SEARCH_SPACES),
                n_iter=iterations, seed=seed, n_jobs=n_jobs, threshold_grid=grid,
                estimator_factory=lambda model, model_seed: tfidf_estimator(
                    model, model_seed, ngram, with_vector
                ),
            )
        cv.durations["total_seconds"] = round(time.perf_counter() - clock, 3)
        results[name] = _variant_entry(cv, spec, label, iterations, frame, capped)
        fits[name] = (cv, spec)

    for name in ("A", "B", "C", "D"):
        run_variant(name, VARIANTS[name])
        print(f"{name} done: stacking CV F1 {results[name]['cv_mean']['stacking']['f1']:.4f} "
              f"in {results[name]['durations']['total_seconds']} s", flush=True)

    def stacking_f1(name: str) -> float:
        return results[name]["cv_mean"]["stacking"]["f1"]

    best = "C" if stacking_f1("C") >= stacking_f1("D") else "D"
    run_variant("E", {**VARIANTS[best], "grouped": True})
    results["E"]["source_variant"] = best

    official = _official_reference(repo_dir)
    for entry in results.values():
        entry["delta_vs_A"] = _delta(entry, results[BASELINE])
        if official is not None:
            entry["stacking_f1_delta_vs_grouped_1_0_0"] = round(
                entry["cv_mean"]["stacking"]["f1"] - official["stacking_cv_f1"], 6
            )

    best_cv, best_spec = fits[best]
    model, refit_seconds = refit_best(frame, best_cv, best_spec, seed)
    sample = np.random.default_rng(seed + LATENCY_SEED_OFFSET).permutation(len(frame))
    latency_urls = frame["url"].to_numpy()[sample[: min(latency_rows, len(frame))]]
    clock = time.perf_counter()
    latency = measure_latency(model, latency_urls, latency_warmup)
    latency_seconds = round(time.perf_counter() - clock, 3)
    size = serialized_size(model)
    vocabulary = model.features.named_transformers_["tfidf"].vocabulary_

    report = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_type": RUN_TYPE,
        "reportable": False,
        "reportable_note": "an exploration is never a final result; the working tree state is "
        "recorded under git",
        "command": command,
        "git": git,
        "seed": seed,
        "input": {
            "features_run": f"processed/{run}",
            "train_file": matrix_file(TRAIN),
            "train_rows": len(frame),
            "train_phishing": int((y == 1).sum()),
            "train_legitimate": int((y == 0).sum()),
            "train_content_hash": recorded["splits"][TRAIN]["content_hash"]["value"],
            "train_content_hash_verified": True,
            "partition_hash": recorded["split_run"]["partition_hash"],
            "test_split": "not read",
        },
        "method": {
            "n_splits": n_splits,
            "n_iter_vector_only": n_iter,
            "n_iter_tfidf": n_iter_tfidf,
            "spaces_source": "P09",
            "threshold_grid": {"min": min(grid), "max": max(grid), "count": len(grid)},
            "tfidf": {
                "analyzer": "char",
                "max_features": MAX_FEATURES,
                "max_features_source": "P02",
                "lowercase": True,
                "norm": "l2",
                "use_idf": True,
                "smooth_idf": True,
                "sublinear_tf": False,
                "transformer_params_source": "P32 (also the scikit-learn defaults)",
                "ngram_ranges_source": "own decision; the corpus does not give them",
                "text": "measured text of vector v2: normalized URL without scheme and separator, "
                "leading 'www.' labels, default port and trailing slashes (own decision, "
                "consistent with the v2 source-bias decision)",
                "fitted_on": "training rows of each fold only (pipeline inside the search and "
                "the out-of-fold predictions)",
            },
            "matrix": "scipy sparse CSR; vector columns joined as sparse columns",
        },
        "variants": results,
        "best_variant": {
            "name": best,
            "rule": "higher plain-fold stacking CV F1 of C and D",
            "grouped_variant": "E",
            "stacking_f1_plain": stacking_f1(best),
            "stacking_f1_grouped": stacking_f1("E"),
            "stacking_f1_grouped_minus_plain": round(stacking_f1("E") - stacking_f1(best), 6),
            "refit_on": "all training rows, best params of the plain-fold search",
            "tfidf_vocabulary_size": len(vocabulary),
            "latency_ms": latency,
            "latency_sample": f"{len(latency_urls)} training URLs drawn with seed "
            f"{seed + LATENCY_SEED_OFFSET}",
            "model_size_bytes": size,
            "model_size_note": "joblib.dump of the feature union, three base models, meta-model "
            "and threshold, in a temporary directory outside the repository, then deleted",
            "requirement": "RNF-01: mean <= 200 ms end to end (this measures the model only)",
        },
        "official_1_0_0": official,
        "parallelism": {"n_jobs": n_jobs, "estimator_threads": 1},
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": _versions(),
    }
    report["durations"] = {
        **{f"{name}_seconds": entry["durations"]["total_seconds"] for name, entry in
           results.items()},
        "refit_seconds": refit_seconds,
        "latency_seconds": latency_seconds,
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _print_report(report: dict, run_dir: Path) -> None:
    for name, entry in report["variants"].items():
        cells = " ".join(
            f"{model} {entry['cv_mean'][model]['f1']:.4f}"
            for model in (*BASE_MODEL_NAMES, "stacking")
        )
        print(f"{name} [{entry['folds'][:20]}] CV F1: {cells} (threshold {entry['threshold']}, "
              f"delta vs A {entry['delta_vs_A']['stacking_f1']:+.4f})")
    best = report["best_variant"]
    latency = best["latency_ms"]
    print(f"best {best['name']}: latency p50 {latency['p50']} ms, p95 {latency['p95']} ms; "
          f"size {best['model_size_bytes']} bytes")
    print(f"total {report['durations']['total_seconds']} s; report not citable (exploration)")
    print(f"report: {run_dir.as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "diagnostics")
    parser.add_argument("--features-run", help="timestamp of the feature run (default: newest)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER,
                        help="candidates per search for the vector-only variant A")
    parser.add_argument("--n-iter-tfidf", type=int, default=DEFAULT_N_ITER,
                        help="candidates per search for the TF-IDF variants B to E")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    parser.add_argument("--latency-rows", type=int, default=DEFAULT_LATENCY_ROWS)
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_LATENCY_WARMUP)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)

    moment = datetime.now(UTC)
    report = tfidf_ablation(
        args.data_dir,
        args.out_dir,
        features_run=args.features_run,
        n_iter=args.n_iter,
        n_iter_tfidf=args.n_iter_tfidf,
        n_splits=args.n_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        threshold_grid=make_threshold_grid(args.threshold_step),
        latency_rows=args.latency_rows,
        latency_warmup=args.latency_warmup,
        now=moment,
        command=" ".join([MODULE_COMMAND, *arguments]),
    )
    _print_report(report, args.out_dir / moment.strftime("%Y%m%dT%H%M%SZ"))


if __name__ == "__main__":
    main()

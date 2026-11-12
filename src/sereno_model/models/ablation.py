"""Measure the candidate list-based feature groups against the base vector, on training only.

Exploration run: its figures are not citable as a final result. Only ``features_train.parquet``
of a feature run is read, through the same hash-checked loader as ``train``; the blind test
matrix is never opened. The candidate columns (``sereno_model.features.candidates``) are
computed from the ``url`` column with the same parser as the base vector, and each variant runs
the approved cross-validation (P09 ranges, same ``n_iter``, grouped folds, same seed):

* ``v2``: the 24 features of the base vector;
* ``v2+G1``, ``v2+G2``, ``v2+G3``: the base vector plus one candidate group;
* ``v2+G1+G2+G3``: the base vector plus every candidate group.

Every variant uses the same folds. The report also gives the prevalence of each candidate
feature by class and by source, to flag features that identify a source rather than phishing.
It never holds a URL: only registered domains and counts (RNF-05). Each run writes a new
``<out-dir>/<UTC timestamp>/report.json`` and never overwrites one.
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

from sereno_model.data.normalize import parse_url
from sereno_model.data.split import TRAIN
from sereno_model.features.build import git_state, matrix_file
from sereno_model.features.candidates import (
    CANDIDATE_GROUPS,
    CANDIDATE_LIST_VERSION,
    CANDIDATE_NAMES,
    SENSITIVE_WORDS,
    SHORTENER_DOMAINS,
    candidates_from_parts,
)
from sereno_model.models.diagnose import _sources, capped_domains, error_analysis
from sereno_model.models.stacking import (
    BASE_MODEL_NAMES,
    DEFAULT_N_ITER,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_STEP,
    SEARCH_SPACES,
    cross_validate,
    feature_columns,
    grouped_folds,
    make_threshold_grid,
)
from sereno_model.models.train import _versions, latest_features_run, load_training_matrix

REPORT_NAME = "report.json"
MODULE_COMMAND = "python -m sereno_model.models.ablation"
DEFAULT_N_JOBS = 16
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
RUN_TYPE = (
    "exploration: ablation of the candidate list-based feature groups on the training split "
    "only; not citable as a final result"
)
BASELINE = "v2"
VARIANTS = {
    "v2": (),
    "v2+G1": ("G1",),
    "v2+G2": ("G2",),
    "v2+G3": ("G3",),
    "v2+G1+G2+G3": ("G1", "G2", "G3"),
}
# Own heuristic, not from the corpus: a feature is flagged when, within one class, its non-zero
# share differs by more than this between two sources with at least SOURCE_MIN_ROWS rows.
SOURCE_SPREAD_FLAG = 0.2
SOURCE_MIN_ROWS = 100
CLASSES = {"phishing": 1, "legitimate": 0}


def candidate_columns(urls) -> pd.DataFrame:
    """The candidate features of every URL, parsed exactly as the base vector parses them."""
    urls = pd.Series(urls)
    rows = [candidates_from_parts(parse_url(url)) for url in urls]
    return pd.DataFrame(rows, columns=list(CANDIDATE_NAMES), index=urls.index).astype("int32")


def _share(mask: np.ndarray, values: np.ndarray) -> float | None:
    rows = int(mask.sum())
    return round(float((values[mask] != 0).mean()), 6) if rows else None


def prevalence(columns: pd.DataFrame, labels, sources) -> dict:
    """Share of rows with a non-zero value, by class and by source within each class."""
    labels = np.asarray(labels).astype(int)
    sources = [_sources(row) for row in sources]
    names = sorted({source for row in sources for source in row})
    table = {}
    for feature in columns.columns:
        values = columns[feature].to_numpy()
        by_source = {}
        for source in names:
            in_source = np.array([source in row for row in sources])
            by_source[source] = {
                label_name: {
                    "rows": int((in_source & (labels == label)).sum()),
                    "nonzero_share": _share(in_source & (labels == label), values),
                }
                for label_name, label in CLASSES.items()
            }
        spread = {}
        for label_name in CLASSES:
            shares = [
                entry[label_name]["nonzero_share"]
                for entry in by_source.values()
                if entry[label_name]["rows"] >= SOURCE_MIN_ROWS
            ]
            spread[label_name] = round(max(shares) - min(shares), 6) if len(shares) > 1 else None
        table[feature] = {
            "by_class": {name: _share(labels == label, values) for name, label in CLASSES.items()},
            "mean_by_class": {
                name: round(float(values[labels == label].mean()), 6) if (labels == label).any()
                else None
                for name, label in CLASSES.items()
            },
            "by_source": by_source,
            "source_spread_within_class": spread,
            "source_identification_flag": any(
                value is not None and value > SOURCE_SPREAD_FLAG for value in spread.values()
            ),
        }
    return table


def _variant_entry(cv, added: list[str], n_features: int, frame: pd.DataFrame, capped) -> dict:
    return {
        "added_features": added,
        "n_features": n_features,
        "best_params": cv.best_params,
        "search_best_f1": {name: round(value, 6) for name, value in cv.search_best_f1.items()},
        "cv_mean": {
            name: {metric: round(value, 6) for metric, value in entry["mean"].items()}
            for name, entry in cv.cv_metrics.items()
        },
        "stacking_fold_f1": [
            round(fold["f1"], 6) for fold in cv.cv_metrics["stacking"]["per_fold"]
        ],
        "threshold": cv.threshold["threshold"],
        "error_analysis": error_analysis(frame, cv, capped),
        "durations": cv.durations,
    }


def _delta(entry: dict, baseline: dict) -> dict:
    delta = {
        f"{name}_f1": round(entry["cv_mean"][name]["f1"] - baseline["cv_mean"][name]["f1"], 6)
        for name in (*BASE_MODEL_NAMES, "stacking")
    }
    stacking, base = entry["cv_mean"]["stacking"], baseline["cv_mean"]["stacking"]
    for metric in ("precision", "recall", "accuracy"):
        delta[f"stacking_{metric}"] = round(stacking[metric] - base[metric], 6)
    return delta


def ablation(
    data_dir: Path,
    out_dir: Path,
    features_run: str | None = None,
    variants=None,
    n_iter: int = DEFAULT_N_ITER,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
    threshold_grid: list[float] | None = None,
    now: datetime | None = None,
    repo_dir: Path | None = None,
    command: str | None = None,
) -> dict:
    """Cross-validate every variant on the training matrix and write ``report.json``."""
    started = time.perf_counter()
    names = list(VARIANTS) if variants is None else list(variants)
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    if BASELINE not in names:
        names.insert(0, BASELINE)
    repo_dir = repo_dir or REPOSITORY_DIR
    git = git_state(repo_dir)
    moment = now or datetime.now(UTC)
    run_dir = out_dir / moment.strftime("%Y%m%dT%H%M%SZ")
    if run_dir.exists():
        raise FileExistsError(f"{run_dir.as_posix()} already exists; diagnostics never overwrite")

    grid = make_threshold_grid() if threshold_grid is None else list(threshold_grid)
    run = features_run or latest_features_run(data_dir)
    frame, recorded = load_training_matrix(data_dir, run)
    base = frame[feature_columns(frame)]
    clock = time.perf_counter()
    candidates = candidate_columns(frame["url"])
    candidate_seconds = round(time.perf_counter() - clock, 3)
    y = frame["label"].to_numpy().astype(int)
    folds = grouped_folds(y, frame["group"].to_numpy(), n_splits=n_splits, seed=seed)
    capped = capped_domains(data_dir, recorded)

    results = {}
    for name in names:
        added = [feature for group in VARIANTS[name] for feature in CANDIDATE_GROUPS[group]]
        X = pd.concat([base, candidates[added]], axis=1) if added else base
        clock = time.perf_counter()
        cv = cross_validate(
            X, y, folds, search_spaces=SEARCH_SPACES, n_iter=n_iter, seed=seed, n_jobs=n_jobs,
            threshold_grid=grid,
        )
        cv.durations["total_seconds"] = round(time.perf_counter() - clock, 3)
        results[name] = _variant_entry(cv, added, X.shape[1], frame, capped)
    for entry in results.values():
        entry["delta_vs_v2"] = _delta(entry, results[BASELINE])

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
            "folds": "StratifiedGroupKFold by registered domain (as in 1.0.0), same folds for "
            "every variant",
            "n_splits": n_splits,
            "n_iter": n_iter,
            "spaces_source": "P09",
            "threshold_grid": {"min": min(grid), "max": max(grid), "count": len(grid)},
        },
        "lists": {
            "version": CANDIDATE_LIST_VERSION,
            "groups": {group: list(features) for group, features in CANDIDATE_GROUPS.items()},
            "sensitive_words": [
                {"word": word, "citations": list(cites)} for word, cites in SENSITIVE_WORDS
            ],
            "shortener_domains": [
                {"domain": domain, "citations": list(cites)} for domain, cites in SHORTENER_DOMAINS
            ],
        },
        "variants": results,
        "prevalence": prevalence(candidates, y, frame["sources"]),
        "prevalence_note": f"share of rows with a non-zero value; a feature is flagged when, "
        f"within one class, two sources with at least {SOURCE_MIN_ROWS} rows differ by more "
        f"than {SOURCE_SPREAD_FLAG} (own heuristic). A row with several sources counts in each",
        "parallelism": {"n_jobs": n_jobs, "estimator_threads": 1},
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": _versions(),
    }
    report["durations"] = {
        "candidate_extraction_seconds": candidate_seconds,
        **{
            f"{name}_seconds": entry["durations"]["total_seconds"]
            for name, entry in results.items()
        },
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _print_report(report: dict, run_dir: Path) -> None:
    for name, entry in report["variants"].items():
        stacking = entry["cv_mean"]["stacking"]
        print(
            f"{name:<13} stacking CV F1 {stacking['f1']:.4f} "
            f"(delta {entry['delta_vs_v2']['stacking_f1']:+.4f}, threshold {entry['threshold']})"
        )
    flagged = [name for name, entry in report["prevalence"].items()
               if entry["source_identification_flag"]]
    print(f"source-identification flags: {flagged or 'none'}")
    print(f"total {report['durations']['total_seconds']} s; report not citable (exploration)")
    print(f"report: {run_dir.as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "diagnostics")
    parser.add_argument("--features-run", help="timestamp of the feature run (default: newest)")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)

    moment = datetime.now(UTC)
    report = ablation(
        args.data_dir,
        args.out_dir,
        features_run=args.features_run,
        variants=args.variants,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        threshold_grid=make_threshold_grid(args.threshold_step),
        now=moment,
        command=" ".join([MODULE_COMMAND, *arguments]),
    )
    _print_report(report, args.out_dir / moment.strftime("%Y%m%dT%H%M%SZ"))


if __name__ == "__main__":
    main()

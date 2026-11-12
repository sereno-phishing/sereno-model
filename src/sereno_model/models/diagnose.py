"""Diagnose the gap between the cross-validated F1 and the ADD target, on the training split only.

Exploration run: its figures are not citable as a final result. Only ``features_train.parquet``
of a feature run is read, through the same hash-checked loader as ``train``; the blind test
matrix is never opened. Three measurements:

1. Grouping effect: the approved pipeline (P09 ranges, same ``n_iter``) over the folds grouped
   by registered domain, as in the official run, and over plain stratified folds that ignore
   the groups.
2. Range effect: the grouped folds with wider, own exploratory ranges (not from the corpus).
3. Error analysis of the grouped stacking's out-of-fold predictions: confusion counts, subgroup
   metrics by source, by synthetic flag and by the domains capped in the assemble step (the
   shared platforms and shorteners of ``docs/01-datos.md``), the most frequent registered
   domains among the errors, and how redundant the base models' errors are.

The report never holds a URL: only registered domains and counts (RNF-05). Each run writes a new
``<out-dir>/<UTC timestamp>/report.json`` and never overwrites one.
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sereno_model.data.split import TRAIN
from sereno_model.features.build import git_state, matrix_file
from sereno_model.models.stacking import (
    BASE_MODEL_NAMES,
    BASE_THRESHOLD,
    DEFAULT_N_ITER,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_STEP,
    SEARCH_SPACES,
    cross_validate,
    feature_columns,
    FOLD_GROUPED,
    FOLD_RANDOM,
    make_folds,
    make_threshold_grid,
)
from sereno_model.models.train import (
    _versions,
    latest_features_run,
    load_training_matrix,
)

REPORT_NAME = "report.json"
MODULE_COMMAND = "python -m sereno_model.models.diagnose"
DEFAULT_N_JOBS = 16
REPOSITORY_DIR = Path(__file__).resolve().parents[3]
RUN_TYPE = (
    "exploration: diagnostic of the gap to the ADD target on the training split only; "
    "not citable as a final result"
)
OFFICIAL_REPORT = Path("artifacts") / "1.0.0" / "report.json"
TOP_K = 10

# Learning rates 0.03 to 0.30 on a 0.01 grid; scipy is not a declared dependency.
_WIDE_LEARNING_RATES = [round(0.03 + 0.01 * step, 2) for step in range(28)]

# Own exploratory ranges, wider than P09; not taken from the corpus.
WIDE_SEARCH_SPACES = {
    "random_forest": {
        "n_estimators": list(range(100, 501)),
        "max_depth": [10, 20, 30, None],
        "min_samples_leaf": list(range(1, 6)),
    },
    "xgboost": {
        "n_estimators": list(range(100, 601)),
        "max_depth": list(range(4, 13)),
        "learning_rate": _WIDE_LEARNING_RATES,
    },
    "lightgbm": {
        "n_estimators": list(range(100, 601)),
        "max_depth": [-1, *range(8, 17)],
        "num_leaves": list(range(31, 256)),
        "learning_rate": _WIDE_LEARNING_RATES,
    },
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def confusion(y, predicted) -> dict:
    y, predicted = np.asarray(y).astype(int), np.asarray(predicted).astype(int)
    return {
        "tp": int(((y == 1) & (predicted == 1)).sum()),
        "fp": int(((y == 0) & (predicted == 1)).sum()),
        "tn": int(((y == 0) & (predicted == 0)).sum()),
        "fn": int(((y == 1) & (predicted == 0)).sum()),
    }


def rates(counts: dict) -> dict:
    """F1, precision, recall and false-positive rate; ``None`` when a rate is undefined."""
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    return {
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
    }


def subgroup_breakdown(y, predicted, keys) -> dict:
    """Metrics per subgroup; a row with several keys counts in each of them."""
    y, predicted = np.asarray(y).astype(int), np.asarray(predicted).astype(int)
    members: dict[str, list[int]] = {}
    for index, row_keys in enumerate(keys):
        for key in dict.fromkeys(row_keys):
            members.setdefault(str(key), []).append(index)
    table = {}
    for key in sorted(members):
        index = np.array(members[key])
        counts = confusion(y[index], predicted[index])
        table[key] = {
            "rows": len(index),
            "phishing": int(y[index].sum()),
            "legitimate": int((y[index] == 0).sum()),
            **counts,
            **rates(counts),
        }
    return table


def top_domains(groups, mask, k: int = TOP_K) -> list[dict]:
    """The ``k`` most frequent registered domains among the masked rows, with their counts."""
    counter = Counter(np.asarray(groups)[np.asarray(mask, dtype=bool)].tolist())
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [{"domain": domain, "count": count} for domain, count in ranked]


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 6)


def base_model_correlation(y, oof_base: dict, threshold: float = BASE_THRESHOLD) -> dict:
    """How alike the base models' out-of-fold probabilities and errors are."""
    y = np.asarray(y).astype(int)
    names = [name for name in BASE_MODEL_NAMES if name in oof_base]
    wrong = {
        name: ((np.asarray(oof_base[name]) >= threshold).astype(int) != y).astype(int)
        for name in names
    }
    wrong_count = np.sum([wrong[name] for name in names], axis=0)
    return {
        "threshold": threshold,
        "probability_pearson": {
            f"{a}|{b}": _pearson(np.asarray(oof_base[a]), np.asarray(oof_base[b]))
            for a, b in combinations(names, 2)
        },
        "error_pearson": {
            f"{a}|{b}": _pearson(wrong[a], wrong[b]) for a, b in combinations(names, 2)
        },
        "rows_wrong_by_n_models": {
            str(n): int((wrong_count == n).sum()) for n in range(len(names) + 1)
        },
    }


def capped_domains(data_dir: Path, recorded: dict) -> list[str] | None:
    """Domains capped in the assemble step, from its report; ``None`` when it is missing."""
    path = data_dir / recorded.get("assemble_run", {}).get("dir", "") / "assemble_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return [entry["domain"] for entry in report["phishing_domain_cap"]["capped_domains"]]


def _sources(value) -> list[str]:
    return [str(item) for item in value]


def error_analysis(frame: pd.DataFrame, cv, capped: list[str] | None) -> dict:
    """Errors of the stacking's out-of-fold scores at the threshold chosen on the same folds."""
    y = frame["label"].to_numpy().astype(int)
    groups = frame["group"].to_numpy()
    threshold = cv.threshold["threshold"]
    predicted = (cv.oof_meta >= threshold).astype(int)
    sources = [_sources(value) for value in frame["sources"]]
    if capped is None:
        by_capped = {
            "domains": None,
            "subgroups": None,
            "note": "not available: the assemble report with the capped domains was not found",
        }
    else:
        capped_set = set(capped)
        keys = [["capped" if group in capped_set else "not_capped"] for group in groups]
        by_capped = {
            "domains": list(capped),
            "subgroups": subgroup_breakdown(y, predicted, keys),
            "note": "domains capped at the per-domain phishing limit in the assemble step; "
            "docs/01-datos.md identifies them as hosting platforms and shorteners. No other "
            "shortener or shared-platform list exists in the code",
        }
    return {
        "scores": "out-of-fold meta-model scores of the grouped measurement",
        "threshold": threshold,
        "rows": len(y),
        "confusion": confusion(y, predicted),
        "rates": rates(confusion(y, predicted)),
        "by_source": subgroup_breakdown(y, predicted, sources),
        "multi_source_rows": int(sum(len(set(row)) > 1 for row in sources)),
        "by_source_note": "a row with several sources counts in each of them",
        "by_synthetic": subgroup_breakdown(
            y, predicted, [[str(bool(value)).lower()] for value in frame["synthetic"]]
        ),
        "by_capped_platform_domain": by_capped,
        "top_domains": {
            "false_positives": top_domains(groups, (y == 0) & (predicted == 1)),
            "false_negatives": top_domains(groups, (y == 1) & (predicted == 0)),
            "note": "registered domains only, never URLs (RNF-05)",
        },
    }


def _describe_space(space: dict) -> dict:
    described = {}
    for key, values in space.items():
        numbers = [value for value in values if value is not None]
        if len(values) > 5 and len(numbers) == len(values):
            described[key] = [min(values), max(values)]
        else:
            described[key] = list(values)
    return described


def _measurement(cv, n_iter: int, folds_label: str, spaces: dict, source: str) -> dict:
    return {
        "folds": folds_label,
        "n_iter": n_iter,
        "spaces": {name: _describe_space(space) for name, space in spaces.items()},
        "spaces_source": source,
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
        "durations": cv.durations,
    }


def _official_reference(repo_dir: Path) -> dict | None:
    path = repo_dir / OFFICIAL_REPORT
    if not path.is_file():
        return None
    official = json.loads(path.read_text(encoding="utf-8"))
    return {
        "report": OFFICIAL_REPORT.as_posix(),
        "stacking_cv_f1": official["stacking"]["cv"]["mean"]["f1"],
        "threshold": official["threshold"]["value"],
        "best_params": {
            name: entry["best_params"] for name, entry in official["base_models"].items()
        },
    }


def diagnose(
    data_dir: Path,
    out_dir: Path,
    features_run: str | None = None,
    n_iter: int = DEFAULT_N_ITER,
    n_iter_wide: int = DEFAULT_N_ITER,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
    threshold_grid: list[float] | None = None,
    now: datetime | None = None,
    repo_dir: Path | None = None,
    command: str | None = None,
) -> dict:
    """Run the three measurements on the training matrix and write ``report.json``."""
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
    X = frame[feature_columns(frame)]
    y = frame["label"].to_numpy().astype(int)
    groups = frame["group"].to_numpy()

    def measure(grouped: bool, spaces: dict, iterations: int):
        strategy = FOLD_GROUPED if grouped else FOLD_RANDOM
        folds = make_folds(y, groups, n_splits=n_splits, seed=seed, strategy=strategy)
        clock = time.perf_counter()
        cv = cross_validate(
            X, y, folds, search_spaces=spaces, n_iter=iterations, seed=seed, n_jobs=n_jobs,
            threshold_grid=grid,
        )
        cv.durations["total_seconds"] = round(time.perf_counter() - clock, 3)
        return cv

    grouped_cv = measure(True, SEARCH_SPACES, n_iter)
    ungrouped_cv = measure(False, SEARCH_SPACES, n_iter)
    wide_cv = measure(True, WIDE_SEARCH_SPACES, n_iter_wide)

    grouped = _measurement(
        grouped_cv, n_iter, "StratifiedGroupKFold by registered domain (as in 1.0.0)",
        SEARCH_SPACES, "P09",
    )
    ungrouped = _measurement(
        ungrouped_cv, n_iter, "StratifiedKFold, shuffled, groups ignored", SEARCH_SPACES, "P09"
    )
    wide = _measurement(
        wide_cv, n_iter_wide, "StratifiedGroupKFold by registered domain (as in 1.0.0)",
        WIDE_SEARCH_SPACES, "own exploratory ranges, wider than P09; not from the corpus",
    )

    def stacking_f1(entry: dict) -> float:
        return entry["cv_mean"]["stacking"]["f1"]

    capped = capped_domains(data_dir, recorded)
    created_at = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    report = {
        "created_at": created_at,
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
        "n_splits": n_splits,
        "threshold_grid": {"min": min(grid), "max": max(grid), "count": len(grid)},
        "measurements": {
            "grouping": {
                "grouped": grouped,
                "ungrouped": ungrouped,
                "stacking_f1_difference_ungrouped_minus_grouped": round(
                    stacking_f1(ungrouped) - stacking_f1(grouped), 6
                ),
                "note": "ungrouped folds let URLs of one registered domain sit in training and "
                "validation at once",
            },
            "ranges": {
                "p09": "same as measurements.grouping.grouped",
                "wide": wide,
                "stacking_f1_difference_wide_minus_p09": round(
                    stacking_f1(wide) - stacking_f1(grouped), 6
                ),
            },
        },
        "official_1_0_0": _official_reference(repo_dir),
        "error_analysis": error_analysis(frame, grouped_cv, capped),
        "base_model_correlation": base_model_correlation(y, grouped_cv.oof_base),
        "parallelism": {"n_jobs": n_jobs, "estimator_threads": 1},
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": _versions(),
    }
    report["durations"] = {
        "grouped_seconds": grouped_cv.durations["total_seconds"],
        "ungrouped_seconds": ungrouped_cv.durations["total_seconds"],
        "wide_seconds": wide_cv.durations["total_seconds"],
        "total_seconds": round(time.perf_counter() - started, 3),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _print_report(report: dict, run_dir: Path) -> None:
    measurements = report["measurements"]
    rows = [
        ("grouped P09", measurements["grouping"]["grouped"]),
        ("ungrouped P09", measurements["grouping"]["ungrouped"]),
        ("grouped wide", measurements["ranges"]["wide"]),
    ]
    for label, entry in rows:
        cells = " ".join(
            f"{name} {entry['cv_mean'][name]['f1']:.4f}" for name in (*BASE_MODEL_NAMES, "stacking")
        )
        print(f"{label:<14} CV F1: {cells} (threshold {entry['threshold']})")
    print(f"confusion {report['error_analysis']['confusion']}")
    print(f"total {report['durations']['total_seconds']} s; report not citable (exploration)")
    print(f"report: {run_dir.as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts") / "diagnostics")
    parser.add_argument("--features-run", help="timestamp of the feature run (default: newest)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument(
        "--n-iter-wide", type=int, default=DEFAULT_N_ITER, help="candidates in the wide ranges"
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--threshold-step", type=float, default=DEFAULT_THRESHOLD_STEP)
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)

    moment = datetime.now(UTC)
    report = diagnose(
        args.data_dir,
        args.out_dir,
        features_run=args.features_run,
        n_iter=args.n_iter,
        n_iter_wide=args.n_iter_wide,
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

"""Source-bias control: can the feature vector recover the scheme, the www. flag or the source?

The control "Sesgo de fuente" of the leakage checklist, applied to the training feature matrix.
With the 24 features and nothing else, a diagnostic probe tries to guess (A) the scheme of the
URL, (B) whether it carried ``www.`` and (C) which source it came from. If it can, a class
model could be learning the source instead of phishing.

Every probe runs inside one class and one shape (with or without a path or query), so neither
the class nor the presence of a path can explain the result. The probe is validated with folds
grouped by registered domain and only its out-of-fold predictions are scored. The design and
the verdict bands were fixed in ``docs/02-caracteristicas.md`` before the first run.

Only the training matrix is read; the test matrix is never opened. The probe is a diagnostic,
not a model of the thesis.
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
import pyarrow as pa
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.clean import CLASS_NAMES
from sereno_model.data.normalize import parse_url
from sereno_model.data.split import TRAIN, group_keys
from sereno_model.features.build import REPORT_NAME as FEATURES_REPORT_NAME
from sereno_model.features.build import REPOSITORY_DIR, git_state, matrix_file, matrix_hash
from sereno_model.features.extract import FEATURE_NAMES, WWW_PREFIX

REPORT_NAME = "source_bias_report.json"
MODULE_COMMAND = "python -m sereno_model.features.source_bias"
RUN_TYPE = (
    "exploration/leakage control on the training split with cross-validation; "
    "not a model result"
)
HYPOTHESIS = (
    "inside one class and one shape, the feature vector cannot recover the scheme, the www. "
    "flag or the source of a URL (out-of-fold ROC AUC below the pass band)"
)

DEFAULT_SEED = 20261111
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 5
N_SPLITS = 5

# Verdict bands, fixed before any run. Do not tune them to a result.
PASS_BELOW = 0.65
FAIL_ABOVE = 0.80
PASS, REVIEW, FAIL, NOT_EVALUABLE = "pass", "review", "fail", "not_evaluable"
SEVERITY = {PASS: 0, REVIEW: 1, FAIL: 2}

MIN_MINORITY_ROWS = 100
MIN_GROUPS_PER_SIDE = 5

SHAPES = {True: "path", False: "bare"}
CONTROLS = {"A": "scheme", "B": "www", "C": "source"}
SOURCE_PREFIX = "source:"
SECURE_SCHEME = "https"


def verdict_for(auc: float | None) -> str:
    """Band of an out-of-fold AUC; ``None`` means the target could not be evaluated."""
    if auc is None:
        return NOT_EVALUABLE
    if auc < PASS_BELOW:
        return PASS
    return FAIL if auc > FAIL_ABOVE else REVIEW


def latest_features_run(data_dir: Path) -> str:
    """Name of the newest folder under ``processed/`` that holds a features report."""
    processed = data_dir / "processed"
    runs = [path.parent.name for path in processed.glob(f"*/{FEATURES_REPORT_NAME}")]
    if not runs:
        raise LookupError(f"no features run found under {processed.as_posix()}")
    return max(runs)


def load_train_matrix(data_dir: Path, run: str) -> tuple[pd.DataFrame, dict]:
    """Read the training matrix of a features run after checking it; never the test matrix.

    The recorded content hash covers the URL, the label and the features. The columns it does
    not cover and that can be derived from the URL (split, scheme, ``www.``, group) are derived
    again and compared. A ``shape`` column is added for the strata.
    """
    run_dir = data_dir / "processed" / run
    recorded = json.loads((run_dir / FEATURES_REPORT_NAME).read_text(encoding="utf-8"))
    if recorded["feature_set"]["names"] != list(FEATURE_NAMES):
        raise ValueError(f"processed/{run}: its feature set is not the one of this code")
    file_name = matrix_file(TRAIN)
    frame = pd.read_parquet(run_dir / file_name)

    expected_hash = recorded["splits"][TRAIN]["content_hash"]["value"]
    found_hash = matrix_hash(frame)
    if found_hash != expected_hash:
        raise ChecksumMismatchError(
            f"processed/{run}/{file_name}: expected content hash {expected_hash}, "
            f"got {found_hash}"
        )

    urls = list(frame["url"])
    parsed = [parse_url(url) for url in urls]
    expected = {
        "split": [TRAIN] * len(frame),
        "audit_scheme": [parts.scheme for parts in parsed],
        "audit_has_www": [parts.host.startswith(WWW_PREFIX) for parts in parsed],
        "group": group_keys(urls),
    }
    for column, values in expected.items():
        altered = int((frame[column].to_numpy() != np.array(values)).sum())
        if altered:
            raise ChecksumMismatchError(
                f"processed/{run}/{file_name}: {altered} rows whose {column} differs from "
                "the value derived from the URL"
            )
    frame["shape"] = [SHAPES[parts.has_path_or_query] for parts in parsed]
    return frame, recorded


def grouped_folds(
    labels: np.ndarray, groups: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Train and held-out row positions of each fold; a group never sits on both sides."""
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC AUC rounded for the report; ``None`` when only one side is present."""
    if len(np.unique(labels)) < 2:
        return None
    return round(float(roc_auc_score(labels, scores)), 6)


def feature_separability(
    features: np.ndarray, labels: np.ndarray, feature_names: tuple[str, ...]
) -> list[dict]:
    """How well each raw feature alone orders the target, best first; no model involved.

    The value is ``max(auc, 1 - auc)``, so 0.5 means no separation and 1.0 a perfect one in
    either direction. Ties keep the order of the feature set.
    """
    ranking = []
    for position, name in enumerate(feature_names):
        auc = roc_auc_score(labels, features[:, position])
        ranking.append({"feature": name, "separability": round(float(max(auc, 1 - auc)), 6)})
    return sorted(ranking, key=lambda entry: -entry["separability"])


def probe_target(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
    n_estimators: int,
    jobs: int = 1,
) -> dict:
    """Probe one binary target: counts, out-of-fold AUC, per-fold AUCs, verdict and ranking.

    A target whose minority side is too small, or that has too few groups on either side, is
    returned as ``not_evaluable`` with the reason and the counts instead of being fitted.
    """
    labels = np.asarray(labels, dtype=bool)
    groups = np.asarray(groups)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    positive_groups = len(set(groups[labels]))
    negative_groups = len(set(groups[~labels]))
    result = {
        "status": "evaluated",
        "reason": None,
        "rows": len(labels),
        "positives": positives,
        "negatives": negatives,
        "positive_groups": positive_groups,
        "negative_groups": negative_groups,
        "oof_auc": None,
        "fold_aucs": None,
        "verdict": NOT_EVALUABLE,
        "feature_separability": None,
    }

    if min(positives, negatives) < MIN_MINORITY_ROWS:
        result["reason"] = (
            f"the minority side has {min(positives, negatives)} rows, "
            f"fewer than {MIN_MINORITY_ROWS} rows"
        )
    elif min(positive_groups, negative_groups) < MIN_GROUPS_PER_SIDE:
        result["reason"] = (
            f"one side has {min(positive_groups, negative_groups)} distinct groups, "
            f"fewer than {MIN_GROUPS_PER_SIDE} distinct groups"
        )
    if result["reason"]:
        result["status"] = NOT_EVALUABLE
        return result

    out_of_fold = np.zeros(len(labels))
    fold_aucs = []
    for train, held_out in grouped_folds(labels, groups, seed):
        probe = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            random_state=seed,
            n_jobs=jobs,
        )
        probe.fit(features[train], labels[train])
        probabilities = probe.predict_proba(features[held_out])
        # A training fold with a single side yields one column; score it as a constant.
        if True in probe.classes_:
            out_of_fold[held_out] = probabilities[:, list(probe.classes_).index(True)]
        fold_aucs.append(_auc(labels[held_out], out_of_fold[held_out]))

    result["oof_auc"] = _auc(labels, out_of_fold)
    result["fold_aucs"] = fold_aucs
    result["verdict"] = verdict_for(result["oof_auc"])
    result["feature_separability"] = feature_separability(features, labels, feature_names)
    return result


def stratum_targets(part: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """Control letter, target name and boolean target of every probe of one stratum."""
    targets = [
        ("A", "scheme", (part["audit_scheme"] == SECURE_SCHEME).to_numpy()),
        ("B", "www", part["audit_has_www"].to_numpy(dtype=bool)),
    ]
    sources = [set(names) for names in part["sources"]]
    for name in sorted(set().union(*sources)):
        held = np.array([name in names for names in sources], dtype=bool)
        targets.append(("C", f"{SOURCE_PREFIX}{name}", held))
    return targets


def probe_strata(frame: pd.DataFrame, seed: int, n_estimators: int, jobs: int) -> list[dict]:
    """Probe every target inside each class and shape, in a fixed order."""
    strata = []
    for label, kind in sorted(CLASS_NAMES.items(), key=lambda item: item[1]):
        for shape in sorted(SHAPES.values()):
            part = frame[(frame["label"] == label) & (frame["shape"] == shape)]
            features = part[list(FEATURE_NAMES)].to_numpy(dtype=float)
            groups = part["group"].to_numpy()
            targets = [
                {
                    "control": control,
                    "target": name,
                    **probe_target(
                        features, labels, groups, FEATURE_NAMES, seed, n_estimators, jobs
                    ),
                }
                for control, name, labels in stratum_targets(part)
            ]
            strata.append(
                {
                    "stratum": f"{kind}/{shape}",
                    "class": kind,
                    "shape": shape,
                    "rows": len(part),
                    "groups": int(part["group"].nunique()),
                    "targets": targets,
                }
            )
    return strata


def control_verdicts(strata: list[dict]) -> dict:
    """Overall verdict of each control: the worst of its targets, ignoring not_evaluable."""
    controls = {}
    for control, name in CONTROLS.items():
        targets = [
            {"stratum": stratum["stratum"], **target}
            for stratum in strata
            for target in stratum["targets"]
            if target["control"] == control
        ]
        evaluated = [target for target in targets if target["status"] == "evaluated"]
        worst = max(evaluated, key=lambda target: target["oof_auc"], default=None)
        verdicts = [target["verdict"] for target in evaluated]
        controls[control] = {
            "name": name,
            "verdict": max(verdicts, key=SEVERITY.get, default=NOT_EVALUABLE),
            "evaluated_targets": len(evaluated),
            "not_evaluable_targets": len(targets) - len(evaluated),
            "highest_oof_auc": worst
            and {key: worst[key] for key in ("stratum", "target", "oof_auc", "verdict")},
        }
    return controls


def run_control(
    data_dir: Path,
    features_run: str | None = None,
    seed: int = DEFAULT_SEED,
    jobs: int = 1,
    n_estimators: int | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    repo_dir: Path | None = None,
    command: str | None = None,
) -> dict:
    """Run the control on a features run and write ``source_bias_report.json`` to a new folder.

    The newest features run is used unless ``features_run`` names one. An existing output
    folder is never overwritten. With ``dry_run`` everything is computed, nothing is written.
    """
    started = time.perf_counter()
    trees = N_ESTIMATORS if n_estimators is None else n_estimators
    moment = now or datetime.now(UTC)
    output_dir = data_dir / "processed" / moment.strftime("%Y%m%dT%H%M%SZ")
    if not dry_run and output_dir.exists():
        raise FileExistsError(f"{output_dir.as_posix()} already exists")

    run = features_run or latest_features_run(data_dir)
    frame, recorded = load_train_matrix(data_dir, run)
    strata = probe_strata(frame, seed, trees, jobs)

    report = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_type": RUN_TYPE,
        "hypothesis": HYPOTHESIS,
        "command": command,
        "dry_run": dry_run,
        "output_dir": None if dry_run else output_dir.relative_to(data_dir).as_posix(),
        "git": git_state(repo_dir or REPOSITORY_DIR),
        "parameters": {
            "seed": seed,
            "jobs": jobs,
            "probe": {
                "model": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": trees,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "random_state": "seed",
                "validation": "sklearn.model_selection.StratifiedGroupKFold(n_splits, "
                "shuffle=True, random_state=seed), grouped by the group column "
                "(registered domain)",
                "n_splits": N_SPLITS,
                "metric": "ROC AUC over all out-of-fold probabilities (oof_auc) and inside "
                "each held-out fold (fold_aucs)",
                "inputs": "the feature columns only, the same probe for every target",
            },
            "feature_separability": "max(auc, 1 - auc) of each raw feature against the "
            "target, no model, listed in descending order",
            "strata": "class (label) x shape; shape is parse_url(url).has_path_or_query",
            "targets": {
                "scheme": "audit_scheme == 'https'",
                "www": "audit_has_www",
                "source:<name>": "the sources of the row contain <name>, one target per "
                "source present in the stratum",
            },
            "evaluability": {
                "min_minority_rows": MIN_MINORITY_ROWS,
                "min_groups_per_side": MIN_GROUPS_PER_SIDE,
            },
            "verdict_bands": {
                "pass": f"oof_auc < {PASS_BELOW:.2f}",
                "review": f"{PASS_BELOW:.2f} <= oof_auc <= {FAIL_ABOVE:.2f}",
                "fail": f"oof_auc > {FAIL_ABOVE:.2f}",
                "pass_below": PASS_BELOW,
                "fail_above": FAIL_ABOVE,
                "control_verdict": "the worst verdict of its targets, ignoring not_evaluable",
            },
        },
        "features_run": {
            "dir": f"processed/{run}",
            "split": TRAIN,
            "file": matrix_file(TRAIN),
            "rows": len(frame),
            "matrix_hash": recorded["splits"][TRAIN]["content_hash"]["value"],
            "partition_hash": recorded["split_run"]["partition_hash"],
            "feature_set_version": recorded["feature_set"]["version"],
            "test_matrix": "never opened",
        },
        "feature_names": list(FEATURE_NAMES),
        "controls": control_verdicts(strata),
        "strata": strata,
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / REPORT_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def _print_report(report: dict, data_dir: Path) -> None:
    run = report["features_run"]
    parameters = report["parameters"]
    print(f"features run {run['dir']}, train matrix sha256 {run['matrix_hash']}")
    print(
        f"seed {parameters['seed']}, jobs {parameters['jobs']}, "
        f"{parameters['probe']['n_estimators']} trees, {parameters['probe']['n_splits']} "
        "folds grouped by registered domain"
    )
    print(
        f"{'stratum':<16} {'target':<26} {'rows':>7} {'positives':>9} {'auc':>6} "
        f"{'verdict':<13} top features"
    )
    for stratum in report["strata"]:
        for target in stratum["targets"]:
            evaluated = target["status"] == "evaluated"
            auc = f"{target['oof_auc']:.3f}" if evaluated else "-"
            top = ", ".join(
                f"{entry['feature']} {entry['separability']:.3f}"
                for entry in (target["feature_separability"] or [])[:3]
            )
            print(
                f"{stratum['stratum']:<16} {target['target']:<26} {target['rows']:>7} "
                f"{target['positives']:>9} {auc:>6} {target['verdict']:<13} "
                f"{top or target['reason']}"
            )
    for letter, control in report["controls"].items():
        print(
            f"control {letter} {control['name']}: {control['verdict']} "
            f"({control['evaluated_targets']} targets evaluated, "
            f"{control['not_evaluable_targets']} not evaluable)"
        )
    print(f"duration {report['duration_seconds']:.1f} s")
    git = report["git"]
    print(f"git commit {git['commit']}, clean {git['clean']}")
    if git["clean"] is not True:
        print("warning: the working tree is not known to be clean; this run cannot be reported")
    if report["dry_run"]:
        print("dry run: nothing was written")
    else:
        print(f"report: {(data_dir / report['output_dir'] / REPORT_NAME).as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run", help="timestamp of the features run (default: newest)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=1, help="parallel jobs of the probe")
    parser.add_argument("--dry-run", action="store_true", help="print a summary, write nothing")
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)

    command = " ".join([MODULE_COMMAND, *arguments])
    report = run_control(
        args.data_dir,
        args.run,
        seed=args.seed,
        jobs=args.jobs,
        dry_run=args.dry_run,
        command=command,
    )
    _print_report(report, args.data_dir)


if __name__ == "__main__":
    main()

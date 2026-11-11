"""Split the assembled dataset into a fixed train/test partition grouped by registered domain.

Rows are sorted by URL and split with ``StratifiedGroupKFold``; one fold is the test set. Every
URL of a registered domain lands on the same side, so the test set only holds unseen domains,
and the class proportions are approximately preserved. The leakage controls that can be run on
the data alone are saved with the report; a failed one stops the run before anything is written.
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import sklearn
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.assemble import (
    DEFAULT_SEED,
    STRATA,
    Stratum,
    dataset_hash,
    form_distance,
    stratum_name,
)
from sereno_model.data.clean import CLASS_NAMES, LEGITIMATE, PHISHING, write_parquet
from sereno_model.data.domains import registered_domain, suffix_list_snapshot_id
from sereno_model.data.normalize import normalize_url, parse_url

DEFAULT_N_SPLITS = 5
# Position, in the order StratifiedGroupKFold yields them, of the fold kept as the test set.
TEST_FOLD = 0
TRAIN, TEST = "train", "test"
SPLITS = (TRAIN, TEST)
LARGEST_GROUPS = 10

SPLIT_SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("label", pa.int8()),
        ("split", pa.string()),
        ("group", pa.string()),
        ("synthetic", pa.bool_()),
    ]
)

NO_LEGITIMATE_DATES = (
    "only phishing URLs reported by PhishTank carry a submission time; legitimate URLs have "
    "no dates, so the two classes cannot be compared in time"
)


class LeakageError(ValueError):
    """A leakage control failed on the partition; nothing was written."""


def _share(part: int, total: int) -> float | None:
    return round(part / total, 6) if total else None


def latest_assemble_run(data_dir: Path) -> str:
    """Name of the newest folder under ``processed/`` that holds an assemble report."""
    processed = data_dir / "processed"
    runs = [path.parent.name for path in processed.glob("*/assemble_report.json")]
    if not runs:
        raise LookupError(f"no assemble run found under {processed.as_posix()}")
    return max(runs)


def load_dataset(data_dir: Path, run: str) -> tuple[pd.DataFrame, str]:
    """Read an assembled dataset, sorted by URL, after checking its recorded content hash."""
    run_dir = data_dir / "processed" / run
    recorded = json.loads((run_dir / "assemble_report.json").read_text(encoding="utf-8"))
    recorded_hash = recorded["content_hash"]["value"]
    frame = pd.read_parquet(run_dir / "dataset.parquet")
    found = dataset_hash(frame)
    if found != recorded_hash:
        raise ChecksumMismatchError(
            f"processed/{run}/dataset.parquet: expected content hash {recorded_hash}, got {found}"
        )
    return frame.sort_values("url", kind="stable").reset_index(drop=True), recorded_hash


def group_keys(urls: list[str]) -> list[str]:
    """Registered domain of the host of each URL."""
    by_host: dict[str, str] = {}
    keys = []
    for url in urls:
        host = parse_url(url).host
        if host not in by_host:
            by_host[host] = registered_domain(host)
        keys.append(by_host[host])
    return keys


def held_out_mask(labels: list[int], groups: list[str], seed: int, n_splits: int) -> list[bool]:
    """True for the rows of the test fold; the rows must already be in their fixed order."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = splitter.split(np.zeros((len(labels), 1)), np.asarray(labels), np.asarray(groups))
    _, test_rows = next(islice(folds, TEST_FOLD, None))
    mask = np.zeros(len(labels), dtype=bool)
    mask[test_rows] = True
    return mask.tolist()


def partition_hash(urls: list[str], splits: list[str]) -> str:
    """SHA-256 over the sorted ``url<TAB>split`` lines, whatever the storage."""
    digest = hashlib.sha256()
    for line in sorted(f"{url}\t{name}\n" for url, name in zip(urls, splits, strict=True)):
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _by_split_and_class(frame: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (name, kind): frame[(frame["split"] == name) & (frame["label"] == label)]
        for name in SPLITS
        for label, kind in CLASS_NAMES.items()
    }


def split_summary(frame: pd.DataFrame) -> dict:
    """Rows, classes, groups and synthetic rows of each split."""
    summary = {}
    for name in SPLITS:
        part = frame[frame["split"] == name]
        phishing = int((part["label"] == PHISHING).sum())
        legitimate = int((part["label"] == LEGITIMATE).sum())
        synthetic = int(part["synthetic"].sum())
        summary[name] = {
            "rows": len(part),
            "share_of_rows": _share(len(part), len(frame)),
            "phishing": phishing,
            "legitimate": legitimate,
            "phishing_share": _share(phishing, len(part)),
            "groups": int(part["group"].nunique()),
            "synthetic": synthetic,
            "synthetic_share": _share(synthetic, len(part)),
            "synthetic_share_of_legitimate": _share(synthetic, legitimate),
        }
    return summary


def group_summary(frame: pd.DataFrame) -> dict:
    """How many groups there are and the largest ones, with their classes and their split."""
    rows = Counter(frame["group"])
    phishing = Counter(frame.loc[frame["label"] == PHISHING, "group"])
    sides: dict[str, set[str]] = {}
    for group, name in zip(frame["group"], frame["split"], strict=True):
        sides.setdefault(group, set()).add(name)
    largest = sorted(rows, key=lambda group: (-rows[group], group))[:LARGEST_GROUPS]
    return {
        "total": len(rows),
        "with_both_classes": sum(0 < phishing[group] < count for group, count in rows.items()),
        "largest": [
            {
                "group": group,
                "rows": rows[group],
                "phishing": phishing[group],
                "legitimate": rows[group] - phishing[group],
                "split": "+".join(sorted(sides[group])),
            }
            for group in largest
        ],
    }


def source_counts(frame: pd.DataFrame) -> dict:
    """URLs of each source per split and class; a URL in two sources counts in both."""
    counts: dict = {name: {} for name in SPLITS}
    for (name, kind), part in _by_split_and_class(frame).items():
        found = Counter(source for sources in part["sources"] for source in sources)
        counts[name][kind] = dict(sorted(found.items()))
    return counts


def form_counts(frame: pd.DataFrame) -> dict[tuple[str, str], dict[Stratum, int]]:
    """URLs in each form stratum, keyed by (split, class)."""
    counts = {}
    for key, part in _by_split_and_class(frame).items():
        columns = (part["has_path_or_query"], part["scheme"], part["has_www"])
        found = Counter(
            (bool(with_path), scheme, bool(www))
            for with_path, scheme, www in zip(*columns, strict=True)
        )
        counts[key] = {stratum: found[stratum] for stratum in STRATA}
    return counts


def form_strata(counts: dict[tuple[str, str], dict[Stratum, int]]) -> dict:
    report: dict = {name: {} for name in SPLITS}
    for (name, kind), found in counts.items():
        report[name][kind] = {stratum_name(stratum): count for stratum, count in found.items()}
    return report


def form_distances(counts: dict[tuple[str, str], dict[Stratum, int]]) -> dict:
    """Total variation distance between the form distributions of the classes, per split."""
    return {
        name: form_distance(counts[name, "phishing"], counts[name, "legitimate"])
        for name in SPLITS
    }


def submission_times(frame: pd.DataFrame) -> dict:
    """Earliest, latest and lower-median submission time of the phishing URLs of each split."""
    ranges = {}
    for name in SPLITS:
        part = frame[(frame["split"] == name) & (frame["label"] == PHISHING)]
        texts = [text for text in part["submission_time"] if isinstance(text, str) and text]
        known = sorted(texts, key=lambda text: datetime.fromisoformat(text).astimezone(UTC))
        ranges[name] = {
            "with_date": len(known),
            "without_date": len(part) - len(known),
            "min": known[0] if known else None,
            "max": known[-1] if known else None,
            "median": known[(len(known) - 1) // 2] if known else None,
        }
    legitimate = frame.loc[frame["label"] == LEGITIMATE, "submission_time"]
    return {
        "note": NO_LEGITIMATE_DATES,
        "median": "lower median: the middle observed value, never an interpolated one",
        "legitimate_with_date": int(legitimate.map(lambda text: isinstance(text, str)).sum()),
        "phishing": ranges,
    }


def _in_both(frame: pd.DataFrame, values: list[str]) -> int:
    """How many distinct ``values`` have rows in more than one split."""
    sides: dict[str, set[str]] = {}
    for value, name in zip(values, frame["split"], strict=True):
        sides.setdefault(value, set()).add(name)
    return sum(len(found) > 1 for found in sides.values())


def leakage_controls(frame: pd.DataFrame) -> list[dict]:
    """One entry per decision gate of the ``control-fuga-datos`` skill, in its order."""
    duplicated = _in_both(frame, [normalize_url(url) for url in frame["url"]])
    shared_groups = _in_both(frame, list(frame["group"]))
    return [
        {
            "control": "duplicates",
            "fails_if": "the same normalized URL appears in train and test",
            "status": "fail" if duplicated else "pass",
            "evidence": {"normalized_urls_in_both_splits": duplicated},
        },
        {
            "control": "domains",
            "fails_if": "the same registered domain appears in train and test",
            "status": "fail" if shared_groups else "pass",
            "evidence": {"groups_in_both_splits": shared_groups},
        },
        {
            "control": "form_bias",
            "fails_if": "the class is separated almost only by path, scheme or length because "
            "the classes come from different sources",
            "status": "reported",
            "evidence": form_distances(form_counts(frame)),
            "note": "total variation distance between the phishing and legitimate form "
            "distributions; the distance with a path is known to stay high and is handled by "
            "the features, see docs/decisiones.md",
        },
        {
            "control": "source_bias",
            "fails_if": "a feature identifies the source instead of the phishing",
            "status": "reported",
            "evidence": source_counts(frame),
            "note": "source counts per split and class; whether a feature identifies the source "
            "can only be checked once the features exist",
        },
        {
            "control": "temporal",
            "fails_if": "the date range is not reported, or the classes come from very "
            "different periods without a justification",
            "status": "reported",
            "evidence": submission_times(frame),
            "note": NO_LEGITIMATE_DATES,
        },
        {
            "control": "importance",
            "fails_if": "a single feature dominates without a reasonable explanation",
            "status": "pending_model",
            "evidence": None,
            "note": "needs a trained model",
        },
        {
            "control": "stability",
            "fails_if": "the test score clearly exceeds the cross-validation score",
            "status": "pending_model",
            "evidence": None,
            "note": "needs a trained model",
        },
    ]


def split(
    data_dir: Path,
    assemble_run: str | None = None,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
    now: datetime | None = None,
) -> dict:
    """Write ``split.parquet`` and ``split_report.json`` to a new ``processed/`` folder.

    The newest assemble run is used unless ``assemble_run`` names one. Its ``dataset.parquet``
    must reproduce the content hash of its report. An existing output folder is never
    overwritten, and nothing is written when a leakage control fails.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    run = assemble_run or latest_assemble_run(data_dir)
    frame, recorded_hash = load_dataset(data_dir, run)
    frame["group"] = group_keys(list(frame["url"]))
    held_out = held_out_mask(list(frame["label"]), list(frame["group"]), seed, n_splits)
    frame["split"] = [TEST if flag else TRAIN for flag in held_out]

    controls = leakage_controls(frame)
    failed = [entry["control"] for entry in controls if entry["status"] == "fail"]
    if failed:
        raise LeakageError(f"leakage controls failed: {', '.join(failed)}")

    moment = now or datetime.now(UTC)
    output_dir = data_dir / "processed" / moment.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_parquet(frame[SPLIT_SCHEMA.names], SPLIT_SCHEMA, output_dir / "split.parquet")

    counts = form_counts(frame)
    report = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": output_dir.relative_to(data_dir).as_posix(),
        "assemble_run": {"dir": f"processed/{run}", "content_hash": recorded_hash},
        "parameters": {
            "seed": seed,
            "n_splits": n_splits,
            "test_fold": TEST_FOLD,
            "splitter": "sklearn.model_selection.StratifiedGroupKFold(n_splits, shuffle=True, "
            "random_state=seed); the fold at position test_fold is the test set",
            "row_order": "sorted by url before splitting",
            "group_key": "registered domain of the host (Public Suffix List, ICANN suffixes "
            "only); a host without one is its own key, without a leading 'www.'",
        },
        "splits": split_summary(frame),
        "groups": group_summary(frame),
        "sources": source_counts(frame),
        "form_strata": form_strata(counts),
        "submission_time": submission_times(frame),
        "leakage_controls": controls,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "tldextract": tldextract.__version__,
            "public_suffix_list_snapshot_sha256": suffix_list_snapshot_id(),
        },
        "partition_hash": {
            "algorithm": "sha256",
            "input": "sorted lines 'url<TAB>split<LF>' in UTF-8",
            "value": partition_hash(list(frame["url"]), list(frame["split"])),
        },
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_report(report: dict, data_dir: Path) -> None:
    parameters = report["parameters"]
    print(
        f"assemble run {report['assemble_run']['dir']}, seed {parameters['seed']}, "
        f"n_splits {parameters['n_splits']}, test fold {parameters['test_fold']}"
    )
    for name, part in report["splits"].items():
        print(
            f"{name:<5} rows {part['rows']:>7} ({_text(part['share_of_rows'])}) "
            f"phishing {part['phishing']:>6} legitimate {part['legitimate']:>6} "
            f"phishing share {_text(part['phishing_share'])} groups {part['groups']:>6} "
            f"synthetic {part['synthetic']:>5} ({_text(part['synthetic_share'])})"
        )
    groups = report["groups"]
    print(f"groups {groups['total']}, with both classes {groups['with_both_classes']}; largest:")
    for entry in groups["largest"]:
        print(
            f"  {entry['group']:<28} rows {entry['rows']:>6} phishing {entry['phishing']:>6} "
            f"legitimate {entry['legitimate']:>6} {entry['split']}"
        )
    for name, dates in report["submission_time"]["phishing"].items():
        print(
            f"phishing dates {name}: {dates['min']} to {dates['max']}, median {dates['median']} "
            f"({dates['with_date']} dated, {dates['without_date']} undated)"
        )
    print(f"note: {report['submission_time']['note']}")
    for entry in report["leakage_controls"]:
        line = f"control {entry['control']:<12} {entry['status']:<14}"
        if entry["control"] in ("duplicates", "domains"):
            line += " " + ", ".join(f"{key}={value}" for key, value in entry["evidence"].items())
        elif entry["control"] == "form_bias":
            line += " " + "; ".join(
                f"{name}: all {_text(found['all_strata'])}, with path "
                f"{_text(found['with_path'])}, without path {_text(found['without_path'])}"
                for name, found in entry["evidence"].items()
            )
        else:
            line += f" {entry['note']}"
        print(line)
    print(f"partition sha256 {report['partition_hash']['value']}")
    print(f"report: {(data_dir / report['output_dir'] / 'split_report.json').as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--assemble-run", help="timestamp of the assemble run (default: newest)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    args = parser.parse_args(argv)

    report = split(args.data_dir, args.assemble_run, args.seed, args.n_splits)
    _print_report(report, args.data_dir)


if __name__ == "__main__":
    main()

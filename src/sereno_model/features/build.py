"""Build the train and test feature matrices from the fixed partition.

Every URL of a split run is measured with ``sereno_model.features.extract`` and written, one
file per split, to a new ``processed/`` folder together with a report. The partition and the
assembled dataset it came from must reproduce their recorded hashes before anything is used.

The label, the split, the registered-domain group, the synthetic flag and the sources travel
next to the 24 features, never inside them, so the later leakage controls can use them. The
scheme and the ``www.`` flag of each URL are kept as ``audit_`` columns for the source-bias
control only: they are not features and must not be given to a model.

This module is training-side: unlike the extractor, it needs pandas and pyarrow.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import tldextract

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.clean import CLASS_NAMES, write_parquet
from sereno_model.data.domains import suffix_list_snapshot_id
from sereno_model.data.normalize import parse_url
from sereno_model.data.split import SPLITS, TRAIN, group_keys, load_dataset, partition_hash
from sereno_model.features.extract import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    FLOAT_FEATURES,
    WWW_PREFIX,
    features_from_parts,
)

METADATA_COLUMNS = ("url", "label", "split", "group", "synthetic", "sources")
# Kept for the source-bias control only; never part of the feature set.
AUDIT_COLUMNS = ("audit_scheme", "audit_has_www")

MATRIX_SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("label", pa.int8()),
        ("split", pa.string()),
        ("group", pa.string()),
        ("synthetic", pa.bool_()),
        ("sources", pa.list_(pa.string())),
        ("audit_scheme", pa.string()),
        ("audit_has_www", pa.bool_()),
        *((name, pa.float64() if name in FLOAT_FEATURES else pa.int32()) for name in FEATURE_NAMES),
    ]
)

REPORT_NAME = "features_report.json"
MODULE_COMMAND = "python -m sereno_model.features.build"
RUN_TYPE = "data preparation: feature matrices; no model is trained, selected or evaluated"
REPOSITORY_DIR = Path(__file__).resolve().parents[3]


def matrix_file(split_name: str) -> str:
    return f"features_{split_name}.parquet"


def git_state(repo_dir: Path) -> dict:
    """Commit and cleanliness of the working tree at ``repo_dir``; ``None`` when unknown."""

    def git(*arguments: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(repo_dir), *arguments], check=True, capture_output=True, text=True
        )
        return done.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        changes = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "clean": None}
    return {"commit": commit, "clean": not changes}


def latest_split_run(data_dir: Path) -> str:
    """Name of the newest folder under ``processed/`` that holds a split report."""
    processed = data_dir / "processed"
    runs = [path.parent.name for path in processed.glob("*/split_report.json")]
    if not runs:
        raise LookupError(f"no split run found under {processed.as_posix()}")
    return max(runs)


def load_partition(data_dir: Path, run: str) -> tuple[pd.DataFrame, dict]:
    """Read a partition with the sources of each URL, after checking both recorded hashes."""
    run_dir = data_dir / "processed" / run
    recorded = json.loads((run_dir / "split_report.json").read_text(encoding="utf-8"))
    recorded_hash = recorded["partition_hash"]["value"]
    frame = pd.read_parquet(run_dir / "split.parquet")
    found = partition_hash(list(frame["url"]), list(frame["split"]))
    if found != recorded_hash:
        raise ChecksumMismatchError(
            f"processed/{run}/split.parquet: expected partition hash {recorded_hash}, got {found}"
        )

    assemble_run = Path(recorded["assemble_run"]["dir"]).name
    dataset, dataset_hash = load_dataset(data_dir, assemble_run)
    if dataset_hash != recorded["assemble_run"]["content_hash"]:
        raise ChecksumMismatchError(
            f"processed/{assemble_run}/dataset.parquet: the split run expects content hash "
            f"{recorded['assemble_run']['content_hash']}, got {dataset_hash}"
        )
    # The partition hash only covers the URL and its split, so the label and the synthetic flag
    # are compared with the verified dataset and the group is derived again from the URL.
    verified = dataset[["url", "sources", "label", "synthetic"]].rename(
        columns={"label": "dataset_label", "synthetic": "dataset_synthetic"}
    )
    frame = frame.merge(verified, on="url", how="left", validate="one_to_one")
    if frame["sources"].isna().any():
        raise ChecksumMismatchError(
            f"processed/{run}/split.parquet holds URLs that are not in processed/{assemble_run}"
        )
    expected = {
        "label": frame["dataset_label"],
        "synthetic": frame["dataset_synthetic"],
        "group": pd.Series(group_keys(list(frame["url"])), index=frame.index),
    }
    for column, values in expected.items():
        altered = int((frame[column] != values).sum())
        if altered:
            raise ChecksumMismatchError(
                f"processed/{run}/split.parquet: {altered} rows whose {column} differs from "
                f"processed/{assemble_run}/dataset.parquet"
            )
    frame = frame.drop(columns=["dataset_label", "dataset_synthetic"])
    frame["sources"] = [list(sources) for sources in frame["sources"]]
    return frame, recorded


def feature_frame(partition: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Add the audit and feature columns; also return the seconds spent parsing and measuring."""
    urls = list(partition["url"])
    started = time.perf_counter()
    parsed = [parse_url(url) for url in urls]
    vectors = [features_from_parts(parts) for parts in parsed]
    seconds = time.perf_counter() - started

    frame = partition[list(METADATA_COLUMNS)].reset_index(drop=True)
    frame["audit_scheme"] = [parts.scheme for parts in parsed]
    frame["audit_has_www"] = [parts.host.startswith(WWW_PREFIX) for parts in parsed]
    features = pd.DataFrame(vectors, columns=list(FEATURE_NAMES))
    kinds = {name: "float64" if name in FLOAT_FEATURES else "int32" for name in FEATURE_NAMES}
    return pd.concat([frame, features.astype(kinds)], axis=1), seconds


def matrix_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over the sorted ``url<TAB>label<TAB>features`` lines, whatever the storage.

    Integer features are written in decimal and float ones with ``repr``, which round-trips.
    """
    columns = [frame["url"], frame["label"]] + [
        frame[name].astype(float if name in FLOAT_FEATURES else int).tolist()
        for name in FEATURE_NAMES
    ]
    digest = hashlib.sha256()
    lines = (
        f"{url}\t{label}\t" + "\t".join(repr(value) for value in values) + "\n"
        for url, label, *values in zip(*columns, strict=True)
    )
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _number(value: object, as_float: bool) -> int | float:
    return round(float(value), 6) if as_float else int(value)


def class_statistics(train: pd.DataFrame) -> dict:
    """Minimum, median, maximum and share of non-zero values of each feature, per class."""
    by_class = {}
    for label, kind in CLASS_NAMES.items():
        part = train[train["label"] == label]
        features = {}
        for name in FEATURE_NAMES:
            values = part[name]
            features[name] = {
                "min": _number(values.min(), name in FLOAT_FEATURES) if len(part) else None,
                "median": _number(values.median(), True) if len(part) else None,
                "max": _number(values.max(), name in FLOAT_FEATURES) if len(part) else None,
                "nonzero_share": _number((values != 0).mean(), True) if len(part) else None,
            }
        by_class[kind] = {"rows": len(part), "features": features}
    return {
        "split": TRAIN,
        "note": "computed on the training split only; the test split is never summarized",
        "median": "interpolated median (pandas default)",
        "by_class": by_class,
    }


def split_summary(frame: pd.DataFrame) -> dict:
    """Rows per class, file name and content hash of the matrix of each split."""
    summary = {}
    for name in SPLITS:
        part = frame[frame["split"] == name]
        summary[name] = {
            "rows": len(part),
            **{kind: int((part["label"] == label).sum()) for label, kind in CLASS_NAMES.items()},
            "file": matrix_file(name),
            "content_hash": {
                "algorithm": "sha256",
                "input": "sorted lines 'url<TAB>label<TAB>features<LF>' in UTF-8, the features "
                "in the order of feature_set.names, floats written with repr",
                "value": matrix_hash(part),
            },
        }
    return summary


def build(
    data_dir: Path,
    split_run: str | None = None,
    now: datetime | None = None,
    dry_run: bool = False,
    repo_dir: Path | None = None,
    command: str | None = None,
) -> dict:
    """Write one feature matrix per split and ``features_report.json`` to a new folder.

    The newest split run is used unless ``split_run`` names one. An existing output folder is
    never overwritten. With ``dry_run`` everything is computed and reported, nothing is written.
    """
    started = time.perf_counter()
    run = split_run or latest_split_run(data_dir)
    partition, recorded = load_partition(data_dir, run)
    frame, seconds = feature_frame(partition)

    moment = now or datetime.now(UTC)
    output_dir = data_dir / "processed" / moment.strftime("%Y%m%dT%H%M%SZ")
    report = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_type": RUN_TYPE,
        "command": command,
        "dry_run": dry_run,
        "output_dir": None if dry_run else output_dir.relative_to(data_dir).as_posix(),
        "git": git_state(repo_dir or REPOSITORY_DIR),
        "split_run": {
            "dir": f"processed/{run}",
            "partition_hash": recorded["partition_hash"]["value"],
            "seed": recorded["parameters"]["seed"],
            # Split runs older than the strategy option were all grouped by registered domain.
            "strategy": recorded["parameters"].get("strategy", "grouped"),
            "splitter": recorded["parameters"]["splitter"],
        },
        "assemble_run": recorded["assemble_run"],
        "feature_set": {
            "version": FEATURE_SET_VERSION,
            "names": list(FEATURE_NAMES),
            "float_features": sorted(FLOAT_FEATURES),
            "definition": "docs/02-caracteristicas.md, section 'Vector base'",
        },
        "columns": {
            "metadata": list(METADATA_COLUMNS),
            "audit": list(AUDIT_COLUMNS),
            "features": list(FEATURE_NAMES),
            "note": "only the feature columns may be given to a model; the audit columns exist "
            "for the source-bias control",
        },
        "splits": split_summary(frame),
        "train_statistics": class_statistics(frame[frame["split"] == TRAIN]),
        "extraction": {
            "rows": len(frame),
            "seconds": round(seconds, 6),
            "rows_per_second": round(len(frame) / seconds, 1),
            "mean_ms_per_url": round(1000 * seconds / len(frame), 6),
            "scope": "parse_url plus features_from_parts, single thread, in memory",
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "numpy": np.__version__,
            "tldextract": tldextract.__version__,
            "public_suffix_list_snapshot_sha256": suffix_list_snapshot_id(),
        },
    }

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=False)
        for name in SPLITS:
            part = frame[frame["split"] == name]
            write_parquet(part, MATRIX_SCHEMA, output_dir / matrix_file(name))
    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    if not dry_run:
        (output_dir / REPORT_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def _print_report(report: dict, data_dir: Path) -> None:
    feature_set = report["feature_set"]
    print(
        f"split run {report['split_run']['dir']}, partition sha256 "
        f"{report['split_run']['partition_hash']}"
    )
    print(f"feature set version {feature_set['version']}, {len(feature_set['names'])} features")
    for name, part in report["splits"].items():
        print(
            f"{name:<5} rows {part['rows']:>7} phishing {part['phishing']:>6} "
            f"legitimate {part['legitimate']:>6} sha256 {part['content_hash']['value']}"
        )
    extraction = report["extraction"]
    print(
        f"extraction: {extraction['rows']} URLs in {extraction['seconds']:.3f} s, "
        f"{extraction['rows_per_second']:.0f} rows per second, "
        f"{extraction['mean_ms_per_url']:.4f} ms per URL"
    )
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
    parser.add_argument("--split-run", help="timestamp of the split run (default: newest)")
    parser.add_argument("--dry-run", action="store_true", help="print a summary, write nothing")
    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)

    command = " ".join([MODULE_COMMAND, *arguments])
    report = build(args.data_dir, args.split_run, dry_run=args.dry_run, command=command)
    _print_report(report, args.data_dir)


if __name__ == "__main__":
    main()

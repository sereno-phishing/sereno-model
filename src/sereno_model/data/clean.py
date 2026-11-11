"""Clean and de-duplicate the raw sources into one labelled URL set with a saved report."""

import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sereno_model.data.acquire import ChecksumMismatchError, load_downloads, sha256_of
from sereno_model.data.normalize import InvalidUrlError, parse_url

PHISHING = 1
LEGITIMATE = 0
CLASS_NAMES = {PHISHING: "phishing", LEGITIMATE: "legitimate"}

# url_phish_2026 publishes numeric labels without a legend. Evidence from the 2026-11-04
# snapshots: 5,405 of its 16,600 label-1 URLs appear verbatim in PhishTank and none of its
# 100,000 label-0 URLs do; label 0 is dominated by .org, .edu and .gov hosts and shares 22 URLs
# with the legitimate class of hannousse_2021, none with its phishing class.
LABEL_MAPPING = {
    "hannousse_2021": {"legitimate": LEGITIMATE, "phishing": PHISHING},
    "url_phish_2026": {"0": LEGITIMATE, "1": PHISHING},
    "phishtank": {"yes": PHISHING},
}
NOT_VERIFIED = "not_verified"
MAJESTIC = "majestic_million"

# Hostname in LDH form (punycode included) with at least two labels.
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DOMAIN = re.compile(rf"(?=.{{1,253}}$)(?:{_LABEL}\.)+{_LABEL}")

# Explicit schemas keep the column types stable even when a column is empty or all null.
URLS_SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("label", pa.int8()),
        ("sources", pa.list_(pa.string())),
        ("has_path_or_query", pa.bool_()),
        ("submission_time", pa.string()),
    ]
)
CONFLICTS_SCHEMA = pa.schema([("url", pa.string()), ("source", pa.string()), ("label", pa.int8())])
DOMAINS_SCHEMA = pa.schema([("domain", pa.string()), ("global_rank", pa.int64())])

Reader = Callable[[Path], pd.DataFrame]


def _read_columns(path: Path, columns: list[str], compression: str | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        compression=compression,
    )


def read_hannousse(path: Path) -> pd.DataFrame:
    frame = _read_columns(path, ["url", "status"])
    return pd.DataFrame({"url": frame["url"], "label": frame["status"], "submission_time": None})


def read_url_phish(path: Path) -> pd.DataFrame:
    frame = _read_columns(path, ["url", "label"])
    return pd.DataFrame({"url": frame["url"], "label": frame["label"], "submission_time": None})


def read_phishtank(path: Path) -> pd.DataFrame:
    """Every row is a phishing report; ``verified`` plays the role of the raw label."""
    frame = _read_columns(path, ["url", "submission_time", "verified"], compression="gzip")
    return pd.DataFrame(
        {
            "url": frame["url"],
            "label": frame["verified"],
            "submission_time": frame["submission_time"],
        }
    )


READERS: dict[str, Reader] = {
    "hannousse_2021": read_hannousse,
    "url_phish_2026": read_url_phish,
    "phishtank": read_phishtank,
}


def latest_record(downloads: list[dict], source: str) -> dict:
    records = [record for record in downloads if record["source"] == source]
    if not records:
        raise LookupError(f"{source}: no download recorded in the manifest")
    return max(records, key=lambda record: record["retrieved_at"])


def verified_path(data_dir: Path, record: dict) -> Path:
    """Return the raw file of a manifest record after checking it still has the recorded hash."""
    path = data_dir / record["path"]
    checksum = sha256_of(path)
    if checksum != record["sha256"]:
        raise ChecksumMismatchError(
            f"{record['source']}: {record['path']} expected {record['sha256']}, got {checksum}"
        )
    return path


def canonical_label(source: str, raw_label: str) -> int | None:
    """Map a raw label to 1 (phishing) or 0 (legitimate); None marks an unverified report."""
    mapping = LABEL_MAPPING[source]
    if raw_label in mapping:
        return mapping[raw_label]
    if source == "phishtank":
        return None
    raise ValueError(f"{source}: unknown label {raw_label!r}")


def collect_source(source: str, frame: pd.DataFrame, entries: dict[str, dict]) -> dict:
    """Normalize one source into ``entries`` (keyed by normalized URL) and return its counts."""
    invalid: Counter[str] = Counter()
    seen: set[str] = set()
    valid_rows = 0
    rows = zip(frame["url"], frame["label"], frame["submission_time"], strict=True)
    for raw_url, raw_label, submission_time in rows:
        label = canonical_label(source, raw_label)
        if label is None:
            invalid[NOT_VERIFIED] += 1
            continue
        try:
            parts = parse_url(raw_url)
        except InvalidUrlError as error:
            invalid[error.reason.value] += 1
            continue
        valid_rows += 1
        url = parts.normalized
        seen.add(url)
        entry = entries.setdefault(
            url,
            {"labels": {}, "has_path_or_query": parts.has_path_or_query, "submission_time": None},
        )
        entry["labels"].setdefault(source, set()).add(label)
        if submission_time and (
            entry["submission_time"] is None or submission_time < entry["submission_time"]
        ):
            entry["submission_time"] = submission_time
    return {
        "rows_read": len(frame),
        "invalid": dict(sorted(invalid.items())),
        "valid_rows": valid_rows,
        "unique_urls": len(seen),
        "duplicates_within_source": valid_rows - len(seen),
    }


def content_hash(urls: list[str], labels: list[int]) -> str:
    """SHA-256 over the sorted ``url<TAB>label`` lines, independent of the storage format."""
    digest = hashlib.sha256()
    for line in sorted(f"{url}\t{label}\n" for url, label in zip(urls, labels, strict=True)):
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _share(part: int, total: int) -> float | None:
    return round(part / total, 6) if total else None


def summarize(entries: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split the collected URLs into the kept set and the label conflicts, with their counts."""
    kept_rows, conflict_rows = [], []
    multi_source = merged_rows = 0
    conflicts_by_sources: Counter[str] = Counter()
    overlap: dict[str, Counter] = {}
    for url in sorted(entries):
        entry = entries[url]
        sources = sorted(entry["labels"])
        labels = set().union(*entry["labels"].values())
        agreed = len(labels) == 1
        for pair in combinations(sources, 2):
            outcome = "same_label" if agreed else "conflicting"
            overlap.setdefault("+".join(pair), Counter())[outcome] += 1
        if not agreed:
            conflicts_by_sources["+".join(sources)] += 1
            conflict_rows.extend(
                {"url": url, "source": source, "label": label}
                for source in sources
                for label in sorted(entry["labels"][source])
            )
            continue
        if len(sources) > 1:
            multi_source += 1
            merged_rows += len(sources) - 1
        kept_rows.append(
            {
                "url": url,
                "label": labels.pop(),
                "sources": sources,
                "has_path_or_query": entry["has_path_or_query"],
                "submission_time": entry["submission_time"],
            }
        )

    urls = pd.DataFrame(kept_rows, columns=URLS_SCHEMA.names).astype(
        {"label": "int8", "has_path_or_query": "bool"}
    )
    conflicts = pd.DataFrame(conflict_rows, columns=CONFLICTS_SCHEMA.names)

    by_class, share_by_class, by_source, share_by_source = {}, {}, {}, {}
    with_path = urls["has_path_or_query"]
    in_source = {
        source: urls["sources"].map(lambda found, name=source: name in found).astype("bool")
        for source in READERS
    }
    for label, name in CLASS_NAMES.items():
        of_class = urls["label"] == label
        by_class[name] = int(of_class.sum())
        share_by_class[name] = _share(int((of_class & with_path).sum()), by_class[name])
        for source, mask in in_source.items():
            selected = of_class & mask
            count = int(selected.sum())
            by_source.setdefault(source, {})[name] = count
            share_by_source.setdefault(source, {})[name] = _share(
                int((selected & with_path).sum()), count
            )

    stats = {
        "cross_source": {
            "urls_in_multiple_sources": multi_source,
            "duplicate_rows_merged": merged_rows,
            "overlap_by_source_pair": {
                pair: {"same_label": counts["same_label"], "conflicting": counts["conflicting"]}
                for pair, counts in sorted(overlap.items())
            },
        },
        "conflicts": {
            "urls": sum(conflicts_by_sources.values()),
            "by_sources": dict(sorted(conflicts_by_sources.items())),
        },
        "kept": {"total": len(urls), "by_class": by_class, "by_source": by_source},
        "path_or_query_share": {"by_class": share_by_class, "by_source": share_by_source},
        "content_hash": {
            "algorithm": "sha256",
            "input": "sorted lines 'url<TAB>label<LF>' in UTF-8",
            "value": content_hash(list(urls["url"]), list(urls["label"])),
        },
    }
    return urls, conflicts, stats


def clean_majestic(path: Path) -> tuple[pd.DataFrame, dict]:
    """Validate, lowercase and de-duplicate the ranked domains. They are not turned into URLs."""
    frame = _read_columns(path, ["GlobalRank", "Domain"])
    domains = frame["Domain"].str.strip().str.lower()
    empty = domains == ""
    well_formed = domains.str.fullmatch(_DOMAIN.pattern)
    ranks = frame.loc[well_formed, "GlobalRank"].astype("int64")
    valid = pd.DataFrame({"domain": domains[well_formed], "global_rank": ranks}).sort_values(
        ["global_rank", "domain"], kind="stable"
    )
    kept = valid.drop_duplicates("domain", keep="first").reset_index(drop=True)
    invalid = {"empty": int(empty.sum()), "invalid_domain": int((~well_formed & ~empty).sum())}
    stats = {
        "rows_read": len(frame),
        "invalid": {reason: count for reason, count in invalid.items() if count},
        "duplicates": len(valid) - len(kept),
        "kept": len(kept),
    }
    return kept, stats


def write_parquet(frame: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), path)


def clean(data_dir: Path, now: datetime | None = None) -> dict:
    """Write the cleaned outputs to ``data_dir/processed/<UTC timestamp>/`` and return the report.

    Every raw file is checked against the hash recorded in the manifest before it is read. The
    newest snapshot of each source is used. An existing output folder is never overwritten.
    """
    downloads = load_downloads(data_dir / "manifest.json")
    records = {source: latest_record(downloads, source) for source in (*READERS, MAJESTIC)}
    paths = {source: verified_path(data_dir, record) for source, record in records.items()}

    entries: dict[str, dict] = {}
    source_stats = {
        source: collect_source(source, reader(paths[source]), entries)
        for source, reader in READERS.items()
    }
    urls, conflicts, stats = summarize(entries)
    domains, majestic_stats = clean_majestic(paths[MAJESTIC])

    moment = now or datetime.now(UTC)
    output_dir = data_dir / "processed" / moment.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_parquet(urls, URLS_SCHEMA, output_dir / "urls.parquet")
    write_parquet(conflicts, CONFLICTS_SCHEMA, output_dir / "conflicts.parquet")
    write_parquet(domains, DOMAINS_SCHEMA, output_dir / "majestic_domains.parquet")

    report = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": output_dir.relative_to(data_dir).as_posix(),
        "inputs": {
            source: {key: record.get(key) for key in ("path", "sha256", "bytes", "retrieved_at")}
            for source, record in records.items()
        },
        "label_mapping": {
            source: {raw: label for raw, label in mapping.items()}
            for source, mapping in LABEL_MAPPING.items()
        },
        "sources": source_stats,
        **stats,
        "majestic_domains": majestic_stats,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    (output_dir / "clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    report = clean(args.data_dir)
    for source, stats in report["sources"].items():
        print(
            f"{source}: read {stats['rows_read']}, invalid {sum(stats['invalid'].values())}, "
            f"duplicates {stats['duplicates_within_source']}"
        )
    kept = report["kept"]
    print(
        f"kept {kept['total']} URLs: {kept['by_class']['phishing']} phishing, "
        f"{kept['by_class']['legitimate']} legitimate; "
        f"merged across sources {report['cross_source']['duplicate_rows_merged']}, "
        f"label conflicts {report['conflicts']['urls']}"
    )
    print(f"majestic domains kept: {report['majestic_domains']['kept']}")
    print(f"content sha256 {report['content_hash']['value']}")
    print(f"report: {(args.data_dir / report['output_dir'] / 'clean_report.json').as_posix()}")


if __name__ == "__main__":
    main()

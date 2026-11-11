"""Assemble the modelling dataset: every phishing URL plus a form-matched legitimate sample.

The form stratum of a URL is the triple (has path or query, scheme, host starts with ``www.``).
The legitimate class is sampled so that each stratum holds ``legit_ratio`` legitimate URLs per
phishing URL. Strata without a path that lack real URLs are completed with URLs built from
Majestic Million domains; strata with a path borrow from the surplus of the other strata with a
path. Whatever cannot be met is reported as a shortfall instead of being hidden.
"""

import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa

from sereno_model.data.acquire import ChecksumMismatchError, sha256_of
from sereno_model.data.clean import LEGITIMATE, MAJESTIC, PHISHING, content_hash, write_parquet
from sereno_model.data.normalize import InvalidUrlError, normalize_url, parse_url

DEFAULT_SEED = 20261111
DEFAULT_LEGIT_RATIO = 0.59
DEFAULT_MAJESTIC_TOP_N = 100_000
DEFAULT_MAX_MAJESTIC_SHARE = 0.25
DEFAULT_RATIO_NOTE = (
    "legit_ratio was not given: the default 0.59 was chosen by the author on 2026-11-11 "
    "from the dry run; see docs/decisiones.md"
)

# (has_path_or_query, scheme, has_www)
Stratum = tuple[bool, str, bool]
STRATA: list[Stratum] = [
    (with_path, scheme, www)
    for with_path in (False, True)
    for scheme in ("http", "https")
    for www in (False, True)
]
WITHOUT_PATH = [stratum for stratum in STRATA if not stratum[0]]
WITH_PATH = [stratum for stratum in STRATA if stratum[0]]
GROUPS = {"all_strata": STRATA, "with_path": WITH_PATH, "without_path": WITHOUT_PATH}

DATASET_SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("label", pa.int8()),
        ("sources", pa.list_(pa.string())),
        ("has_path_or_query", pa.bool_()),
        ("scheme", pa.string()),
        ("has_www", pa.bool_()),
        ("synthetic", pa.bool_()),
        ("submission_time", pa.string()),
    ]
)


@dataclass(frozen=True)
class Prepared:
    """A cleaning run indexed by form stratum, ready to be sampled with any ratio."""

    clean_run: str
    clean_hash: str
    majestic_sha256: str
    urls: pd.DataFrame
    phishing: dict[Stratum, int]
    phishing_trailing_slash: dict[Stratum, int]
    legitimate: dict[Stratum, list[str]]
    candidates: list[str]
    candidate_stats: dict


@dataclass(frozen=True)
class Selection:
    """Which legitimate URLs enter the dataset, with the per-stratum accounting."""

    real_urls: list[str]
    synthetic: list[tuple[str, Stratum]]
    rows: list[dict]
    synthetic_stats: dict


def stratum_name(stratum: Stratum) -> str:
    with_path, scheme, www = stratum
    return f"{'path' if with_path else 'no_path'}/{scheme}/{'www' if www else 'no_www'}"


def quota(legit_ratio: float, phishing_count: int) -> int:
    """Legitimate URLs wanted in a stratum: ``legit_ratio * phishing_count`` rounded half up."""
    exact = Decimal(str(legit_ratio)) * phishing_count
    return int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def allocate(total: int, wanted: dict[Stratum, int]) -> dict[Stratum, int]:
    """Split ``total`` units proportionally to ``wanted`` (largest remainder), never above it."""
    asked = sum(wanted.values())
    if total >= asked:
        return dict(wanted)
    shares = {key: divmod(total * count, asked) for key, count in wanted.items()}
    granted = {key: whole for key, (whole, _) in shares.items()}
    by_remainder = sorted(wanted, key=lambda key: -shares[key][1])
    for key in by_remainder[: total - sum(granted.values())]:
        granted[key] += 1
    return granted


def synthetic_limit(real_count: int, max_share: float) -> int | None:
    """Most synthetic rows that keep their share of the legitimate class within ``max_share``."""
    if max_share >= 1:
        return None
    limit = int(max_share * real_count / (1 - max_share))
    while limit > 0 and limit / (real_count + limit) > max_share:
        limit -= 1
    while (limit + 1) / (real_count + limit + 1) <= max_share:
        limit += 1
    return limit


def distance(first: dict[Stratum, int], second: dict[Stratum, int], strata: list) -> float | None:
    """Total variation distance between two form distributions restricted to ``strata``."""
    first_total = sum(first[stratum] for stratum in strata)
    second_total = sum(second[stratum] for stratum in strata)
    if not first_total or not second_total:
        return None
    gaps = (abs(first[s] / first_total - second[s] / second_total) for s in strata)
    return round(sum(gaps) / 2, 6)


def form_distance(phishing: dict[Stratum, int], legitimate: dict[Stratum, int]) -> dict:
    return {name: distance(phishing, legitimate, strata) for name, strata in GROUPS.items()}


def _share(part: int, total: int) -> float | None:
    return round(part / total, 6) if total else None


def _rng(seed: int, *purpose: object) -> random.Random:
    """An independent, reproducible stream per purpose, so one draw never shifts another."""
    return random.Random("/".join(str(part) for part in (seed, *purpose)))


def latest_clean_run(data_dir: Path) -> str:
    """Name of the newest folder under ``processed/`` that holds a cleaning report."""
    processed = data_dir / "processed"
    runs = [path.parent.name for path in processed.glob("*/clean_report.json")]
    if not runs:
        raise LookupError(f"no cleaning run found under {processed.as_posix()}")
    return max(runs)


def candidate_domains(domains: pd.DataFrame, hosts: set[str], top_n: int) -> tuple[list, dict]:
    """Domains among the top ``top_n`` by rank that can become new legitimate URLs."""
    ranked = domains.sort_values(["global_rank", "domain"], kind="stable")
    top = list(ranked["domain"].head(top_n))
    stats = {
        "top_n_domains": len(top),
        "excluded_existing_host": 0,
        "excluded_www_prefix": 0,
        "excluded_not_normalized": 0,
    }
    kept = []
    for domain in top:
        if domain.removeprefix("www.") in hosts:
            stats["excluded_existing_host"] += 1
        elif domain.startswith("www."):
            # It would land in the wrong stratum or become ``www.www.``.
            stats["excluded_www_prefix"] += 1
        elif not _is_normalized(f"http://{domain}"):
            stats["excluded_not_normalized"] += 1
        else:
            kept.append(domain)
    stats["available"] = len(kept)
    return kept, stats


def _is_normalized(url: str) -> bool:
    try:
        return normalize_url(url) == url
    except InvalidUrlError:
        return False


def prepare(data_dir: Path, clean_run: str | None, majestic_top_n: int) -> Prepared:
    """Load a cleaning run, check its content hash and index its URLs by form stratum."""
    run = clean_run or latest_clean_run(data_dir)
    run_dir = data_dir / "processed" / run
    recorded = json.loads((run_dir / "clean_report.json").read_text(encoding="utf-8"))
    recorded_hash = recorded["content_hash"]["value"]
    urls = pd.read_parquet(run_dir / "urls.parquet")
    found = content_hash(list(urls["url"]), list(urls["label"]))
    if found != recorded_hash:
        raise ChecksumMismatchError(
            f"processed/{run}/urls.parquet: expected content hash {recorded_hash}, got {found}"
        )

    parts = [parse_url(url) for url in urls["url"]]
    urls = urls.assign(
        scheme=[part.scheme for part in parts],
        has_www=[part.host.startswith("www.") for part in parts],
        synthetic=False,
    )
    phishing = dict.fromkeys(STRATA, 0)
    trailing_slash = dict.fromkeys(STRATA, 0)
    legitimate: dict[Stratum, list[str]] = {stratum: [] for stratum in STRATA}
    columns = (urls["url"], urls["label"], urls["has_path_or_query"], parts)
    for url, label, with_path, part in zip(*columns, strict=True):
        stratum = (bool(with_path), part.scheme, part.host.startswith("www."))
        if label == PHISHING:
            phishing[stratum] += 1
            trailing_slash[stratum] += part.path == "/"
        else:
            legitimate[stratum].append(url)

    hosts = {part.host.removeprefix("www.") for part in parts}
    domains_path = run_dir / "majestic_domains.parquet"
    candidates, stats = candidate_domains(pd.read_parquet(domains_path), hosts, majestic_top_n)
    return Prepared(
        clean_run=run,
        clean_hash=recorded_hash,
        majestic_sha256=sha256_of(domains_path),
        urls=urls,
        phishing=phishing,
        phishing_trailing_slash=trailing_slash,
        legitimate={stratum: sorted(found) for stratum, found in legitimate.items()},
        candidates=candidates,
        candidate_stats=stats,
    )


def build_synthetic(stratum: Stratum, domains: list[str], with_slash: int) -> list[str]:
    """``scheme://[www.]domain[/]`` for each domain; the first ``with_slash`` end in a slash."""
    _, scheme, www = stratum
    prefix = f"{scheme}://{'www.' if www else ''}"
    urls = [
        f"{prefix}{domain}{'/' if index < with_slash else ''}"
        for index, domain in enumerate(domains)
    ]
    for url in urls:
        if not _is_normalized(url):
            raise ValueError(f"synthetic URL is not in normalized form: {url!r}")
    return urls


def select(
    prepared: Prepared, legit_ratio: float, seed: int, max_majestic_share: float
) -> Selection:
    """Choose the legitimate URLs: real ones first, then borrowed or synthetic ones."""
    quotas = {stratum: quota(legit_ratio, prepared.phishing[stratum]) for stratum in STRATA}
    taken: dict[Stratum, list[str]] = {}
    surplus: list[tuple[str, Stratum]] = []
    for stratum in STRATA:
        available = prepared.legitimate[stratum]
        count = min(len(available), quotas[stratum])
        chosen = set(_rng(seed, "real", stratum_name(stratum)).sample(available, count))
        taken[stratum] = [url for url in available if url in chosen]
        if stratum[0]:
            surplus.extend((url, stratum) for url in available if url not in chosen)
    deficits = {stratum: quotas[stratum] - len(taken[stratum]) for stratum in STRATA}

    # Strata with a path: cover the deficit with the surplus of the other strata with a path.
    with_path_deficits = {stratum: deficits[stratum] for stratum in WITH_PATH}
    wanted = sum(with_path_deficits.values())
    borrowed = _rng(seed, "borrow").sample(surplus, min(wanted, len(surplus)))
    lent = Counter(stratum for _, stratum in borrowed)
    covered = allocate(len(borrowed), with_path_deficits)

    # Strata without a path: cover the deficit with Majestic domains, within the cap.
    real_count = sum(len(urls) for urls in taken.values()) + len(borrowed)
    needs = {stratum: deficits[stratum] for stratum in WITHOUT_PATH}
    requested = sum(needs.values())
    limit = synthetic_limit(real_count, max_majestic_share)
    within_cap = requested if limit is None else min(requested, limit)
    granted = allocate(min(within_cap, len(prepared.candidates)), needs)
    shuffled = list(prepared.candidates)
    _rng(seed, "majestic").shuffle(shuffled)
    synthetic: list[tuple[str, Stratum]] = []
    used = 0
    for stratum in WITHOUT_PATH:
        count, phishing = granted[stratum], prepared.phishing[stratum]
        if not count:
            continue
        # Half-up rounding of count * (phishing share with a trailing slash), in integers.
        with_slash = prepared.phishing_trailing_slash[stratum]
        slashes = (2 * count * with_slash + phishing) // (2 * phishing)
        domains = shuffled[used : used + count]
        used += count
        synthetic.extend((url, stratum) for url in build_synthetic(stratum, domains, slashes))

    rows = []
    for stratum in STRATA:
        added = granted.get(stratum, 0)
        borrowed_here = covered.get(stratum, 0)
        rows.append(
            {
                "stratum": stratum_name(stratum),
                "has_path_or_query": stratum[0],
                "scheme": stratum[1],
                "has_www": stratum[2],
                "phishing": prepared.phishing[stratum],
                "quota": quotas[stratum],
                "legitimate_available": len(prepared.legitimate[stratum]),
                "real_taken": len(taken[stratum]),
                "synthetic_added": added,
                "borrowed_from_other_strata": borrowed_here,
                "lent_to_other_strata": lent[stratum],
                "shortfall": deficits[stratum] - added - borrowed_here,
                "legitimate_final": len(taken[stratum]) + lent[stratum] + added,
            }
        )
    legitimate_total = real_count + len(synthetic)
    synthetic_stats = {
        "share_of_legitimate": _share(len(synthetic), legitimate_total),
        "requested": requested,
        "added": len(synthetic),
        "cap_limit": limit,
        "cap_binding": limit is not None and requested > limit,
        "candidates_exhausted": within_cap > len(prepared.candidates),
    }
    real_urls = [url for urls in taken.values() for url in urls] + [url for url, _ in borrowed]
    return Selection(real_urls, synthetic, rows, synthetic_stats)


def build_dataset(prepared: Prepared, selection: Selection) -> pd.DataFrame:
    """All phishing rows, the selected real legitimate rows and the synthetic ones, by URL."""
    urls = prepared.urls
    real = urls[(urls["label"] == PHISHING) | urls["url"].isin(selection.real_urls)]
    synthetic = pd.DataFrame(
        {
            "url": [url for url, _ in selection.synthetic],
            "label": LEGITIMATE,
            "sources": [[MAJESTIC] for _ in selection.synthetic],
            "has_path_or_query": False,
            "scheme": [stratum[1] for _, stratum in selection.synthetic],
            "has_www": [stratum[2] for _, stratum in selection.synthetic],
            "synthetic": True,
            "submission_time": None,
        },
        columns=DATASET_SCHEMA.names,
    )
    frame = pd.concat([real[DATASET_SCHEMA.names].astype(object), synthetic.astype(object)])
    frame = frame.sort_values("url", kind="stable").reset_index(drop=True)
    if not frame["url"].is_unique:
        duplicated = sorted(frame.loc[frame["url"].duplicated(), "url"])[:5]
        raise ValueError(f"duplicate URLs in the assembled dataset: {duplicated}")
    frame["sources"] = frame["sources"].map(list)
    times = frame["submission_time"]
    frame["submission_time"] = times.where(times.notna(), None)
    return frame.astype(
        {"label": "int8", "has_path_or_query": "bool", "has_www": "bool", "synthetic": "bool"}
    )


def dataset_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over the sorted ``url<TAB>label<TAB>synthetic`` lines, whatever the storage."""
    digest = hashlib.sha256()
    columns = (frame["url"], frame["label"], frame["synthetic"])
    lines = (f"{url}\t{label}\t{int(flag)}\n" for url, label, flag in zip(*columns, strict=True))
    for line in sorted(lines):
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def summarize(prepared: Prepared, selection: Selection) -> dict:
    """Totals, shares and form distances of a selection; nothing here depends on the clock."""
    rows = selection.rows
    phishing = prepared.phishing
    before = {stratum: len(prepared.legitimate[stratum]) for stratum in STRATA}
    after = {stratum: row["legitimate_final"] for stratum, row in zip(STRATA, rows, strict=True)}
    phishing_total, legitimate_total = sum(phishing.values()), sum(after.values())
    synthetic_total = selection.synthetic_stats["added"]

    def with_path_share(counts: dict[Stratum, int]) -> float | None:
        return _share(sum(counts[stratum] for stratum in WITH_PATH), sum(counts.values()))

    return {
        "strata": rows,
        "totals": {
            "rows": phishing_total + legitimate_total,
            "phishing": phishing_total,
            "legitimate": legitimate_total,
            "legitimate_quota": sum(row["quota"] for row in rows),
            "legitimate_real": legitimate_total - synthetic_total,
            "legitimate_synthetic": synthetic_total,
            "shortfall": sum(row["shortfall"] for row in rows),
        },
        "synthetic": selection.synthetic_stats,
        "majestic_candidates": {**prepared.candidate_stats, "used": synthetic_total},
        "with_path_share": {
            "phishing": with_path_share(phishing),
            "legitimate_before": with_path_share(before),
            "legitimate_after": with_path_share(after),
        },
        "form_distance": {
            "metric": "total variation distance between the phishing and legitimate form "
            "distributions; with_path and without_path are conditional on that group",
            "before": form_distance(phishing, before),
            "after": form_distance(phishing, after),
        },
    }


def _check_parameters(legit_ratio: float, majestic_top_n: int, max_majestic_share: float) -> None:
    if not legit_ratio > 0:
        raise ValueError(f"legit_ratio must be positive, got {legit_ratio}")
    if majestic_top_n < 0:
        raise ValueError(f"majestic_top_n must not be negative, got {majestic_top_n}")
    if not 0 <= max_majestic_share <= 1:
        raise ValueError(f"max_majestic_share must be within [0, 1], got {max_majestic_share}")


def assemble(
    data_dir: Path,
    clean_run: str | None = None,
    legit_ratio: float | None = None,
    seed: int = DEFAULT_SEED,
    majestic_top_n: int = DEFAULT_MAJESTIC_TOP_N,
    max_majestic_share: float = DEFAULT_MAX_MAJESTIC_SHARE,
    now: datetime | None = None,
) -> dict:
    """Write ``dataset.parquet`` and ``assemble_report.json`` to a new ``processed/`` folder.

    The newest cleaning run is used unless ``clean_run`` names one. Its ``urls.parquet`` must
    reproduce the content hash of its report. An existing output folder is never overwritten.
    """
    ratio = DEFAULT_LEGIT_RATIO if legit_ratio is None else legit_ratio
    _check_parameters(ratio, majestic_top_n, max_majestic_share)
    prepared = prepare(data_dir, clean_run, majestic_top_n)
    selection = select(prepared, ratio, seed, max_majestic_share)
    frame = build_dataset(prepared, selection)

    moment = now or datetime.now(UTC)
    output_dir = data_dir / "processed" / moment.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_parquet(frame, DATASET_SCHEMA, output_dir / "dataset.parquet")

    report = {
        "created_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": output_dir.relative_to(data_dir).as_posix(),
        "clean_run": {
            "dir": f"processed/{prepared.clean_run}",
            "content_hash": prepared.clean_hash,
            "majestic_domains_sha256": prepared.majestic_sha256,
        },
        "parameters": {
            "legit_ratio": ratio,
            "legit_ratio_source": "default" if legit_ratio is None else "explicit",
            "legit_ratio_note": DEFAULT_RATIO_NOTE if legit_ratio is None else None,
            "seed": seed,
            "majestic_top_n": majestic_top_n,
            "max_majestic_share": max_majestic_share,
        },
        **summarize(prepared, selection),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "content_hash": {
            "algorithm": "sha256",
            "input": "sorted lines 'url<TAB>label<TAB>synthetic(0|1)<LF>' in UTF-8",
            "value": dataset_hash(frame),
        },
    }
    (output_dir / "assemble_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def dry_run(
    data_dir: Path,
    ratios: list[float],
    clean_run: str | None = None,
    seed: int = DEFAULT_SEED,
    majestic_top_n: int = DEFAULT_MAJESTIC_TOP_N,
    max_majestic_share: float = DEFAULT_MAX_MAJESTIC_SHARE,
) -> list[dict]:
    """Summarize what each ratio would produce, without writing anything."""
    for ratio in ratios:
        _check_parameters(ratio, majestic_top_n, max_majestic_share)
    prepared = prepare(data_dir, clean_run, majestic_top_n)
    summaries = []
    for ratio in ratios:
        summary = summarize(prepared, select(prepared, ratio, seed, max_majestic_share))
        summaries.append(
            {
                "legit_ratio": ratio,
                "phishing": summary["totals"]["phishing"],
                "legitimate": summary["totals"]["legitimate"],
                "synthetic": summary["totals"]["legitimate_synthetic"],
                "synthetic_share": summary["synthetic"]["share_of_legitimate"],
                "cap_binding": summary["synthetic"]["cap_binding"],
                "candidates_exhausted": summary["synthetic"]["candidates_exhausted"],
                "borrowed": sum(row["borrowed_from_other_strata"] for row in summary["strata"]),
                "shortfall": summary["totals"]["shortfall"],
                "shortfall_by_stratum": {
                    row["stratum"]: row["shortfall"]
                    for row in summary["strata"]
                    if row["shortfall"]
                },
                "form_distance": summary["form_distance"]["after"],
                "form_distance_before": summary["form_distance"]["before"],
            }
        )
    return summaries


def _text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_dry_run(summaries: list[dict]) -> None:
    before = summaries[0]["form_distance_before"]
    print(
        f"phishing {summaries[0]['phishing']}; form distance before sampling: "
        f"all {_text(before['all_strata'])}, with path {_text(before['with_path'])}, "
        f"without path {_text(before['without_path'])}"
    )
    header = ("ratio", "legitimate", "synthetic", "syn_share", "borrowed", "shortfall")
    print(" ".join(f"{name:>10}" for name in (*header, "tvd_all", "tvd_path", "tvd_nopath")))
    for row in summaries:
        cells = [
            f"{row['legit_ratio']:g}",
            str(row["legitimate"]),
            str(row["synthetic"]),
            _text(row["synthetic_share"]),
            str(row["borrowed"]),
            str(row["shortfall"]),
            *(_text(row["form_distance"][group]) for group in GROUPS),
        ]
        print(" ".join(f"{cell:>10}" for cell in cells))
        notes = [f"shortfall {name}={count}" for name, count in row["shortfall_by_stratum"].items()]
        notes += ["synthetic cap binds"] if row["cap_binding"] else []
        notes += ["Majestic candidates exhausted"] if row["candidates_exhausted"] else []
        if notes:
            print(f"{'':>10} {'; '.join(notes)}")


def _print_report(report: dict, data_dir: Path) -> None:
    totals, synthetic = report["totals"], report["synthetic"]
    parameters = report["parameters"]
    print(f"cleaning run {report['clean_run']['dir']}, legit_ratio {parameters['legit_ratio']:g}")
    if parameters["legit_ratio_note"]:
        print(f"note: {parameters['legit_ratio_note']}")
    for row in report["strata"]:
        print(
            f"{row['stratum']:<22} phishing {row['phishing']:>6} quota {row['quota']:>6} "
            f"available {row['legitimate_available']:>6} real {row['real_taken']:>6} "
            f"synthetic {row['synthetic_added']:>6} "
            f"borrowed {row['borrowed_from_other_strata']:>6} "
            f"lent {row['lent_to_other_strata']:>6} shortfall {row['shortfall']:>6}"
        )
    print(
        f"dataset {totals['rows']} URLs: {totals['phishing']} phishing, "
        f"{totals['legitimate']} legitimate ({totals['legitimate_synthetic']} synthetic, "
        f"share {_text(synthetic['share_of_legitimate'])}); shortfall {totals['shortfall']}"
    )
    if synthetic["cap_binding"]:
        print(
            f"synthetic cap binds: {synthetic['requested']} requested, "
            f"limit {synthetic['cap_limit']}"
        )
    if synthetic["candidates_exhausted"]:
        print("Majestic candidates exhausted")
    for moment in ("before", "after"):
        distances = report["form_distance"][moment]
        print(
            f"form distance {moment}: all {_text(distances['all_strata'])}, "
            f"with path {_text(distances['with_path'])}, "
            f"without path {_text(distances['without_path'])}"
        )
    print(f"content sha256 {report['content_hash']['value']}")
    print(f"report: {(data_dir / report['output_dir'] / 'assemble_report.json').as_posix()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--clean-run", help="timestamp of the cleaning run (default: the newest)")
    parser.add_argument("--legit-ratio", type=float, help="legitimate URLs per phishing URL")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--majestic-top-n", type=int, default=DEFAULT_MAJESTIC_TOP_N)
    parser.add_argument("--max-majestic-share", type=float, default=DEFAULT_MAX_MAJESTIC_SHARE)
    parser.add_argument("--dry-run", action="store_true", help="print a summary, write nothing")
    parser.add_argument("--ratios", help="comma-separated ratios for --dry-run")
    args = parser.parse_args(argv)

    shared = {
        "clean_run": args.clean_run,
        "seed": args.seed,
        "majestic_top_n": args.majestic_top_n,
        "max_majestic_share": args.max_majestic_share,
    }
    if args.dry_run:
        default = [DEFAULT_LEGIT_RATIO if args.legit_ratio is None else args.legit_ratio]
        ratios = [float(ratio) for ratio in args.ratios.split(",")] if args.ratios else default
        _print_dry_run(dry_run(args.data_dir, ratios, **shared))
        return
    if args.ratios:
        parser.error("--ratios requires --dry-run")
    _print_report(assemble(args.data_dir, legit_ratio=args.legit_ratio, **shared), args.data_dir)


if __name__ == "__main__":
    main()

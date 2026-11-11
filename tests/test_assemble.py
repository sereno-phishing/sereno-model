import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from sereno_model.data.acquire import ChecksumMismatchError
from sereno_model.data.assemble import assemble, dry_run, main, quota
from sereno_model.data.clean import DOMAINS_SCHEMA, URLS_SCHEMA, content_hash, write_parquet
from sereno_model.data.normalize import normalize_url, parse_url

NOW = datetime(2026, 11, 12, 9, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 11, 12, 10, 0, 0, tzinfo=UTC)
STAMP = "20261112T090000Z"
CLEAN_STAMP = "20261111T120000Z"

PHISHING, LEGITIMATE = 1, 0


def many(count: int, template: str) -> list[str]:
    return [template.format(index) for index in range(count)]


def make_clean_run(
    root: Path,
    phishing: list[str],
    legitimate: list[str],
    majestic: list[str] = (),
    stamp: str = CLEAN_STAMP,
) -> Path:
    """Write a tiny cleaning run (URLs, ranked domains and report) and return the data dir."""
    rows = sorted([(url, PHISHING) for url in phishing] + [(url, LEGITIMATE) for url in legitimate])
    urls = pd.DataFrame(
        {
            "url": [url for url, _ in rows],
            "label": [label for _, label in rows],
            "sources": [["phishtank"] if label else ["url_phish_2026"] for _, label in rows],
            "has_path_or_query": [parse_url(url).has_path_or_query for url, _ in rows],
            "submission_time": [
                "2026-11-01T10:00:00+00:00" if label else None for _, label in rows
            ],
        }
    ).astype({"label": "int8", "has_path_or_query": "bool"})
    domains = pd.DataFrame(
        {"domain": list(majestic), "global_rank": range(1, len(majestic) + 1)}
    ).astype({"global_rank": "int64"})
    data_dir = root / "data"
    run_dir = data_dir / "processed" / stamp
    run_dir.mkdir(parents=True)
    write_parquet(urls, URLS_SCHEMA, run_dir / "urls.parquet")
    write_parquet(domains, DOMAINS_SCHEMA, run_dir / "majestic_domains.parquet")
    report = {
        "output_dir": f"processed/{stamp}",
        "content_hash": {"value": content_hash(list(urls["url"]), list(urls["label"]))},
    }
    (run_dir / "clean_report.json").write_text(json.dumps(report), encoding="utf-8")
    return data_dir


def read_dataset(data_dir: Path, stamp: str = STAMP) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "processed" / stamp / "dataset.parquet")


def strata(report: dict) -> dict[str, dict]:
    return {row["stratum"]: row for row in report["strata"]}


def test_quota_rounds_ratio_times_phishing_count_half_up():
    assert quota(0.5, 4) == 2
    assert quota(0.5, 3) == 2
    assert quota(0.5, 5) == 3
    assert quota(0.59, 100) == 59
    assert quota(0.3, 1) == 0
    assert quota(0.3, 0) == 0


def test_keeps_every_phishing_url_and_samples_the_quota_of_each_stratum(tmp_path):
    phishing = many(4, "https://bad{}.example/") + many(6, "https://bad{}.example/login")
    legitimate = many(5, "https://good{}.example/") + many(5, "https://good{}.example/about")
    data_dir = make_clean_run(tmp_path, phishing, legitimate)

    report = assemble(data_dir, legit_ratio=0.5, now=NOW)

    frame = read_dataset(data_dir)
    assert set(frame.loc[frame["label"] == PHISHING, "url"]) == set(phishing)
    chosen = frame[frame["label"] == LEGITIMATE]
    assert set(chosen["url"]) <= set(legitimate)
    assert int((~chosen["has_path_or_query"]).sum()) == 2
    assert int(chosen["has_path_or_query"].sum()) == 3
    assert not frame["synthetic"].any()
    rows = strata(report)
    assert len(rows) == 8
    assert rows["no_path/https/no_www"] == {
        "stratum": "no_path/https/no_www",
        "has_path_or_query": False,
        "scheme": "https",
        "has_www": False,
        "phishing": 4,
        "quota": 2,
        "legitimate_available": 5,
        "real_taken": 2,
        "synthetic_added": 0,
        "borrowed_from_other_strata": 0,
        "lent_to_other_strata": 0,
        "shortfall": 0,
        "legitimate_final": 2,
    }
    assert rows["path/https/no_www"]["quota"] == 3
    assert rows["path/https/no_www"]["real_taken"] == 3
    assert rows["path/http/www"]["phishing"] == 0
    assert report["totals"] == {
        "rows": 15,
        "phishing": 10,
        "legitimate": 5,
        "legitimate_quota": 5,
        "legitimate_real": 5,
        "legitimate_synthetic": 0,
        "shortfall": 0,
    }


def test_writes_expected_columns_types_and_a_deterministic_order(tmp_path):
    phishing = many(4, "https://bad{}.example/") + many(4, "http://www.bad{}.example/a?b=1")
    legitimate = many(1, "https://good{}.example/") + many(3, "http://www.good{}.example/a")
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(5, "site{}.example"))

    assemble(data_dir, legit_ratio=0.5, max_majestic_share=0.5, now=NOW)

    frame = read_dataset(data_dir)
    assert list(frame.columns) == [
        "url",
        "label",
        "sources",
        "has_path_or_query",
        "scheme",
        "has_www",
        "synthetic",
        "submission_time",
    ]
    assert list(frame["url"]) == sorted(frame["url"])
    schema = pq.read_schema(data_dir / "processed" / STAMP / "dataset.parquet")
    types = {name: str(schema.field(name).type) for name in schema.names}
    assert types["label"] == "int8"
    assert types["has_path_or_query"] == types["has_www"] == types["synthetic"] == "bool"
    assert types["url"] == types["scheme"] == types["submission_time"] == "string"
    assert types["sources"].startswith("list<")
    by_url = {row["url"]: row for row in frame.to_dict("records")}
    row = by_url["http://www.bad0.example/a?b=1"]
    assert (row["scheme"], row["has_www"], row["has_path_or_query"]) == ("http", True, True)
    assert list(row["sources"]) == ["phishtank"]
    assert row["submission_time"] == "2026-11-01T10:00:00+00:00"
    assert by_url["https://bad0.example/"]["has_www"] is False


def test_takes_real_legitimate_urls_before_adding_synthetic_ones(tmp_path):
    phishing = many(6, "https://bad{}.example")
    legitimate = many(2, "https://good{}.example")
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(10, "site{}.example"))

    report = assemble(data_dir, legit_ratio=0.5, max_majestic_share=0.5, now=NOW)

    frame = read_dataset(data_dir)
    chosen = frame[frame["label"] == LEGITIMATE]
    assert set(legitimate) <= set(chosen["url"])
    assert int(chosen["synthetic"].sum()) == 1
    row = strata(report)["no_path/https/no_www"]
    assert (row["quota"], row["real_taken"]) == (3, 2)
    assert (row["synthetic_added"], row["shortfall"]) == (1, 0)
    assert report["synthetic"]["share_of_legitimate"] == pytest.approx(1 / 3, abs=1e-6)
    assert report["synthetic"]["cap_binding"] is False


def test_synthetic_urls_follow_the_stratum_form_and_the_phishing_trailing_slash_share(tmp_path):
    phishing = many(6, "http://www.bad{}.example/") + many(2, "http://www.worse{}.example")
    domains = many(20, "site{}.example")
    data_dir = make_clean_run(tmp_path, phishing, [], domains)

    report = assemble(data_dir, legit_ratio=1.0, max_majestic_share=1.0, now=NOW)

    frame = read_dataset(data_dir)
    synthetic = frame[frame["synthetic"]]
    assert len(synthetic) == 8
    assert all(url.startswith("http://www.site") for url in synthetic["url"])
    assert sum(url.endswith("/") for url in synthetic["url"]) == 6
    assert all(normalize_url(url) == url for url in synthetic["url"])
    assert all(url.removeprefix("http://www.").rstrip("/") in domains for url in synthetic["url"])
    assert set(synthetic["label"]) == {LEGITIMATE}
    assert all(list(sources) == ["majestic_million"] for sources in synthetic["sources"])
    assert not synthetic["has_path_or_query"].any()
    assert set(synthetic["scheme"]) == {"http"}
    assert synthetic["has_www"].all()
    assert synthetic["submission_time"].isna().all()
    assert report["synthetic"]["share_of_legitimate"] == 1.0
    assert report["form_distance"]["before"]["all_strata"] is None
    assert report["form_distance"]["after"]["all_strata"] == 0.0


def test_never_builds_synthetic_urls_from_domains_already_present_as_hosts(tmp_path):
    phishing = [
        "https://hacked.example/",
        "https://www.spoof.example/",
        "https://shared.example:8443/",
        *many(9, "https://bad{}.example/"),
    ]
    legitimate = ["http://www.known.example/about", "http://plain.example/about"]
    majestic = [
        "hacked.example",
        "spoof.example",
        "shared.example",
        "known.example",
        "www.plain.example",
        "www.prefixed.example",
        "free0.example",
        "free1.example",
        "free2.example",
        "beyond-top-n.example",
    ]
    data_dir = make_clean_run(tmp_path, phishing, legitimate, majestic)

    report = assemble(
        data_dir, legit_ratio=0.5, majestic_top_n=9, max_majestic_share=1.0, now=NOW
    )

    frame = read_dataset(data_dir)
    synthetic = list(frame.loc[frame["synthetic"], "url"])
    hosts = {parse_url(url).host for url in synthetic}
    assert hosts == {"free0.example", "free1.example", "free2.example"}
    assert report["majestic_candidates"] == {
        "top_n_domains": 9,
        "excluded_existing_host": 5,
        "excluded_www_prefix": 1,
        "excluded_not_normalized": 0,
        "available": 3,
        "used": 3,
    }
    row = strata(report)["no_path/https/no_www"]
    assert (row["quota"], row["synthetic_added"], row["shortfall"]) == (6, 3, 3)
    assert report["synthetic"]["candidates_exhausted"] is True


def test_dataset_has_no_duplicate_urls_and_uses_each_domain_once(tmp_path):
    phishing = (
        many(10, "https://bad{}.example/")
        + many(10, "http://bad{}.example")
        + many(10, "https://www.bad{}.example/")
        + many(10, "https://bad{}.example/login")
    )
    legitimate = many(2, "https://good{}.example/") + many(20, "https://good{}.example/about")
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(40, "site{}.example"))

    assemble(data_dir, legit_ratio=1.0, max_majestic_share=1.0, now=NOW)

    frame = read_dataset(data_dir)
    assert frame["url"].is_unique
    synthetic = frame[frame["synthetic"]]
    assert len(synthetic) == 28
    hosts = [parse_url(url).host.removeprefix("www.") for url in synthetic["url"]]
    assert len(set(hosts)) == len(hosts)


def test_with_path_shortfall_borrows_from_other_with_path_strata(tmp_path):
    phishing = many(8, "https://bad{}.example/login") + many(2, "https://bad{}.example/")
    legitimate = (
        ["https://good.example/about"]
        + many(10, "http://www.good{}.example/about")
        + many(5, "https://good{}.example/")
    )
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(5, "site{}.example"))

    report = assemble(data_dir, legit_ratio=0.5, now=NOW)

    frame = read_dataset(data_dir)
    chosen = frame[frame["label"] == LEGITIMATE]
    assert len(chosen) == 5
    assert int(chosen["has_path_or_query"].sum()) == 4
    assert "https://good.example/about" in set(chosen["url"])
    assert not chosen["synthetic"].any()
    rows = strata(report)
    short = rows["path/https/no_www"]
    assert (short["quota"], short["real_taken"], short["borrowed_from_other_strata"]) == (4, 1, 3)
    assert (short["shortfall"], short["legitimate_final"]) == (0, 1)
    donor = rows["path/http/www"]
    assert (donor["quota"], donor["lent_to_other_strata"], donor["legitimate_final"]) == (0, 3, 3)
    assert report["with_path_share"] == {
        "phishing": 0.8,
        "legitimate_before": pytest.approx(11 / 16, abs=1e-6),
        "legitimate_after": 0.8,
    }
    distance = report["form_distance"]
    assert distance["after"] == {"all_strata": 0.6, "with_path": 0.75, "without_path": 0.0}
    assert distance["before"]["with_path"] == pytest.approx(10 / 11, abs=1e-6)
    assert distance["before"]["all_strata"] == pytest.approx(0.8 - 1 / 16, abs=1e-6)


def test_reports_shortfall_when_real_urls_with_path_run_out(tmp_path):
    phishing = many(8, "https://bad{}.example/login") + many(4, "http://bad{}.example/login")
    legitimate = many(2, "https://good{}.example/about") + ["http://www.good.example/about"]
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(5, "site{}.example"))

    report = assemble(data_dir, legit_ratio=0.5, now=NOW)

    rows = strata(report)
    first, second = rows["path/https/no_www"], rows["path/http/no_www"]
    assert (first["quota"], first["real_taken"]) == (4, 2)
    assert (second["quota"], second["real_taken"]) == (2, 0)
    assert first["borrowed_from_other_strata"] + second["borrowed_from_other_strata"] == 1
    assert first["shortfall"] + second["shortfall"] == 3
    assert rows["path/http/www"]["lent_to_other_strata"] == 1
    totals = report["totals"]
    assert (totals["legitimate_quota"], totals["legitimate"], totals["shortfall"]) == (6, 3, 3)
    assert len(read_dataset(data_dir)) == 15


def test_caps_synthetic_rows_as_a_share_of_the_legitimate_class(tmp_path):
    phishing = many(8, "https://bad{}.example/") + many(8, "https://bad{}.example/login")
    legitimate = many(4, "https://good{}.example/about")
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(20, "site{}.example"))

    report = assemble(data_dir, legit_ratio=0.5, max_majestic_share=0.25, now=NOW)

    frame = read_dataset(data_dir)
    assert int(frame["synthetic"].sum()) == 1
    assert report["synthetic"] == {
        "share_of_legitimate": 0.2,
        "requested": 4,
        "added": 1,
        "cap_limit": 1,
        "cap_binding": True,
        "candidates_exhausted": False,
    }
    assert strata(report)["no_path/https/no_www"]["shortfall"] == 3
    assert report["totals"]["legitimate"] == 5
    assert report["totals"]["shortfall"] == 3


def test_fails_when_urls_parquet_does_not_reproduce_the_recorded_hash(tmp_path):
    data_dir = make_clean_run(
        tmp_path, many(4, "https://bad{}.example/"), many(4, "https://good{}.example/")
    )
    report_path = data_dir / "processed" / CLEAN_STAMP / "clean_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["content_hash"]["value"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ChecksumMismatchError, match=CLEAN_STAMP):
        assemble(data_dir, legit_ratio=0.5, now=NOW)

    assert not (data_dir / "processed" / STAMP).exists()


def test_same_seed_reproduces_the_content_hash_and_another_seed_changes_it(tmp_path):
    phishing = many(20, "https://bad{}.example/") + many(20, "https://bad{}.example/login")
    legitimate = many(30, "https://good{}.example/") + many(30, "https://good{}.example/about")
    data_dir = make_clean_run(tmp_path, phishing, legitimate)

    first = assemble(data_dir, legit_ratio=0.5, seed=7, now=NOW)
    again = assemble(data_dir, legit_ratio=0.5, seed=7, now=LATER)
    other = assemble(data_dir, legit_ratio=0.5, seed=8, now=datetime(2026, 11, 12, 11, tzinfo=UTC))

    assert first["content_hash"]["value"] == again["content_hash"]["value"]
    assert first["content_hash"]["value"] != other["content_hash"]["value"]
    assert again["output_dir"] == "processed/20261112T100000Z"
    assert list(read_dataset(data_dir)["url"]) == list(
        read_dataset(data_dir, "20261112T100000Z")["url"]
    )


def test_synthetic_choice_is_reproducible_and_depends_on_the_seed(tmp_path):
    phishing = many(10, "https://bad{}.example/")
    data_dir = make_clean_run(tmp_path, phishing, [], many(200, "site{}.example"))

    def synthetic_urls(seed: int, moment: datetime) -> list[str]:
        assemble(data_dir, legit_ratio=1.0, seed=seed, max_majestic_share=1.0, now=moment)
        frame = read_dataset(data_dir, moment.strftime("%Y%m%dT%H%M%SZ"))
        return list(frame.loc[frame["synthetic"], "url"])

    first = synthetic_urls(7, NOW)

    assert first == synthetic_urls(7, LATER)
    assert first != synthetic_urls(8, datetime(2026, 11, 12, 11, tzinfo=UTC))


def test_content_hash_covers_sorted_url_label_synthetic_lines(tmp_path):
    data_dir = make_clean_run(
        tmp_path, ["https://bad0.example/", "https://bad1.example/"], [], ["site.example"]
    )
    lines = [
        "https://bad0.example/\t1\t0\n",
        "https://bad1.example/\t1\t0\n",
        "https://site.example/\t0\t1\n",
    ]

    report = assemble(data_dir, legit_ratio=0.5, max_majestic_share=1.0, now=NOW)

    expected = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert report["content_hash"]["value"] == expected
    assert report["content_hash"]["algorithm"] == "sha256"


def test_report_records_the_cleaning_run_parameters_and_versions(tmp_path):
    data_dir = make_clean_run(
        tmp_path, many(4, "https://bad{}.example/"), many(4, "https://good{}.example/")
    )
    clean_report = json.loads(
        (data_dir / "processed" / CLEAN_STAMP / "clean_report.json").read_text(encoding="utf-8")
    )

    report = assemble(data_dir, now=NOW)

    saved = json.loads(
        (data_dir / "processed" / STAMP / "assemble_report.json").read_text(encoding="utf-8")
    )
    assert saved == report
    assert report["created_at"] == "2026-11-12T09:00:00Z"
    assert report["output_dir"] == f"processed/{STAMP}"
    assert report["clean_run"]["dir"] == f"processed/{CLEAN_STAMP}"
    assert report["clean_run"]["content_hash"] == clean_report["content_hash"]["value"]
    assert len(report["clean_run"]["majestic_domains_sha256"]) == 64
    parameters = report["parameters"]
    assert parameters["legit_ratio"] == 0.59
    assert parameters["legit_ratio_source"] == "default"
    assert "2026-11-11" in parameters["legit_ratio_note"]
    assert parameters["seed"] == 20261111
    assert parameters["majestic_top_n"] == 100000
    assert parameters["max_majestic_share"] == 0.25
    assert set(report["versions"]) == {"python", "pandas", "pyarrow"}
    assert report["form_distance"]["after"]["all_strata"] == 0.0

    explicit = assemble(data_dir, legit_ratio=0.5, now=LATER)
    assert explicit["parameters"]["legit_ratio_source"] == "explicit"
    assert explicit["parameters"]["legit_ratio_note"] is None


def test_uses_the_latest_cleaning_run_unless_one_is_named(tmp_path):
    older = many(4, "https://old{}.example/")
    newer = many(4, "https://new{}.example/")
    data_dir = make_clean_run(tmp_path, older, [], stamp="20261110T000000Z")
    make_clean_run(tmp_path, newer, [], stamp=CLEAN_STAMP)

    latest = assemble(data_dir, legit_ratio=0.5, now=NOW)
    after_own_output = assemble(data_dir, legit_ratio=0.5, now=LATER)
    named = assemble(
        data_dir,
        clean_run="20261110T000000Z",
        legit_ratio=0.5,
        now=datetime(2026, 11, 12, 11, tzinfo=UTC),
    )

    assert latest["clean_run"]["dir"] == f"processed/{CLEAN_STAMP}"
    assert after_own_output["clean_run"]["dir"] == f"processed/{CLEAN_STAMP}"
    assert named["clean_run"]["dir"] == "processed/20261110T000000Z"
    assert set(read_dataset(data_dir)["url"]) == set(newer)
    assert set(read_dataset(data_dir, "20261112T110000Z")["url"]) == set(older)


def test_fails_without_a_cleaning_run(tmp_path):
    (tmp_path / "data" / "processed").mkdir(parents=True)

    with pytest.raises(LookupError, match="cleaning run"):
        assemble(tmp_path / "data", now=NOW)


def test_rejects_parameters_out_of_range(tmp_path):
    data_dir = make_clean_run(tmp_path, many(4, "https://bad{}.example/"), [])

    with pytest.raises(ValueError, match="legit_ratio"):
        assemble(data_dir, legit_ratio=0, now=NOW)
    with pytest.raises(ValueError, match="max_majestic_share"):
        assemble(data_dir, max_majestic_share=1.5, now=NOW)
    with pytest.raises(ValueError, match="majestic_top_n"):
        assemble(data_dir, majestic_top_n=-1, now=NOW)


def test_refuses_to_overwrite_an_existing_output_folder(tmp_path):
    data_dir = make_clean_run(tmp_path, many(4, "https://bad{}.example/"), [])

    assemble(data_dir, now=NOW)
    with pytest.raises(FileExistsError):
        assemble(data_dir, now=NOW)


def test_dry_run_summarizes_each_ratio_without_writing(tmp_path):
    phishing = many(8, "https://bad{}.example/") + many(8, "https://bad{}.example/login")
    legitimate = many(4, "https://good{}.example/about")
    data_dir = make_clean_run(tmp_path, phishing, legitimate, many(20, "site{}.example"))

    rows = dry_run(data_dir, [0.25, 0.5])

    assert [row["legit_ratio"] for row in rows] == [0.25, 0.5]
    assert rows[0]["legitimate"] == 2 + 0
    assert rows[0]["shortfall"] == 2
    assert rows[1]["legitimate"] == 5
    assert rows[1]["synthetic"] == 1
    assert rows[1]["synthetic_share"] == 0.2
    assert rows[1]["shortfall"] == 3
    assert rows[1]["shortfall_by_stratum"] == {"no_path/https/no_www": 3}
    assert rows[1]["form_distance"] == {"all_strata": 0.3, "with_path": 0.0, "without_path": 0.0}
    assert [path.name for path in (data_dir / "processed").iterdir()] == [CLEAN_STAMP]


def test_cli_dry_run_prints_one_line_per_ratio_and_writes_nothing(tmp_path, capsys):
    data_dir = make_clean_run(
        tmp_path, many(8, "https://bad{}.example/"), many(8, "https://good{}.example/")
    )

    main(["--data-dir", str(data_dir), "--dry-run", "--ratios", "0.25,0.5"])

    output = capsys.readouterr().out
    assert "0.25" in output and "0.5" in output
    assert [path.name for path in (data_dir / "processed").iterdir()] == [CLEAN_STAMP]


def test_cli_assembles_the_given_data_dir_and_prints_a_summary(tmp_path, capsys):
    data_dir = make_clean_run(
        tmp_path, many(8, "https://bad{}.example/"), many(8, "https://good{}.example/")
    )

    main(["--data-dir", str(data_dir), "--legit-ratio", "0.75", "--seed", "3"])

    output = capsys.readouterr().out
    assert "8 phishing" in output and "6 legitimate" in output
    folders = [path for path in (data_dir / "processed").iterdir() if path.name != CLEAN_STAMP]
    assert len(folders) == 1
    report = json.loads((folders[0] / "assemble_report.json").read_text(encoding="utf-8"))
    assert report["parameters"]["legit_ratio"] == 0.75
    assert report["parameters"]["seed"] == 3
    assert (folders[0] / "dataset.parquet").exists()

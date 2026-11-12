import math
import subprocess
import sys
from collections import Counter

import pytest

from sereno_model.data.normalize import InvalidUrlError, parse_url
from sereno_model.features import extract as extract_module
from sereno_model.features.extract import (
    FEATURE_NAMES,
    FEATURE_SET_VERSION,
    FLOAT_FEATURES,
    extract_features,
    features_from_parts,
    measured_text,
)

DOCUMENTED_ORDER = (
    "url_length",
    "dot_count",
    "has_at",
    "host_length",
    "subdomain_count",
    "host_hyphen_count",
    "host_is_ip",
    "has_port",
    "tld_length",
    "path_length",
    "path_depth",
    "has_double_slash",
    "query_length",
    "query_param_count",
    "digit_count",
    "digit_ratio",
    "question_mark_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "underscore_count",
    "tilde_count",
    "slash_count",
    "url_entropy",
)


def named(url: str) -> dict:
    return dict(zip(FEATURE_NAMES, extract_features(url), strict=True))


def shannon_bits(text: str) -> float:
    return -sum(n / len(text) * math.log2(n / len(text)) for n in Counter(text).values())


def test_feature_names_follow_the_documented_order_and_version():
    assert FEATURE_NAMES == DOCUMENTED_ORDER
    assert isinstance(FEATURE_NAMES, tuple)
    assert FEATURE_SET_VERSION == "2"
    assert FLOAT_FEATURES == {"digit_ratio", "url_entropy"}


def test_scheme_and_www_are_not_features():
    assert not [name for name in FEATURE_NAMES if "scheme" in name or "https" in name]
    assert not [name for name in FEATURE_NAMES if "www" in name]


def test_worked_example_matches_every_documented_definition():
    url = "https://www.Shop.example.co.uk:8443/a/b//c.php?id=12&x=y_z~%20#top"
    text = "shop.example.co.uk:8443/a/b//c.php?id=12&x=y_z~%20"

    assert measured_text(parse_url(url)) == text
    assert named(url) == {
        "url_length": 50,
        "dot_count": 4,
        "has_at": 0,
        "host_length": 18,
        "subdomain_count": 1,
        "host_hyphen_count": 0,
        "host_is_ip": 0,
        "has_port": 1,
        "tld_length": 5,
        "path_length": 11,
        "path_depth": 3,
        "has_double_slash": 1,
        "query_length": 15,
        "query_param_count": 2,
        "digit_count": 8,
        "digit_ratio": pytest.approx(8 / 50),
        "question_mark_count": 1,
        "equals_count": 2,
        "ampersand_count": 1,
        "percent_count": 1,
        "underscore_count": 1,
        "tilde_count": 1,
        "slash_count": 4,
        "url_entropy": pytest.approx(shannon_bits(text)),
    }


def test_returns_a_tuple_of_ints_with_two_floats():
    values = extract_features("https://example.com/a1?b=2")

    assert isinstance(values, tuple) and len(values) == len(FEATURE_NAMES)
    for name, value in zip(FEATURE_NAMES, values, strict=True):
        assert type(value) is (float if name in FLOAT_FEATURES else int), name


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("http://www.example.com/a?b=1", "https://example.com/a?b=1"),
        ("http://example.com", "https://www.example.com"),
        ("HTTP://WWW.Example.com:80/a", "https://example.com:443/a"),
        ("http://www.login.example.com/x//y", "https://login.example.com/x//y"),
    ],
)
def test_scheme_and_www_do_not_leak_into_the_vector(first, second):
    assert extract_features(first) == extract_features(second)


def test_www_is_removed_before_measuring_length_and_subdomains():
    found = named("http://www.a.b.example.com/")

    assert (found["host_length"], found["subdomain_count"]) == (len("a.b.example.com"), 2)
    assert found["url_length"] == len("a.b.example.com")
    assert found["dot_count"] == 3


def test_only_the_leading_www_label_is_removed():
    assert named("http://www2.example.com")["subdomain_count"] == 1
    assert named("http://a.www.example.com")["subdomain_count"] == 2
    assert named("http://wwwexample.com")["host_length"] == len("wwwexample.com")


def test_counts_hyphens_in_the_host_only():
    found = named("https://secure-login.my-bank.com/a-b-c")

    assert found["host_hyphen_count"] == 2


@pytest.mark.parametrize(
    "url",
    ["http://203.0.113.9/panel/1", "http://[2001:db8::1]/x", "http://[::ffff:192.0.2.1]:8080/"],
)
def test_ip_hosts_have_no_subdomains_and_no_suffix(url):
    found = named(url)

    assert (found["host_is_ip"], found["subdomain_count"], found["tld_length"]) == (1, 0, 0)


@pytest.mark.parametrize(
    "url",
    [
        "http://1.2.3/",
        "http://1.2.3.4.5/",
        "http://256.1.1.1/",
        "http://3232235777/",
        "http://0x7f.0.0.1/",
        "http://[not-an-address]/",
        "http://1.2.3.4.example.com/",
    ],
)
def test_only_dotted_quad_ipv4_and_bracketed_ipv6_count_as_ip(url):
    assert named(url)["host_is_ip"] == 0


def test_ipv6_colons_are_not_a_port():
    assert named("http://[2001:db8::1]/")["has_port"] == 0
    assert named("http://[2001:db8::1]:8080/")["has_port"] == 1


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://example.com/", 0),
        ("http://example.com:80/", 0),
        ("https://example.com:443/", 0),
        ("http://example.com:8080/", 1),
        ("https://example.com:80/", 0),
        ("http://example.com:443/", 0),
        ("http://example.com:/", 0),
    ],
)
def test_has_port_only_for_an_explicit_non_default_port(url, expected):
    assert named(url)["has_port"] == expected


@pytest.mark.parametrize(
    "rest",
    ["example.com/a?b=1", "www.example.com/a", "www.www.example.com", "example.com:8080/a//b"],
)
def test_a_www_prefix_and_the_scheme_never_change_the_vector(rest):
    # Holds for any host, also one that still starts with ``www.`` after the prefix is removed.
    assert extract_features(f"http://www.{rest}") == extract_features(f"https://{rest}")


@pytest.mark.parametrize(
    ("with_slash", "without_slash"),
    [
        ("https://example.com/", "https://example.com"),
        ("https://example.com/a/b/", "https://example.com/a/b"),
        ("https://example.com//", "https://example.com"),
    ],
)
def test_a_slash_that_ends_the_url_never_changes_the_vector(with_slash, without_slash):
    # Sources differ in whether they write the final slash; it must not tell them apart.
    assert extract_features(with_slash) == extract_features(without_slash)


def test_a_slash_before_the_query_is_kept():
    assert extract_features("https://example.com/a/?x=1") != extract_features(
        "https://example.com/a?x=1"
    )


@pytest.mark.parametrize("port", ["80", "443", "8080"])
def test_the_scheme_does_not_reach_the_vector_through_the_port(port):
    # Normalization keeps ``:443`` on http and drops it on https; the vector must not notice.
    over_http = extract_features(f"http://example.com:{port}/a")
    over_https = extract_features(f"https://example.com:{port}/a")

    assert over_http == over_https


@pytest.mark.parametrize(
    ("url", "subdomains", "tld_length"),
    [
        ("http://example.com/", 0, 3),
        ("http://a.b.example.co.uk/", 2, 5),
        ("https://victim.pages.dev/pay", 1, 3),
        ("http://co.uk/", 0, 5),
        ("http://localhost/", 0, 0),
        ("http://files.server.invalidsuffixzz/", 0, 0),
    ],
)
def test_subdomains_and_suffix_come_from_icann_suffixes_only(url, subdomains, tld_length):
    found = named(url)

    assert (found["subdomain_count"], found["tld_length"]) == (subdomains, tld_length)


def test_userinfo_stays_in_the_measured_text_but_not_in_the_host():
    found = named("http://user:pw@www.example.com/a")

    assert found["has_at"] == 1
    assert found["host_length"] == len("example.com")
    assert found["url_length"] == len("user:pw@example.com/a")
    assert found["has_port"] == 0


def test_has_at_also_sees_the_path_and_the_query():
    assert named("http://example.com/a@b")["has_at"] == 1
    assert named("http://example.com/?mail=a@b.com")["has_at"] == 1
    assert named("http://example.com/a")["has_at"] == 0


@pytest.mark.parametrize(
    ("url", "length", "depth", "slashes"),
    [
        ("http://example.com", 0, 0, 0),
        ("http://example.com/", 0, 0, 0),
        ("http://example.com/a", 2, 1, 1),
        ("http://example.com/a/", 2, 1, 1),
        ("http://example.com/a//b///", 5, 2, 3),
        ("http://example.com/a/?x=1", 3, 1, 2),
    ],
)
def test_path_length_and_depth_ignore_the_slashes_that_end_the_url(url, length, depth, slashes):
    found = named(url)

    assert (found["path_length"], found["path_depth"], found["slash_count"]) == (
        length,
        depth,
        slashes,
    )


@pytest.mark.parametrize(
    ("url", "length", "parameters", "question_marks"),
    [
        ("http://example.com/a", 0, 0, 0),
        ("http://example.com/a?", 0, 0, 1),
        ("http://example.com/a?x", 1, 1, 1),
        ("http://example.com/a?x=1&y=2", 7, 2, 1),
        ("http://example.com/a?x=1&&y=2&", 9, 2, 1),
        ("http://example.com/a?x=1?y=2", 7, 1, 2),
    ],
)
def test_query_length_and_parameters(url, length, parameters, question_marks):
    found = named(url)

    assert (found["query_length"], found["query_param_count"], found["question_mark_count"]) == (
        length,
        parameters,
        question_marks,
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://example.com/a/b", 0),
        ("http://example.com//a", 1),
        ("http://example.com/a?next=//evil.example", 1),
        ("http://example.com/a?next=http://evil.example", 1),
    ],
)
def test_double_slash_ignores_the_one_after_the_scheme(url, expected):
    assert named(url)["has_double_slash"] == expected


def test_digit_count_and_ratio_use_ascii_digits_of_the_measured_text():
    found = named("http://a1.example.com:8080/p2?q=345")

    assert found["digit_count"] == 9
    assert found["digit_ratio"] == pytest.approx(9 / len("a1.example.com:8080/p2?q=345"))
    assert named("http://example.com/١٢")["digit_count"] == 0


def test_digit_ratio_is_zero_without_digits():
    found = named("http://example.com/")

    assert found["digit_ratio"] == 0.0
    assert extract_module._ratio(0, 0) == 0.0


def test_entropy_is_shannon_bits_over_the_measured_text():
    assert named("http://aaaa")["url_entropy"] == 0.0
    assert named("http://ab")["url_entropy"] == pytest.approx(1.0)
    assert named("https://www.abcd")["url_entropy"] == pytest.approx(2.0)


def test_counts_the_remaining_symbols():
    found = named("http://example.com/~u/a_b_c?x=%41%2F&y=1&z=_")

    assert found["tilde_count"] == 1
    assert found["underscore_count"] == 3
    assert found["percent_count"] == 1
    assert found["equals_count"] == 3
    assert found["ampersand_count"] == 2


@pytest.mark.parametrize("raw", ["", "   ", "example.com/a", "ftp://example.com/a", "http:///a"])
def test_invalid_urls_raise_the_existing_error(raw):
    with pytest.raises(InvalidUrlError):
        extract_features(raw)


def test_extraction_is_deterministic_and_pure():
    url = "https://www.secure-login.example.co.uk:8443/a/b?c=1&d=2"
    parts = parse_url(url)

    first = extract_features(url)

    assert all(extract_features(url) == first for _ in range(5))
    assert features_from_parts(parts) == first
    assert parts == parse_url(url)


def test_extractor_imports_no_data_science_library():
    code = (
        "import sys\n"
        "import sereno_model.features\n"
        "from sereno_model.features.extract import extract_features\n"
        "extract_features('https://example.com/a')\n"
        "heavy = {'pandas', 'numpy', 'sklearn', 'pyarrow', 'scipy'}\n"
        "found = sorted(heavy & {name.split('.')[0] for name in sys.modules})\n"
        "assert not found, found\n"
    )

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr

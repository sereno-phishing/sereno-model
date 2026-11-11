import pytest

from sereno_model.data.normalize import InvalidReason, InvalidUrlError, normalize_url, parse_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # scheme and host lowercased, everything else keeps its case
        ("HTTP://Example.COM/Path?Q=A", "http://example.com/Path?Q=A"),
        ("https://WWW.Example.com", "https://www.example.com"),
        # default port removed, other ports kept
        ("http://example.com:80/a", "http://example.com/a"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:443/a", "http://example.com:443/a"),
        ("https://example.com:80/a", "https://example.com:80/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("http://example.com:80", "http://example.com"),
        # fragment removed
        ("http://example.com/a#section", "http://example.com/a"),
        ("http://example.com/a?b=1#section", "http://example.com/a?b=1"),
        ("http://example.com#section", "http://example.com"),
        # percent-encoded unreserved characters decoded
        ("http://example.com/%41%62%7A%30%2D%2E%5F%7E", "http://example.com/Abz0-._~"),
        ("http://example.com/%7euser?k=%61", "http://example.com/~user?k=a"),
        ("http://ex%61mple.com/", "http://example.com/"),
        # every other percent-encoding left byte for byte
        ("http://example.com/a%2Fb%2fc%20d%3F", "http://example.com/a%2Fb%2fc%20d%3F"),
        ("http://example.com/%E2%82%ac", "http://example.com/%E2%82%ac"),
        ("http://example.com/%2541", "http://example.com/%2541"),
        ("http://example.com/100%/%zz", "http://example.com/100%/%zz"),
        ("http://EXAMPLE%2Ecom%2F/", "http://example.com%2F/"),
        # surrounding whitespace trimmed
        ("  http://example.com/a \n", "http://example.com/a"),
        # userinfo and IPv6 literals survive
        ("http://User:Pass@Example.com:80/", "http://User:Pass@example.com/"),
        ("http://[2001:DB8::1]:8080/a", "http://[2001:db8::1]:8080/a"),
        ("http://[2001:DB8::1]:80/a", "http://[2001:db8::1]/a"),
    ],
)
def test_normalizes_safe_equivalences(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("http://example.com/a", "https://example.com/a"),
        ("http://www.example.com/a", "http://example.com/a"),
        ("http://example.com/a/", "http://example.com/a"),
        ("http://example.com/", "http://example.com"),
        ("http://example.com/?a=1&b=2", "http://example.com/?b=2&a=1"),
        ("http://example.com/A", "http://example.com/a"),
        ("http://example.com/a/../b", "http://example.com/b"),
        ("http://example.com/%2f", "http://example.com/%2F"),
        ("http://example.com/a?", "http://example.com/a"),
    ],
)
def test_keeps_non_equivalent_urls_distinct(first, second):
    assert normalize_url(first) != normalize_url(second)
    assert normalize_url(first) == first
    assert normalize_url(second) == second


def test_normalization_is_idempotent():
    once = normalize_url("HTTP://Example.com:80/%41%2F?x=%7E#frag")

    assert normalize_url(once) == once


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", InvalidReason.EMPTY),
        ("   \t\n", InvalidReason.EMPTY),
        ("ftp://example.com/file", InvalidReason.UNSUPPORTED_SCHEME),
        ("example.com/path", InvalidReason.UNSUPPORTED_SCHEME),
        ("//example.com/path", InvalidReason.UNSUPPORTED_SCHEME),
        ("javascript:alert(1)", InvalidReason.UNSUPPORTED_SCHEME),
        ("http://", InvalidReason.MISSING_HOST),
        ("http:///path", InvalidReason.MISSING_HOST),
        ("http://:80/path", InvalidReason.MISSING_HOST),
        ("http://user@/path", InvalidReason.MISSING_HOST),
        ("http:example.com", InvalidReason.MISSING_HOST),
        ("https://?q=1", InvalidReason.MISSING_HOST),
        ("http://example.com/a b", InvalidReason.WHITESPACE_OR_CONTROL),
        ("http://exa mple.com/", InvalidReason.WHITESPACE_OR_CONTROL),
        ("http://example.com/a\tb", InvalidReason.WHITESPACE_OR_CONTROL),
        ("http://example.com/a\x00b", InvalidReason.WHITESPACE_OR_CONTROL),
        ("http://example.com/a\x7fb", InvalidReason.WHITESPACE_OR_CONTROL),
        ("http://example.com/a b", InvalidReason.WHITESPACE_OR_CONTROL),
    ],
)
def test_rejects_invalid_urls_with_a_reason(raw, reason):
    with pytest.raises(InvalidUrlError) as caught:
        normalize_url(raw)

    assert caught.value.reason is reason


def test_reason_codes_are_distinct_strings():
    codes = [reason.value for reason in InvalidReason]

    assert len(set(codes)) == len(codes) == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://example.com", False),
        ("http://example.com/", False),
        ("http://example.com/#frag", False),
        ("http://example.com/a", True),
        ("http://example.com/?q=1", True),
        ("http://example.com?q=1", True),
        ("http://example.com//", True),
    ],
)
def test_reports_whether_url_has_path_or_query(raw, expected):
    assert parse_url(raw).has_path_or_query is expected


def test_parsed_host_excludes_userinfo_and_port():
    assert parse_url("https://user@Shop.Example.com:8443/a").host == "shop.example.com"

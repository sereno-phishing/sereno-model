import re

import pytest

from sereno_model.features import extract
from sereno_model.features.candidates import (
    CANDIDATE_GROUPS,
    CANDIDATE_NAMES,
    SENSITIVE_WORDS,
    SHORTENER_DOMAINS,
    extract_candidates,
)
from sereno_model.features.extract import FEATURE_NAMES, extract_features

CITATION = re.compile(r"^P\d{2}:\d+(-\d+)?$")


def values(url: str) -> dict:
    return extract_candidates(url)


def test_groups_cover_the_candidate_names_in_order():
    assert CANDIDATE_GROUPS == {
        "G1": ("tld_in_subdomain", "tld_in_path"),
        "G2": ("sensitive_word_count",),
        "G3": ("is_shortener",),
    }
    assert CANDIDATE_NAMES == ("tld_in_subdomain", "tld_in_path", "sensitive_word_count",
                               "is_shortener")
    assert not set(CANDIDATE_NAMES) & set(FEATURE_NAMES)


def test_v2_vector_is_untouched():
    assert extract.FEATURE_SET_VERSION == "2"
    assert len(FEATURE_NAMES) == 24
    assert FEATURE_NAMES[0] == "url_length" and FEATURE_NAMES[-1] == "url_entropy"
    # The extractor still returns a 24-value vector of the same values.
    vector = extract_features("https://www.paypal.com.secure-login.net/bit.ly/Login?a=1")
    assert len(vector) == 24
    assert vector == extract_features("http://paypal.com.secure-login.net/bit.ly/Login?a=1")


def test_lists_hold_only_corpus_cited_entries():
    words = [word for word, _ in SENSITIVE_WORDS]
    assert words == ["login", "log in", "sign in", "update", "validate", "activate", "secure",
                     "verify", "account", "submit", "bank", "bonus", "service"]
    domains = [domain for domain, _ in SHORTENER_DOMAINS]
    assert domains == ["bit.ly", "w.wiki"]
    for _, citations in (*SENSITIVE_WORDS, *SHORTENER_DOMAINS):
        assert citations
        assert all(CITATION.match(citation) for citation in citations)


# G1 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://paypal.com.evil.net/", 1),
        ("https://www.paypal.com.evil.net/", 1),
        ("https://secure.co.uk.evil.net/", 1),
        ("https://de.example.org/", 1),  # a ccTLD label counts, as defined
        ("https://login.example.org/", 0),
        ("https://example.org/", 0),
        ("https://www.example.org/", 0),
        ("https://192.168.0.1/paypal.com/", 0),
        ("https://localhost/", 0),
        ("https://a..example.org/", 0),
    ],
)
def test_tld_in_subdomain(url, expected):
    assert values(url)["tld_in_subdomain"] == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://evil.net/paypal.com/login", 1),
        ("https://evil.net/x/www.bank.co.uk", 1),
        ("https://evil.net/PayPal.COM/", 1),
        ("https://evil.net/paypal.com.verify/", 1),
        ("https://evil.net/index.html", 0),
        ("https://evil.net/com/", 0),  # no dot before the label
        ("https://evil.net/a..com", 1),
        ("https://evil.net/.com", 0),  # no label before the dot
        ("https://evil.net", 0),
        ("https://evil.net/?next=paypal.com", 0),  # the query is not the path
        ("https://192.168.0.1/paypal.com", 1),
    ],
)
def test_tld_in_path(url, expected):
    assert values(url)["tld_in_path"] == expected


# G2 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.org/", 0),
        ("https://example.org/login", 1),
        ("https://example.org/LOGIN", 1),
        ("https://example.org/log-in", 1),  # a word split across separators
        ("https://example.org/sign_in", 1),
        ("https://example.org/sign.in/signin", 2),
        ("https://www.secure-bank.com/", 2),
        ("https://www.example.org/", 0),  # the www. prefix is not measured
        ("https://example.org/accounts/update?verify=1&bonuses", 4),
        ("https://example.org/services/banks", 2),
        ("https://example.org/loginlogin", 2),
        ("https://192.168.0.1/submit", 1),
        ("https://user:login@example.org/", 1),  # userinfo is part of the measured text
    ],
)
def test_sensitive_word_count(url, expected):
    assert values(url)["sensitive_word_count"] == expected


def test_sensitive_words_ignore_the_scheme():
    assert values("http://example.org/a")["sensitive_word_count"] == 0
    assert values("https://example.org/a")["sensitive_word_count"] == 0


# G3 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://bit.ly/abc", 1),
        ("https://www.bit.ly/abc", 1),
        ("https://BIT.LY/abc", 1),
        ("https://w.wiki/U", 1),
        ("https://x.bit.ly/", 1),
        ("https://bit.ly.evil.net/", 0),
        ("https://evil.net/bit.ly", 0),
        ("https://tinyurl.com/abc", 0),  # not printed as a domain in the corpus
        ("https://goo.gl/abc", 0),
        ("https://192.168.0.1/", 0),
        ("https://example.org/", 0),
    ],
)
def test_is_shortener(url, expected):
    assert values(url)["is_shortener"] == expected


def test_values_are_ints_in_name_order():
    result = values("https://bit.ly/login")
    assert tuple(result) == CANDIDATE_NAMES
    assert all(type(value) is int for value in result.values())

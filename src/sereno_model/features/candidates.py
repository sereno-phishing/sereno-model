"""Candidate list-based features, measured apart from the base vector (version 2).

These features are not part of ``FEATURE_NAMES`` and never reach the official model: they exist
only so the ablation in ``sereno_model.models.ablation`` can measure what each group adds
(``docs/02-caracteristicas.md``, list-feature measurement approved on 2026-11-12). Like the
extractor, the module needs only the standard library and ``tldextract``, uses the same parsed
URL (:func:`sereno_model.data.normalize.parse_url`) and never logs the URL.

Lists are fixed constants of this module, each entry with the corpus passages that print it
(``P##:line`` in the extracted text of the TDP papers). Nothing is downloaded and nothing is
derived from the data.

Groups and definitions (own implementation details where the corpus gives no exact rule):

* G1 ``tld_in_subdomain``: 1 when a run of one or more consecutive non-empty subdomain labels is
  an ICANN public suffix of the bundled list (``paypal.com.evil.net``; also ``de.example.org``,
  since a country code is a suffix). 0 for an IP host or a host without a registered domain.
* G1 ``tld_in_path``: 1 when a path segment (between slashes, lowercased, without the slashes
  that end the URL) holds a run of consecutive non-empty dot-separated labels that is an ICANN
  public suffix, is preceded by a dot and has a non-empty label somewhere before it
  (``/paypal.com/``). The query is not the path. File extensions that are also suffixes
  (``.zip``, ``.py``, ``.do``) count: the list cannot tell them apart.
* G2 ``sensitive_word_count``: occurrences of the listed words in the measured text of the
  extractor (no scheme, no ``www.``), lowercased and with every character that is not an ASCII
  letter or digit removed; the words are compared after the same removal, so ``log in``,
  ``log-in`` and ``login`` are one word. Substring counting, non-overlapping per word, summed
  over the distinct words: ``banks`` counts ``bank``, and a word can also arise across a
  boundary (``blog.info`` holds ``login``).
* G3 ``is_shortener``: 1 when the registered domain of the host (ICANN suffixes, as in
  ``sereno_model.data.domains``) is a listed shortener domain.
"""

from functools import cache

from sereno_model.data.domains import _extractor, registered_domain
from sereno_model.data.normalize import UrlParts, parse_url
from sereno_model.features.extract import is_ip_address, measured_path, measured_text, strip_www

CANDIDATE_LIST_VERSION = "1"

CANDIDATE_GROUPS = {
    "G1": ("tld_in_subdomain", "tld_in_path"),
    "G2": ("sensitive_word_count",),
    "G3": ("is_shortener",),
}
CANDIDATE_NAMES = tuple(name for names in CANDIDATE_GROUPS.values() for name in names)

# Words printed in the corpus, with the passages that print them. ``bank``, ``bonus`` and
# ``service`` are printed only in the plural (P09); the singular also matches the plural.
# ``signin`` is not printed as such: it is what ``sign in`` becomes after separator removal.
SENSITIVE_WORDS = (
    ("login", ("P10:489-491", "P30:293-295", "P33:555-562")),
    ("log in", ("P09:303-305",)),
    ("sign in", ("P09:303-305",)),
    ("update", ("P09:303-305", "P33:555-562")),
    ("validate", ("P33:555-562",)),
    ("activate", ("P33:555-562",)),
    ("secure", ("P18:358", "P30:293-295", "P33:555-562")),
    ("verify", ("P30:293-295",)),
    ("account", ("P09:303-305", "P10:489-491")),
    ("submit", ("P18:358",)),
    ("bank", ("P09:303-305",)),
    ("bonus", ("P09:303-305",)),
    ("service", ("P09:303-305",)),
)

# Shortener domains printed in the corpus. ``tinyurl.com``, ``goo.gl`` and ``shorturl.at`` are
# not printed as domains (P33 names the service "tinyURL" only), so they are left out.
SHORTENER_DOMAINS = (
    ("bit.ly", ("P09:321", "P10:496", "P25:529")),
    ("w.wiki", ("P33:535",)),
)


def _squeeze(text: str) -> str:
    """``text`` lowercased, keeping only ASCII letters and digits."""
    return "".join(char for char in text.lower() if char.isascii() and char.isalnum())


_WORDS = tuple(dict.fromkeys(_squeeze(word) for word, _ in SENSITIVE_WORDS))
_SHORTENERS = frozenset(domain for domain, _ in SHORTENER_DOMAINS)


@cache
def _is_public_suffix(candidate: str) -> bool:
    """True when ``candidate`` (lowercase, dot-separated) is an ICANN public suffix."""
    found = _extractor()(candidate)
    return not found.domain and found.suffix == candidate


def _has_suffix_run(labels: list[str], first: int) -> bool:
    """True when ``labels[i:j]``, all non-empty and with ``i >= first``, is a public suffix."""
    for start in range(first, len(labels)):
        for end in range(start + 1, len(labels) + 1):
            run = labels[start:end]
            if not run[-1]:
                break
            if _is_public_suffix(".".join(run)):
                return True
    return False


def tld_in_subdomain(parts: UrlParts) -> int:
    host = strip_www(parts.host)
    if is_ip_address(host):
        return 0
    found = _extractor()(host)
    if not (found.domain and found.suffix and found.subdomain):
        return 0
    return int(_has_suffix_run(found.subdomain.split("."), 0))


def tld_in_path(parts: UrlParts) -> int:
    for segment in measured_path(parts).lower().split("/"):
        labels = segment.split(".")
        named = [index for index, label in enumerate(labels) if label]
        # The run must follow a dot and a non-empty label somewhere before it.
        if named and _has_suffix_run(labels, named[0] + 1):
            return 1
    return 0


def sensitive_word_count(parts: UrlParts) -> int:
    text = _squeeze(measured_text(parts))
    return sum(text.count(word) for word in _WORDS)


def is_shortener(parts: UrlParts) -> int:
    host = strip_www(parts.host)
    if is_ip_address(host):
        return 0
    return int(registered_domain(host) in _SHORTENERS)


def candidates_from_parts(parts: UrlParts) -> dict[str, int]:
    """The candidate features of an already parsed URL, in the order of ``CANDIDATE_NAMES``."""
    return {
        "tld_in_subdomain": tld_in_subdomain(parts),
        "tld_in_path": tld_in_path(parts),
        "sensitive_word_count": sensitive_word_count(parts),
        "is_shortener": is_shortener(parts),
    }


def extract_candidates(url: str) -> dict[str, int]:
    """The candidate features of a raw URL; raises ``InvalidUrlError`` like the extractor."""
    return candidates_from_parts(parse_url(url))

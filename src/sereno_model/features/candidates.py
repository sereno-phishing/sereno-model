"""Candidate list-based features (G1 to G3), measured apart from the base vector.

They never reach the official model. Standard library and tldextract only; the URL is
never logged. Lists, papers and definitions: docs/02-caracteristicas.md.
"""

from functools import cache

from sereno_model.data.domains import registered_domain, split_host
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
    found = split_host(candidate)
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
    found = split_host(host)
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

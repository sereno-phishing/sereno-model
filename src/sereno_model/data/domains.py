"""Registered domain of a host, from the Public Suffix List bundled with tldextract.

ICANN suffixes only and no network access (docs/01-datos.md).
"""

import hashlib
import pkgutil
from functools import cache

import tldextract

SNAPSHOT_RESOURCE = ".tld_set_snapshot"


@cache
def _extractor() -> tldextract.TLDExtract:
    return tldextract.TLDExtract(
        cache_dir=None,
        suffix_list_urls=(),
        fallback_to_snapshot=True,
        include_psl_private_domains=False,
    )


def split_host(host: str) -> tldextract.ExtractResult:
    """Subdomain, domain and ICANN suffix of ``host``; the host is not lowercased."""
    return _extractor()(host)


def registered_domain(host: str) -> str:
    """Lowercase registered domain of ``host``.

    A host without one (IP, single label, unknown or bare public suffix) is its own key,
    lowercased and without a leading ``www.``.
    """
    lowered = host.lower()
    found = split_host(lowered)
    if found.domain and found.suffix:
        return f"{found.domain}.{found.suffix}"
    return lowered.removeprefix("www.") or lowered


def suffix_list_snapshot_id() -> str:
    """SHA-256 of the suffix list shipped in the installed tldextract."""
    snapshot = pkgutil.get_data("tldextract", SNAPSHOT_RESOURCE)
    if snapshot is None:
        raise FileNotFoundError(f"tldextract does not ship {SNAPSHOT_RESOURCE}")
    return hashlib.sha256(snapshot).hexdigest()

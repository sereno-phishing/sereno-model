"""Registered domain of a URL host, from the Public Suffix List bundled with tldextract.

Only ICANN suffixes count. Private suffixes such as ``pages.dev`` or ``github.io`` are not
treated as suffixes, so every site hosted on a shared platform gets the same key. The list is
the snapshot shipped inside the installed package: nothing is downloaded and nothing is cached.
"""

import hashlib
import pkgutil
from functools import cache

import tldextract

SNAPSHOT_RESOURCE = ".tld_set_snapshot"


@cache
def _extractor() -> tldextract.TLDExtract:
    """The single extractor of the process, bound to the bundled snapshot."""
    return tldextract.TLDExtract(
        cache_dir=None,
        suffix_list_urls=(),
        fallback_to_snapshot=True,
        include_psl_private_domains=False,
    )


def registered_domain(host: str) -> str:
    """Grouping key of ``host``: its registered domain, in lowercase.

    A host without one (an IP address, a single label, an unknown suffix or a bare public
    suffix) is its own key, in lowercase and without a leading ``www.``.
    """
    lowered = host.lower()
    found = _extractor()(lowered)
    if found.domain and found.suffix:
        return f"{found.domain}.{found.suffix}"
    return lowered.removeprefix("www.") or lowered


def suffix_list_snapshot_id() -> str:
    """SHA-256 of the Public Suffix List snapshot as shipped in the installed tldextract."""
    snapshot = pkgutil.get_data("tldextract", SNAPSHOT_RESOURCE)
    if snapshot is None:
        raise FileNotFoundError(f"tldextract does not ship {SNAPSHOT_RESOURCE}")
    return hashlib.sha256(snapshot).hexdigest()

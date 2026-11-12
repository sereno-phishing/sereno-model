"""Base feature vector of a URL, computed from its text alone (feature set version 2).

The 24 features, their order and their definitions are the ones approved in
``docs/02-caracteristicas.md``. The module is shared by the training pipeline and by the
inference server, so it only needs the standard library and ``tldextract``; it never touches
the network or the disk and never logs the URL, which lives only in memory while it is measured.

Two terms are used throughout:

* *host*: the host of the normalized URL without a leading ``www.``;
* *measured text*: the normalized URL without the scheme, without ``://``, with that host and
  without the slashes that end the URL.

Neither the scheme nor the ``www.`` prefix reaches any feature, so ``http://www.example.com/a``
and ``https://example.com/a`` produce the same vector.

Cases the design document leaves open, as decided here:

* ``www.``: every literal leading ``www.`` label is removed (``www2.`` stays), as long as
  something is left after removing it. Removing only the first one would let the prefix change
  the vector of a host such as ``www.www.example.com``.
* IP hosts: a host is an IP address when it is a dotted-quad decimal IPv4 address or a
  bracketed IPv6 literal that the standard ``ipaddress`` module accepts. Integer, hexadecimal,
  octal or shortened IPv4 spellings (``3232235777``, ``0x7f.0.0.1``, ``1.2.3``) are not.
* Hosts without a registered domain (an IP address, a single label, an unknown suffix or a bare
  public suffix): ``subdomain_count`` is 0. ``tld_length`` is 0 unless the host is itself a
  public suffix, whose length is then reported.
* Suffixes: ICANN suffixes only, so ``victim.pages.dev`` has one subdomain and the suffix
  ``dev``. Empty labels (``a..example.com``) are not counted as subdomains.
* Port: 80 and 443 count as default ports whatever the scheme. The normalization only drops
  the default port of the URL's own scheme, so ``http://host:443/`` would otherwise differ from
  ``https://host:443/`` and the scheme would reach the vector. A default port is left out of
  the measured text and ``has_port`` is 1 only for any other non-empty port. An empty port
  (``host:``) gives 0, although its colon stays in the measured text.
* Userinfo: ``user:password@`` stays in the measured text, so it counts for ``has_at``, the
  length, the symbol counts and the entropy; it is not part of the host.
* Path: the slashes that end the URL are not measured, so ``/a/b/`` counts as ``/a/b`` and a
  bare ``/`` as an empty path; sources differ in whether they write them (version 2 of the
  feature set, after Ghalechyan et al., 2024). A slash that precedes a query is kept. An empty
  path has length 0 and depth 0. Depth counts the non-empty segments between slashes.
* Query: an absent query and an empty one (a bare ``?``) both give length 0 and no parameters;
  the ``?`` itself is never part of the length. Parameters are the non-empty pieces between
  ``&``, so ``a=1&&b=2&`` has two.
* ``has_double_slash`` looks at the path and at the query only; the ``//`` that follows the
  scheme is not in either.
* Digits: ASCII ``0``-``9`` only. ``digit_ratio`` is 0.0 if the measured text were empty, which
  a valid URL cannot produce because its host is never empty.
* Entropy: Shannon entropy in bits over the characters (code points) of the measured text.
"""

import ipaddress
import math
from collections import Counter

from sereno_model.data.domains import _extractor
from sereno_model.data.normalize import DEFAULT_PORTS, UrlParts, parse_url

FEATURE_SET_VERSION = "2"

FEATURE_NAMES = (
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

# Features valued as floats; every other one is an int.
FLOAT_FEATURES = frozenset({"digit_ratio", "url_entropy"})

WWW_PREFIX = "www."
DEFAULT_PORT_NUMBERS = frozenset(DEFAULT_PORTS.values())
ASCII_DIGITS = frozenset("0123456789")

FeatureVector = tuple[int | float, ...]


def strip_www(host: str) -> str:
    """``host`` without its leading ``www.`` labels; a label is kept when nothing would be left."""
    while host.startswith(WWW_PREFIX) and len(host) > len(WWW_PREFIX):
        host = host[len(WWW_PREFIX) :]
    return host


def visible_port(parts: UrlParts) -> str | None:
    """The port of ``parts``, or ``None`` when it is absent or a default port of either scheme."""
    return None if parts.port in DEFAULT_PORT_NUMBERS else parts.port


def measured_path(parts: UrlParts) -> str:
    """The path of ``parts`` without the slashes that end the URL; kept whole before a query."""
    return parts.path.rstrip("/") if parts.query is None else parts.path


def measured_text(parts: UrlParts) -> str:
    """The normalized URL without the scheme, ``://``, the host ``www.``, a default port and
    the slashes that end it."""
    userinfo = "" if parts.userinfo is None else f"{parts.userinfo}@"
    port = visible_port(parts)
    port = "" if port is None else f":{port}"
    query = "" if parts.query is None else f"?{parts.query}"
    return f"{userinfo}{strip_www(parts.host)}{port}{measured_path(parts)}{query}"


def is_ip_address(host: str) -> bool:
    """True for a dotted-quad decimal IPv4 address or a bracketed IPv6 literal."""
    try:
        if host.startswith("[") and host.endswith("]"):
            ipaddress.IPv6Address(host[1:-1])
        else:
            ipaddress.IPv4Address(host)
    except ValueError:
        return False
    return True


def _host_structure(host: str) -> tuple[int, int]:
    """``(subdomain_count, tld_length)`` of a host that is not an IP address."""
    found = _extractor()(host)
    if not (found.domain and found.suffix):
        return 0, len(found.suffix)
    return sum(1 for label in found.subdomain.split(".") if label), len(found.suffix)


def _ratio(part: int, total: int) -> float:
    return part / total if total else 0.0


def _entropy(text: str) -> float:
    """Shannon entropy of the characters of ``text``, in bits; 0.0 for an empty text."""
    total = len(text)
    if not total:
        return 0.0
    # Sorted counts keep the floating-point sum independent of the order of the characters.
    bits = -sum(n / total * math.log2(n / total) for n in sorted(Counter(text).values()))
    return max(bits, 0.0)


def features_from_parts(parts: UrlParts) -> FeatureVector:
    """The feature vector of an already parsed URL, in the order of ``FEATURE_NAMES``."""
    host = strip_www(parts.host)
    text = measured_text(parts)
    path = measured_path(parts)
    query = parts.query or ""
    host_is_ip = is_ip_address(host)
    subdomain_count, tld_length = (0, 0) if host_is_ip else _host_structure(host)
    digit_count = sum(1 for char in text if char in ASCII_DIGITS)

    return (
        len(text),
        text.count("."),
        int("@" in text),
        len(host),
        subdomain_count,
        host.count("-"),
        int(host_is_ip),
        int(bool(visible_port(parts))),
        tld_length,
        len(path),
        sum(1 for segment in path.split("/") if segment),
        int("//" in path or "//" in query),
        len(query),
        sum(1 for piece in query.split("&") if piece),
        digit_count,
        _ratio(digit_count, len(text)),
        text.count("?"),
        text.count("="),
        text.count("&"),
        text.count("%"),
        text.count("_"),
        text.count("~"),
        text.count("/"),
        _entropy(text),
    )


def extract_features(url: str) -> FeatureVector:
    """The feature vector of a raw URL, in the order of ``FEATURE_NAMES``.

    Raises :class:`sereno_model.data.normalize.InvalidUrlError` when ``url`` is not a usable
    http(s) URL.
    """
    return features_from_parts(parse_url(url))

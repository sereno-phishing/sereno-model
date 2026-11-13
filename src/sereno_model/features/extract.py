"""Base feature vector of a URL, from its text alone (feature set version 2).

Shared with the inference server: standard library and tldextract only, no network or
disk, URL never logged. Definitions and edge cases: docs/02-caracteristicas.md.
"""

import ipaddress
import math
from collections import Counter

from sereno_model.data.domains import split_host
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
    """The normalized URL without scheme, ``://``, host ``www.``, default port or end slashes."""
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
    found = split_host(host)
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

    Raises ``InvalidUrlError`` when ``url`` is not a usable http(s) URL.
    """
    return features_from_parts(parse_url(url))

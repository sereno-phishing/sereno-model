"""Conservative URL normalization shared by training de-duplication and the backend cache key.

Only the RFC 3986 section 6.2.2 safe equivalences, plus dropping the default port and the
fragment. What stays distinct: docs/decisiones.md.
"""

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_PORTS = {"http": "80", "https": "443"}
UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")

_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.\-]*):")
_TRIPLET = re.compile(r"%([0-9A-Fa-f]{2})")
_TRIPLET_OR_UPPERCASE = re.compile(r"%[0-9A-Fa-f]{2}|[A-Z]+")


class InvalidReason(StrEnum):
    EMPTY = "empty"
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    MISSING_HOST = "missing_host"
    WHITESPACE_OR_CONTROL = "whitespace_or_control"


class InvalidUrlError(ValueError):
    """The input cannot be used as an http(s) URL; ``reason`` says why."""

    def __init__(self, reason: InvalidReason, raw: str):
        super().__init__(f"{reason.value}: {raw!r}")
        self.reason = reason


@dataclass(frozen=True)
class UrlParts:
    """Components of a normalized URL; ``None`` means absent, not empty."""

    scheme: str
    userinfo: str | None
    host: str
    port: str | None
    path: str
    query: str | None

    @property
    def normalized(self) -> str:
        userinfo = "" if self.userinfo is None else f"{self.userinfo}@"
        port = "" if self.port is None else f":{self.port}"
        query = "" if self.query is None else f"?{self.query}"
        return f"{self.scheme}://{userinfo}{self.host}{port}{self.path}{query}"

    @property
    def has_path_or_query(self) -> bool:
        """False when the URL points at the bare site: empty path or ``/`` and no query."""
        return self.path not in ("", "/") or bool(self.query)


def _decode_unreserved(text: str) -> str:
    """Decode percent-encoded triplets of unreserved characters; keep any other one untouched."""

    def replace(match: re.Match) -> str:
        character = chr(int(match.group(1), 16))
        return character if character in UNRESERVED else match.group(0)

    return _TRIPLET.sub(replace, text)


def _lowercase_host(host: str) -> str:
    """Lowercase ASCII letters, skipping percent-encoded triplets so their hex case survives."""

    def replace(match: re.Match) -> str:
        text = match.group(0)
        return text if text.startswith("%") else text.lower()

    return _TRIPLET_OR_UPPERCASE.sub(replace, host)


def _split_host_port(hostport: str) -> tuple[str, str | None]:
    """Split a trailing ``:port`` off the host; an IPv6 literal keeps the colons it encloses."""
    literal_end = hostport.find("]") if hostport.startswith("[") else -1
    host, colon, port = hostport.rpartition(":")
    is_port = port == "" or (port.isascii() and port.isdigit())
    if colon and is_port and len(host) > literal_end:
        return host, port
    return hostport, None


def parse_url(raw: str) -> UrlParts:
    """Validate ``raw`` and return its normalized components."""
    text = raw.strip()
    if not text:
        raise InvalidUrlError(InvalidReason.EMPTY, raw)
    if any(char.isspace() or unicodedata.category(char) == "Cc" for char in text):
        raise InvalidUrlError(InvalidReason.WHITESPACE_OR_CONTROL, raw)

    match = _SCHEME.match(text)
    scheme = match.group(1).lower() if match else ""
    if scheme not in DEFAULT_PORTS:
        raise InvalidUrlError(InvalidReason.UNSUPPORTED_SCHEME, raw)

    rest = text[match.end() :]
    if not rest.startswith("//"):
        raise InvalidUrlError(InvalidReason.MISSING_HOST, raw)
    rest = rest[2:].split("#", 1)[0]

    ends = [index for index in (rest.find("/"), rest.find("?")) if index != -1]
    cut = min(ends, default=len(rest))
    authority, tail = rest[:cut], _decode_unreserved(rest[cut:])
    userinfo, at, hostport = authority.rpartition("@")
    host, port = _split_host_port(hostport)
    host = _lowercase_host(_decode_unreserved(host))
    if not host:
        raise InvalidUrlError(InvalidReason.MISSING_HOST, raw)

    path, question, query = tail.partition("?")
    return UrlParts(
        scheme=scheme,
        userinfo=_decode_unreserved(userinfo) if at else None,
        host=host,
        port=None if port == DEFAULT_PORTS[scheme] else port,
        path=path,
        query=query if question else None,
    )


def normalize_url(raw: str) -> str:
    """Return the normalized form of ``raw`` or raise :class:`InvalidUrlError`."""
    return parse_url(raw).normalized

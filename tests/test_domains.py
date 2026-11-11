import hashlib
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import tldextract

from sereno_model.data import domains
from sereno_model.data.domains import registered_domain, suffix_list_snapshot_id


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("example.com", "example.com"),
        ("tienda.com.pe", "tienda.com.pe"),
        ("pagos.tienda.com.pe", "tienda.com.pe"),
        ("shop.co.uk", "shop.co.uk"),
        ("a.b.shop.co.uk", "shop.co.uk"),
        ("login.secure.example.com", "example.com"),
    ],
)
def test_registered_domain_is_the_label_under_the_public_suffix(host, expected):
    assert registered_domain(host) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("www.example.com", "example.com"),
        ("www.tienda.com.pe", "tienda.com.pe"),
        ("www.localhost", "localhost"),
        ("www.192.168.10.5", "192.168.10.5"),
    ],
)
def test_leading_www_does_not_change_the_key(host, expected):
    assert registered_domain(host) == expected
    assert registered_domain(host.removeprefix("www.")) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("victim.pages.dev", "pages.dev"),
        ("another-victim.pages.dev", "pages.dev"),
        ("user.github.io", "github.io"),
        ("store.blogspot.com", "blogspot.com"),
    ],
)
def test_shared_platforms_fall_in_one_group_because_private_suffixes_are_ignored(host, expected):
    assert registered_domain(host) == expected


@pytest.mark.parametrize(
    "host",
    [
        "192.168.10.5",
        "localhost",
        "intranet",
        "host.notarealsuffix",
        "[2001:db8::1]",
        "com.pe",
    ],
)
def test_hosts_without_a_registered_domain_are_their_own_key(host):
    assert registered_domain(host) == host


def test_key_is_lowercase_whatever_the_case_of_the_host():
    assert registered_domain("WWW.Example.COM") == "example.com"
    assert registered_domain("LOCALHOST") == "localhost"
    assert registered_domain("Pagos.Tienda.Com.PE") == "tienda.com.pe"


def test_works_without_any_network_access(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the Public Suffix List must come from the bundled snapshot")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    domains._extractor.cache_clear()

    assert registered_domain("www.shop.co.uk") == "shop.co.uk"


def test_extractor_is_built_once():
    assert domains._extractor() is domains._extractor()


def test_snapshot_id_is_the_sha256_of_the_list_shipped_with_tldextract():
    shipped = Path(tldextract.__file__).parent / ".tld_set_snapshot"

    found = suffix_list_snapshot_id()

    assert found == hashlib.sha256(shipped.read_bytes()).hexdigest()
    assert found == suffix_list_snapshot_id()


def test_module_does_not_import_pandas():
    code = "import sys; import sereno_model.data.domains; sys.exit('pandas' in sys.modules)"

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0

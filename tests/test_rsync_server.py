"""Unit tests for the config layer and pure builders of mirror_plugin_rsync_server.

Tests cover load_rsync_config, validate_rsync_config, validate_users,
check_config_permissions, and all pure text builder functions using pytest
and SimpleNamespace mocks.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirror_plugin_rsync_server import (
    CONF_HEADER,
    check_config_permissions,
    build_global_section,
    build_private_module,
    build_public_module,
    build_rsyncd_conf,
    build_rsyncd_secrets,
    format_param,
    load_rsync_config,
    sanitize_comment_value,
    select_visible_packages,
    validate_module_name,
    validate_rsync_config,
    validate_users,
)


def valid_config() -> dict:
    """Return a representative valid rsync.json dict (all keys present)."""
    return {
        "rsyncd_conf": "/etc/rsyncd.conf",
        "secrets_file": "/mirror/etc/rsyncd.secrets",
        "users": {
            "kaist-mirror": "examplepassword",
        },
        "global": {
            "uid": "rsync",
            "gid": "nogroup",
            "max connections": 20,
        },
        "private_modules": {
            "enabled": True,
            "auth_users": "*",
            "list": False,
            "lock_file": "/var/run/rsyncd-private.lock",
        },
    }


# --- load_rsync_config tests ---

def test_load_rsync_config_missing_file_raises_valueerror(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no_such_file.json"
    with pytest.raises(ValueError, match="Cannot read rsync config"):
        load_rsync_config(nonexistent)


def test_load_rsync_config_invalid_json_raises_valueerror(tmp_path: Path) -> None:
    bad = tmp_path / "rsync.json"
    bad.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_rsync_config(bad)


def test_load_rsync_config_reads_and_normalizes(tmp_path: Path) -> None:
    minimal = {"rsyncd_conf": "/etc/rsyncd.conf", "secrets_file": "/etc/rsyncd.secrets"}
    config_file = tmp_path / "rsync.json"
    config_file.write_text(json.dumps(minimal), encoding="utf-8")

    result = load_rsync_config(config_file)

    assert result["users"] == {}
    assert result["global"] == {}
    pm = result["private_modules"]
    assert "enabled" in pm
    assert "auth_users" in pm
    assert "list" in pm
    assert "lock_file" in pm


# --- validate_rsync_config tests ---

def test_validate_rsync_config_rejects_unknown_top_level_key() -> None:
    cfg = valid_config()
    cfg["extra_unknown"] = "value"
    with pytest.raises(ValueError, match="Unknown key"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_relative_rsyncd_conf() -> None:
    cfg = valid_config()
    cfg["rsyncd_conf"] = "relative/path/rsyncd.conf"
    with pytest.raises(ValueError, match="absolute path"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_relative_secrets_file() -> None:
    cfg = valid_config()
    cfg["secrets_file"] = "relative/secrets"
    with pytest.raises(ValueError, match="absolute path"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_control_char_in_paths() -> None:
    cfg = valid_config()
    cfg["secrets_file"] = "/etc/rsyncd\nsecrets"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_secrets_file_key_in_global() -> None:
    cfg = valid_config()
    cfg["global"]["secrets file"] = "/some/path"
    with pytest.raises(ValueError, match="reserved"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_global_key_with_equals() -> None:
    cfg = valid_config()
    cfg["global"]["key=bad"] = "value"
    with pytest.raises(ValueError, match="'='"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_global_key_with_leading_bracket() -> None:
    cfg = valid_config()
    cfg["global"]["[section]"] = "value"
    with pytest.raises(ValueError, match="'\\['"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_newline_in_global_value() -> None:
    cfg = valid_config()
    cfg["global"]["uid"] = "rsync\nevil"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_bad_global_value_type() -> None:
    cfg = valid_config()
    cfg["global"]["uid"] = ["list", "value"]
    with pytest.raises(ValueError, match="must be str, int, or bool"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_preserves_global_insertion_order() -> None:
    cfg = {
        "rsyncd_conf": "/etc/rsyncd.conf",
        "secrets_file": "/etc/rsyncd.secrets",
        "global": {
            "zebra": "z",
            "alpha": "a",
            "middle": "m",
        },
    }
    result = validate_rsync_config(cfg)
    assert list(result["global"].keys()) == ["zebra", "alpha", "middle"]


def test_validate_rsync_config_rejects_unknown_private_modules_key() -> None:
    cfg = valid_config()
    cfg["private_modules"]["unknown_key"] = "value"
    with pytest.raises(ValueError, match="Unknown key in 'private_modules'"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_non_absolute_lock_file() -> None:
    cfg = valid_config()
    cfg["private_modules"]["lock_file"] = "relative/lock"
    with pytest.raises(ValueError, match="absolute path"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_control_char_in_global_key() -> None:
    cfg = valid_config()
    cfg["global"]["uid\x7fbad"] = "rsync"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_newline_in_rsyncd_conf() -> None:
    cfg = valid_config()
    cfg["rsyncd_conf"] = "/etc/rsyncd\nconf"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_del_control_char_in_secrets_file() -> None:
    cfg = valid_config()
    cfg["secrets_file"] = "/etc/rsyncd\x7fsecrets"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_c1_control_char_in_global_value() -> None:
    cfg = valid_config()
    cfg["global"]["uid"] = "rsync\x85evil"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_newline_in_lock_file() -> None:
    cfg = valid_config()
    cfg["private_modules"]["lock_file"] = "/var/run/rsyncd\nprivate.lock"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_control_char_in_lock_file() -> None:
    cfg = valid_config()
    cfg["private_modules"]["lock_file"] = "/var/run/rsyncd\x7fprivate.lock"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_newline_in_auth_users() -> None:
    cfg = valid_config()
    cfg["private_modules"]["auth_users"] = "*\nevil = injected"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_rejects_control_char_in_auth_users() -> None:
    cfg = valid_config()
    cfg["private_modules"]["auth_users"] = "*\x7f"
    with pytest.raises(ValueError, match="control character"):
        validate_rsync_config(cfg)


def test_validate_rsync_config_private_modules_defaults_merge() -> None:
    # Omit private_modules entirely - should get full defaults
    cfg = {"rsyncd_conf": "/etc/rsyncd.conf", "secrets_file": "/etc/rsyncd.secrets"}
    result = validate_rsync_config(cfg)
    pm = result["private_modules"]
    assert pm["enabled"] is True
    assert pm["auth_users"] == "*"
    assert pm["list"] is False
    assert pm["lock_file"] == "/var/run/rsyncd-private.lock"

    # Provide partial - should merge with defaults
    cfg2 = {
        "rsyncd_conf": "/etc/rsyncd.conf",
        "secrets_file": "/etc/rsyncd.secrets",
        "private_modules": {"auth_users": "trusted-mirror"},
    }
    result2 = validate_rsync_config(cfg2)
    pm2 = result2["private_modules"]
    assert pm2["auth_users"] == "trusted-mirror"
    assert pm2["enabled"] is True
    assert pm2["lock_file"] == "/var/run/rsyncd-private.lock"


# --- validate_users tests ---

def test_validate_users_accepts_valid() -> None:
    validate_users({"alice": "password123", "bob": "p:assword"})


def test_validate_users_rejects_empty_username() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_users({"": "password"})


def test_validate_users_rejects_colon_in_username() -> None:
    with pytest.raises(ValueError, match="':'"):
        validate_users({"user:name": "password"})


def test_validate_users_rejects_whitespace_in_username() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        validate_users({"user name": "password"})


def test_validate_users_rejects_leading_at() -> None:
    with pytest.raises(ValueError, match="'@'"):
        validate_users({"@user": "password"})


def test_validate_users_rejects_leading_hash() -> None:
    with pytest.raises(ValueError, match="'@'"):
        validate_users({"#user": "password"})


def test_validate_users_rejects_empty_password() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_users({"alice": ""})


def test_validate_users_rejects_newline_in_password() -> None:
    with pytest.raises(ValueError, match="control characters"):
        validate_users({"alice": "pass\nword"})


def test_validate_users_rejects_control_char_in_password() -> None:
    for bad_password in ("pass\tword", "pass\x7fword", "pass\x85word"):
        with pytest.raises(ValueError, match="control characters"):
            validate_users({"alice": bad_password})


def test_validate_users_error_never_contains_password(caplog: pytest.LogCaptureFixture) -> None:
    secret_password = "SuperSecret123!@#"
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as exc_info:
            validate_users({"user\twith\ttab": secret_password})

    assert secret_password not in str(exc_info.value)
    for record in caplog.records:
        assert secret_password not in record.getMessage()


# --- check_config_permissions tests ---

def test_check_config_permissions_warns_when_group_readable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_file = tmp_path / "rsync.json"
    config_file.write_text("{}", encoding="utf-8")

    os.chmod(config_file, 0o644)
    with caplog.at_level(logging.WARNING):
        check_config_permissions(config_file)

    assert any("chmod 600" in record.message for record in caplog.records)

    caplog.clear()
    os.chmod(config_file, 0o600)
    with caplog.at_level(logging.WARNING):
        check_config_permissions(config_file)

    assert not any("chmod 600" in record.message for record in caplog.records)


# --- Helper: package mock factory ---

def make_pkg(pkgid: str, name: str, dst: str, src: str = "", hidden: bool = False):
    """Return a SimpleNamespace package mock matching the mirror.py package shape."""
    return SimpleNamespace(
        pkgid=pkgid,
        name=name,
        settings=SimpleNamespace(hidden=hidden, src=src, dst=dst),
    )


def golden_config() -> dict:
    """Return the config dict used in the golden test."""
    return {
        "rsyncd_conf": "/etc/rsyncd.conf",
        "secrets_file": "/mirror/etc/rsyncd.secrets",
        "users": {},
        "global": {
            "uid": "rsync",
            "gid": "nogroup",
            "use chroot": "no",
            "max connections": 20,
            "motd file": "/mirror/etc/motd",
            "log file": "/var/log/geoul/rsyncd/all.log",
            "transfer logging": "yes",
            "log format": '%o %a - %u [%t] "%P/%f" - %l',
            "pid file": "/var/run/rsyncd.pid",
            "exclude": ".~tmp~",
        },
        "private_modules": {
            "enabled": True,
            "auth_users": "*",
            "list": False,
            "lock_file": "/var/run/rsyncd-private.lock",
        },
    }


# --- validate_module_name tests ---

@pytest.mark.parametrize("name,expected", [
    ("ArchLinux", True),
    ("debian", True),
    ("foo-bar", True),
    ("foo.bar", True),
    ("foo+bar", True),
    ("A1", True),
    ("a", True),
    # Invalid cases
    ("global", False),
    ("GLOBAL", False),
    ("Global", False),
    ("", False),
    (".hidden", False),
    ("has/slash", False),
    ("has]bracket", False),
    ("has space", False),
    ("has\nnewline", False),
    ("has\ttab", False),
])
def test_validate_module_name_matrix(name: str, expected: bool) -> None:
    assert validate_module_name(name) is expected


# --- sanitize_comment_value tests ---

def test_sanitize_comment_value_strips_newlines() -> None:
    # Control characters (including newline) are removed; remaining text
    # on both sides of the newline is concatenated. No newline remains, so
    # the value cannot inject a new rsyncd.conf line.
    dirty = "rsync://example.org/ftp\nmalicious = x"
    result = sanitize_comment_value(dirty)
    assert "\n" not in result
    assert result == "rsync://example.org/ftpmalicious = x"


def test_sanitize_comment_value_strips_surrounding_whitespace() -> None:
    assert sanitize_comment_value("  hello  ") == "hello"


def test_sanitize_comment_value_strips_control_chars() -> None:
    assert sanitize_comment_value("a\tb\rc") == "abc"


# --- format_param tests ---

def test_format_param_bool_lowercase() -> None:
    assert format_param("list", True) == "list = true"
    assert format_param("list", False) == "list = false"


def test_format_param_strips_control_chars() -> None:
    result = format_param("key", "val\nue")
    assert "\n" not in result
    assert result == "key = value"


def test_format_param_int_value() -> None:
    assert format_param("max connections", 20) == "max connections = 20"


def test_format_param_str_value() -> None:
    assert format_param("uid", "rsync") == "uid = rsync"


# --- build_global_section tests ---

def test_build_global_section_preserves_order() -> None:
    params = {"zebra": "z", "alpha": "a", "middle": "m"}
    result = build_global_section(params, "/etc/rsyncd.secrets")
    lines = result.splitlines()
    assert lines[0] == CONF_HEADER
    assert lines[1] == "zebra = z"
    assert lines[2] == "alpha = a"
    assert lines[3] == "middle = m"
    assert lines[4] == ""
    assert lines[5] == "secrets file = /etc/rsyncd.secrets"


def test_build_global_section_empty_params() -> None:
    result = build_global_section({}, "/etc/rsyncd.secrets")
    lines = result.splitlines()
    assert lines[0] == CONF_HEADER
    assert lines[1] == ""
    assert lines[2] == "secrets file = /etc/rsyncd.secrets"


# --- build_public_module tests ---

def test_build_public_module_omits_comment_when_src_empty() -> None:
    pkg = make_pkg("debian", "Debian", "/mirror/ftp/Debian", src="")
    result = build_public_module(pkg)
    assert "comment" not in result
    assert result == "[debian]\n    path = /mirror/ftp/Debian"


def test_build_public_module_includes_comment_when_src_set() -> None:
    pkg = make_pkg("p1", "ArchLinux", "/mirror/ftp/ArchLinux",
                   src="rsync://rsync.archlinux.org/ftp_tier1")
    result = build_public_module(pkg)
    assert "comment = from rsync://rsync.archlinux.org/ftp_tier1" in result


# --- build_private_module tests ---

def test_build_private_module_structure() -> None:
    pkg = make_pkg("archlinux", "ArchLinux", "/mirror/ftp/ArchLinux")
    pm = {
        "enabled": True,
        "auth_users": "*",
        "list": False,
        "lock_file": "/var/run/rsyncd-private.lock",
    }
    result = build_private_module(pkg, pm)
    lines = result.splitlines()
    assert lines[0] == "[.archlinux]"
    assert lines[1] == "    path = /mirror/ftp/ArchLinux"
    assert "private module for ArchLinux without connection limits" in lines[2]
    assert lines[3] == "    auth users = *"
    assert lines[4] == "    list = false"
    assert lines[5] == "    lock file = /var/run/rsyncd-private.lock"


# --- build_rsyncd_secrets tests ---

def test_build_rsyncd_secrets_roundtrip() -> None:
    users = {"u1": "p1", "u2": "p2"}
    result = build_rsyncd_secrets(users)
    assert result == "u1:p1\nu2:p2\n"


def test_build_rsyncd_secrets_empty_users_returns_empty() -> None:
    assert build_rsyncd_secrets({}) == ""


# --- select_visible_packages tests ---

def test_select_visible_packages_skips_empty_dst(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pkg = make_pkg("p1", "Debian", dst="")
    with caplog.at_level(logging.WARNING, logger="mirror"):
        result = select_visible_packages([pkg])
    assert result == []
    assert any("p1" in r.message for r in caplog.records)


def test_select_visible_packages_skips_relative_dst(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pkg = make_pkg("p1", "Debian", dst="relative/path")
    with caplog.at_level(logging.WARNING, logger="mirror"):
        result = select_visible_packages([pkg])
    assert result == []
    assert any("p1" in r.message for r in caplog.records)


def test_select_visible_packages_skips_control_char_dst(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pkg = make_pkg("p1", "Debian", dst="/mirror/ftp/De\nbian")
    with caplog.at_level(logging.WARNING, logger="mirror"):
        result = select_visible_packages([pkg])
    assert result == []
    assert any("p1" in r.message for r in caplog.records)


def test_select_visible_packages_skips_invalid_module_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    invalid_pkgids = [
        "has/slash",
        "has]bracket",
        "has space",
        ".leading_dot",
        "global",
        "GLOBAL",
    ]
    for pkgid in invalid_pkgids:
        pkgs = [make_pkg(pkgid, "Display Name", dst="/mirror/ftp/pkg")]
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="mirror"):
            result = select_visible_packages(pkgs)
        assert result == [], f"Expected {pkgid!r} to be skipped"
        assert any(pkgid in r.message for r in caplog.records), (
            f"Expected warning for {pkgid!r}"
        )


def test_select_visible_packages_dedup_case_insensitive(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # "Foo" sorts before "foo", so "Foo" is kept and "foo" is skipped.
    pkg_upper = make_pkg("Foo", "FooName", dst="/mirror/ftp/Foo")
    pkg_lower = make_pkg("foo", "fooName", dst="/mirror/ftp/foo")
    with caplog.at_level(logging.WARNING, logger="mirror"):
        result = select_visible_packages([pkg_upper, pkg_lower])
    assert len(result) == 1
    assert result[0].pkgid == "Foo"
    warning_messages = " ".join(r.message for r in caplog.records)
    assert "Foo" in warning_messages
    assert "foo" in warning_messages


def test_hidden_package_excluded_entirely(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pkg = make_pkg("p1", "ArchLinux", dst="/mirror/ftp/ArchLinux", hidden=True)
    with caplog.at_level(logging.WARNING, logger="mirror"):
        result = select_visible_packages([pkg])
    assert result == []
    assert caplog.records == []


# --- build_rsyncd_conf tests ---

GOLDEN_CONF = """\
# WARNING: DO NOT EDIT!! Generated by the mirror.py rsync-server plugin.
uid = rsync
gid = nogroup
use chroot = no
max connections = 20
motd file = /mirror/etc/motd
log file = /var/log/geoul/rsyncd/all.log
transfer logging = yes
log format = %o %a - %u [%t] "%P/%f" - %l
pid file = /var/run/rsyncd.pid
exclude = .~tmp~

secrets file = /mirror/etc/rsyncd.secrets

[archlinux]
    path = /mirror/ftp/ArchLinux
    comment = from rsync://rsync.archlinux.org/ftp_tier1

[.archlinux]
    path = /mirror/ftp/ArchLinux
    comment = private module for ArchLinux without connection limits for authorized mirrors
    auth users = *
    list = false
    lock file = /var/run/rsyncd-private.lock
"""


def test_build_rsyncd_conf_golden_matches_sample() -> None:
    config = golden_config()
    pkg = make_pkg(
        "archlinux",
        "ArchLinux",
        dst="/mirror/ftp/ArchLinux",
        src="rsync://rsync.archlinux.org/ftp_tier1",
    )
    result = build_rsyncd_conf([pkg], config)
    assert result == GOLDEN_CONF


def test_build_rsyncd_conf_idempotent() -> None:
    config = golden_config()
    pkg = make_pkg(
        "archlinux",
        "ArchLinux",
        dst="/mirror/ftp/ArchLinux",
        src="rsync://rsync.archlinux.org/ftp_tier1",
    )
    first = build_rsyncd_conf([pkg], config)
    second = build_rsyncd_conf([pkg], config)
    assert first == second


def test_build_rsyncd_conf_two_packages_sorted() -> None:
    config = golden_config()
    pkg_z = make_pkg("pz", "ZPackage", dst="/mirror/ftp/ZPackage", src="rsync://z.example.org/z")
    pkg_a = make_pkg("pa", "APackage", dst="/mirror/ftp/APackage", src="rsync://a.example.org/a")
    result = build_rsyncd_conf([pkg_z, pkg_a], config)
    lines = result.splitlines()
    # APackage must appear before ZPackage
    idx_a_pub = next(i for i, l in enumerate(lines) if l == "[pa]")
    idx_z_pub = next(i for i, l in enumerate(lines) if l == "[pz]")
    assert idx_a_pub < idx_z_pub
    # Private blocks must also be present and in order
    idx_a_priv = next(i for i, l in enumerate(lines) if l == "[.pa]")
    idx_z_priv = next(i for i, l in enumerate(lines) if l == "[.pz]")
    assert idx_a_priv < idx_z_priv
    # Blank line before each module section
    assert lines[idx_a_pub - 1] == ""
    assert lines[idx_a_priv - 1] == ""
    # Single trailing newline
    assert result.endswith("\n")
    assert not result.endswith("\n\n")


def test_private_modules_disabled_omits_private() -> None:
    config = golden_config()
    config["private_modules"]["enabled"] = False
    pkg = make_pkg("archlinux", "ArchLinux", dst="/mirror/ftp/ArchLinux")
    result = build_rsyncd_conf([pkg], config)
    assert "[.archlinux]" not in result
    assert "[archlinux]" in result


def test_private_modules_custom_auth_users_and_lock_file() -> None:
    config = golden_config()
    config["private_modules"]["auth_users"] = "trusted-mirror"
    config["private_modules"]["lock_file"] = "/var/run/custom-lock.lock"
    pkg = make_pkg("p1", "ArchLinux", dst="/mirror/ftp/ArchLinux")
    result = build_rsyncd_conf([pkg], config)
    assert "auth users = trusted-mirror" in result
    assert "lock file = /var/run/custom-lock.lock" in result


# --- Unit C: I/O layer tests ---

from mirror_plugin_rsync_server import (
    handle_regenerate_event,
    plugin,
    regenerate_rsync_files,
    setup,
    write_file_atomic,
)
import mirror
import mirror.config
import mirror.event
import mirror.plugin


def test_write_file_atomic_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "dir" / "file.txt"
    write_file_atomic(target, "hello", 0o644)
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello"


def test_write_file_atomic_sets_mode_0600(tmp_path: Path) -> None:
    target = tmp_path / "secrets"
    write_file_atomic(target, "secret", 0o600)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


def test_write_file_atomic_sets_mode_0644(tmp_path: Path) -> None:
    target = tmp_path / "rsyncd.conf"
    write_file_atomic(target, "content", 0o644)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o644


def test_write_file_atomic_skip_unchanged_returns_false(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    first = write_file_atomic(target, "same content", 0o644)
    assert first is True
    second = write_file_atomic(target, "same content", 0o644)
    assert second is False


def test_write_file_atomic_repairs_mode_on_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    write_file_atomic(target, "content", 0o644)
    result = write_file_atomic(target, "content", 0o600)
    assert result is False
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


def test_write_file_atomic_secrets_never_world_readable(tmp_path: Path) -> None:
    target = tmp_path / "rsyncd.secrets"
    write_file_atomic(target, "user:pass\n", 0o600)
    assert stat.S_IMODE(os.stat(target).st_mode) & 0o077 == 0


def _make_file_config(tmp_path: Path) -> dict:
    """Return a valid config dict with paths inside tmp_path."""
    return {
        "rsyncd_conf": str(tmp_path / "rsyncd.conf"),
        "secrets_file": str(tmp_path / "rsyncd.secrets"),
        "users": {"kaist-mirror": "examplepassword"},
        "global": {"uid": "rsync"},
        "private_modules": {
            "enabled": True,
            "auth_users": "*",
            "list": False,
            "lock_file": "/var/run/rsyncd-private.lock",
        },
    }


def test_regenerate_rsync_files_writes_both(tmp_path: Path) -> None:
    config = _make_file_config(tmp_path)
    pkg = make_pkg("archlinux", "ArchLinux", dst="/mirror/ftp/ArchLinux",
                   src="rsync://rsync.archlinux.org/ftp_tier1")
    regenerate_rsync_files([pkg], config)

    secrets_path = Path(config["secrets_file"])
    conf_path = Path(config["rsyncd_conf"])

    assert secrets_path.exists()
    assert conf_path.exists()
    assert stat.S_IMODE(os.stat(secrets_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(conf_path).st_mode) == 0o644
    assert "[archlinux]" in conf_path.read_text(encoding="utf-8")
    assert "kaist-mirror:examplepassword" in secrets_path.read_text(encoding="utf-8")


def test_regenerate_rsync_files_writes_secrets_before_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_file_config(tmp_path)
    pkg = make_pkg("archlinux", "ArchLinux", dst="/mirror/ftp/ArchLinux")

    call_order: list[str] = []
    original_write = write_file_atomic

    def recording_write(path: Path, content: str, mode: int) -> bool:
        call_order.append(path.name)
        return original_write(path, content, mode)

    import mirror_plugin_rsync_server as mod
    monkeypatch.setattr(mod, "write_file_atomic", recording_write)

    regenerate_rsync_files([pkg], config)

    secrets_name = Path(config["secrets_file"]).name
    conf_name = Path(config["rsyncd_conf"]).name
    assert call_order.index(secrets_name) < call_order.index(conf_name)


def test_regenerate_rsync_files_partial_failure_keeps_old_conf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_file_config(tmp_path)
    old_conf_content = "# old conf\n"
    conf_path = Path(config["rsyncd_conf"])
    conf_path.write_text(old_conf_content, encoding="utf-8")

    import mirror_plugin_rsync_server as mod

    def raise_on_secrets(*args, **kwargs):
        raise OSError("injected failure")

    monkeypatch.setattr(mod, "build_rsyncd_secrets", raise_on_secrets)

    with pytest.raises(OSError, match="injected failure"):
        regenerate_rsync_files([], config)

    assert conf_path.read_text(encoding="utf-8") == old_conf_content


def test_handle_regenerate_event_tolerates_arbitrary_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mirror.config, "CONFIG_PATH", config_path, raising=False)

    rsync_json = tmp_path / "rsync.json"
    rsync_cfg = {
        "rsyncd_conf": str(tmp_path / "rsyncd.conf"),
        "secrets_file": str(tmp_path / "rsyncd.secrets"),
        "users": {"user": "pass"},
        "global": {},
        "private_modules": {
            "enabled": True,
            "auth_users": "*",
            "list": False,
            "lock_file": "/var/run/rsyncd-private.lock",
        },
    }
    rsync_json.write_text(json.dumps(rsync_cfg), encoding="utf-8")

    pkg = make_pkg("archlinux", "ArchLinux", dst="/mirror/ftp/ArchLinux")
    monkeypatch.setattr(mirror, "packages", {"archlinux": pkg}, raising=False)

    handle_regenerate_event("x", 1, foo="bar")

    assert Path(rsync_cfg["rsyncd_conf"]).exists()
    assert Path(rsync_cfg["secrets_file"]).exists()


def test_handle_regenerate_event_missing_rsync_json_logs_and_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mirror.config, "CONFIG_PATH", config_path, raising=False)

    with caplog.at_level(logging.ERROR, logger="mirror"):
        handle_regenerate_event()

    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_handle_regenerate_event_no_packages_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mirror.config, "CONFIG_PATH", config_path, raising=False)

    rsync_json = tmp_path / "rsync.json"
    rsync_cfg = {
        "rsyncd_conf": str(tmp_path / "rsyncd.conf"),
        "secrets_file": str(tmp_path / "rsyncd.secrets"),
    }
    rsync_json.write_text(json.dumps(rsync_cfg), encoding="utf-8")

    monkeypatch.setattr(mirror, "packages", None, raising=False)

    with caplog.at_level(logging.WARNING, logger="mirror"):
        handle_regenerate_event()

    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_handle_regenerate_event_never_logs_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mirror.config, "CONFIG_PATH", config_path, raising=False)

    secret_password = "s3cr3t-PLAINTEXT"
    rsync_json = tmp_path / "rsync.json"
    rsync_cfg = {
        "rsyncd_conf": str(tmp_path / "rsyncd.conf"),
        "secrets_file": str(tmp_path / "rsyncd.secrets"),
        "users": {"kaist-mirror": secret_password},
        "global": {},
        "private_modules": {
            "enabled": True,
            "auth_users": "*",
            "list": False,
            "lock_file": "/var/run/rsyncd-private.lock",
        },
    }
    rsync_json.write_text(json.dumps(rsync_cfg), encoding="utf-8")

    pkg = make_pkg("archlinux", "ArchLinux", dst="/mirror/ftp/ArchLinux")
    monkeypatch.setattr(mirror, "packages", {"archlinux": pkg}, raising=False)

    with caplog.at_level(logging.DEBUG, logger="mirror"):
        handle_regenerate_event()

    for record in caplog.records:
        assert secret_password not in record.getMessage()


def test_setup_registers_both_event_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: list[tuple[str, object]] = []

    def recording_on(event_name: str, listener) -> None:
        registered.append((event_name, listener))

    monkeypatch.setattr(mirror.event, "on", recording_on)

    setup()

    assert ("MASTER.INIT.POST", handle_regenerate_event) in registered
    assert ("MASTER.CONFIG_RELOAD.POST", handle_regenerate_event) in registered


def test_plugin_returns_event_record() -> None:
    record = plugin()
    assert isinstance(record, mirror.plugin.PluginRecord)
    assert record.type == "event"
    assert record.name == "rsync-server"
    assert record.setup is setup

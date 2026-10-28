"""Destination recognition for the destination-aware egress rule (v0.5.9 P2-a).

The shipped network-client denylist hardened against known payload shapes
without seeing where a client connects. The exfiltration decision replaces
that blanket for the MODELED clients (``curl`` and ``wget``, off the shipped
denylist): recognition keys on destinations, not client names. A modeled
invocation resolves run-class exactly when every destination it names is
literal loopback and no destination-changing behavior gate fired; everything
else resolves ask-class (the ``network.egress`` dangerous operation, the same
class as git push), so interactive main sessions surface the approval dialog
and headless or child sessions deny with remediation.

Recognition is syntactic and fail-closed on every axis:

- Only literal-URL destinations are classified. A schemeless operand
  (``curl host:port``), an environment-variable indirection (variable
  expansion is already rejected by the shell grammar), a config-file or
  input-file indirection, or any other operand the recognizer cannot parse
  resolves as NON-loopback — never as loopback.
- Loopback is the literal name ``localhost`` (with or without the
  root-anchored trailing dot), the whole ``127.0.0.0/8`` range, ``::1``
  recognized through ``ipaddress`` normalization — so every spelling that
  normalizes to ``::1``, compressed or fully expanded (``0:0:0:0:0:0:0:1``
  included), is loopback — the ``inet_aton`` shorthand spellings whose every
  reading lands in the loopback range or on the unspecified address, and
  ``0.0.0.0`` (which Linux connects to loopback). Other IP literals, DNS
  names (including aliases of loopback), hex or out-of-range components, and
  multi-label names are not loopback.
- Redirects are not followed: destination-changing behavior resolves ask
  regardless of a literal loopback destination, in four classes — (a)
  redirect-following flags, (b) clients whose default follows redirects
  unless the invocation disables it, (c) destination-overriding or
  destination-indirecting options, (d) an invocation that disables curl's
  FTP PASV-IP protection (--no-ftp-skip-pasv-ip), which lets the server's
  PASV response choose the FTP data connection's address.

  A disclosed residual of the same destination-choosing class: wget's
  default passive FTP honors the address in the server's PASV response
  (wget's ftp_pasv overrides the control connection's address and has no
  --ftp-skip-pasv-ip equivalent), so a loopback wget FTP server can steer
  the data connection off loopback. Plain passive FTP stays run-class for
  both modeled clients — curl's default is safe, and class (d) gates only
  the disabling spelling — so the wget side is disclosed here rather than
  gated.
- A modeled invocation with NO recognizable literal-URL destination
  resolves ask: the recognizer cannot see where the client connects.

Documented limits, same class as every recognizer gap: an option the
per-client model does not know never silently reclassifies the
invocation. A value the model cannot account for — an unknown
``--option=value`` attached value, or the operand that follows an unknown
separately spelled option — surfaces as a destination candidate, so the
invocation resolves ask (fail-closed) unless that value is itself a
literal loopback URL. An unknown BOOLEAN long option (no value in either
spelling) is recorded raw and passes silently: the client itself rejects
an unknown option and fails before connecting, so no egress hides behind
it. The residual is a future option that changes the destination set
without naming it in a value — a new redirect-following flag, say — the
same class as every other recognizer gap. The
ambient (non-injected) environment is not inspected; the trust-gated
project environment is, proxy keys only, because its values are the ones
the subprocess actually receives, and so is the invocation's OWN
command-line environment (its command-prefix and env-wrapper assignments,
which override the injected values): it is inspected for the proxy keys
and for the clients' config-discovery variables — CURL_HOME (like
XDG_CONFIG_HOME and HOME) points curl at an uninspectable .curlrc,
WGETRC (like HOME) points wget at an uninspectable startup file, and a
config's proxy/output lines choose the destination and the writes, so
carrying one resolves ask. An ambient pre-existing ~/.curlrc or
WGETRC-named startup file is the user's own trust boundary, exactly like
the ambient environment. Unmodeled egress clients stay on the shipped
denylist (fail-closed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ipaddress
import re
from typing import Literal
from urllib.parse import urlparse

from chartreux.core.tools.builtins._shell_argv import (
    _command_name,
    _has_long,
    _has_short,
    _match_long,
    _OptionTable,
    _ParsedOptions,
    _scan_options,
)

# The modeled network clients: off the shipped denylist since v0.5.9 P2-a,
# classified by destination here. Every other egress-capable client stays
# denylisted until modeled.
NETWORK_CLIENTS = frozenset({"curl", "wget"})

# curl's value-taking options. The scanner consumes their operands (separate
# and attached spellings alike) so they never surface as positional
# destination candidates; an unmodeled option can only over-block, never hide
# an operand.
_CURL_TABLE = _OptionTable(
    long_values=frozenset({
        "--abstract-unix-socket",
        "--alt-svc",
        "--cacert",
        "--capath",
        "--cert",
        "--cert-type",
        "--config",
        "--connect-timeout",
        "--connect-to",
        "--continue-at",
        "--cookie",
        "--cookie-jar",
        "--crlfile",
        "--data",
        "--data-ascii",
        "--data-binary",
        "--data-raw",
        "--data-urlencode",
        "--dump-header",
        "--egd-file",
        "--engine",
        "--etag-compare",
        "--etag-save",
        "--form",
        "--form-string",
        "--ftp-account",
        "--ftp-alternative-to-user",
        "--ftp-port",
        "--header",
        "--hostpubmd5",
        "--hsts",
        "--interface",
        "--json",
        "--key",
        "--key-type",
        "--libcurl",
        "--limit-rate",
        "--local-port",
        "--mail-from",
        "--mail-rcpt",
        "--max-filesize",
        "--max-redirs",
        "--max-time",
        "--netrc-file",
        "--output",
        "--output-dir",
        "--pass",
        "--proto",
        "--proto-default",
        "--proto-redir",
        "--proxy",
        "--proxy-cacert",
        "--proxy-capath",
        "--proxy-cert",
        "--proxy-header",
        "--proxy-key",
        "--proxy-pass",
        "--proxy-user",
        "--quote",
        "--random-file",
        "--range",
        "--referer",
        "--request",
        "--resolve",
        "--retry",
        "--speed-limit",
        "--speed-time",
        "--telnet-option",
        "--time-cond",
        "--tlsauthtype",
        "--trace",
        "--trace-ascii",
        "--unix-socket",
        "--upload-file",
        "--url",
        "--user",
        "--user-agent",
        "--write-out",
    }),
    short_values=frozenset({
        "A",
        "b",
        "c",
        "C",
        "d",
        "D",
        "e",
        "E",
        "F",
        "H",
        "K",
        "m",
        "o",
        "P",
        "Q",
        "r",
        "t",
        "T",
        "u",
        "U",
        "w",
        "x",
        "X",
        "y",
        "Y",
        "z",
    }),
)

# wget's value-taking options, same consumption contract as curl's table.
_WGET_TABLE = _OptionTable(
    long_values=frozenset({
        "--append-output",
        "--backups",
        "--base",
        "--bind-address",
        "--body-data",
        "--body-file",
        "--ca-certificate",
        "--cache",
        "--certificate",
        "--config",
        "--connect-timeout",
        "--cut-dirs",
        "--default-page",
        "--directory-prefix",
        "--dns-timeout",
        "--domains",
        "--execute",
        "--exclude",
        "--exclude-directories",
        "--follow-tags",
        "--ftp-password",
        "--ftp-passwd",
        "--ftp-user",
        "--header",
        "--hsts-file",
        "--http-password",
        "--http-passwd",
        "--http-user",
        "--ignore-tags",
        "--include",
        "--include-directories",
        "--input-file",
        "--level",
        "--limit-rate",
        "--load-cookies",
        "--local-encoding",
        "--max-redirect",
        "--method",
        "--netrc-file",
        "--output-document",
        "--output-file",
        "--password",
        "--post-data",
        "--post-file",
        "--prefer-family",
        "--private-key",
        "--proxy",
        "--quota",
        "--read-timeout",
        "--referer",
        "--rejected-log",
        "--remote-encoding",
        "--save-cookies",
        "--timeout",
        "--tries",
        "--user",
        "--user-agent",
        "--warc-file",
    }),
    short_values=frozenset({
        "a",
        "B",
        "D",
        "e",
        "i",
        "I",
        "l",
        "O",
        "o",
        "P",
        "Q",
        "T",
        "t",
        "U",
        "X",
        "Y",
    }),
    optional_long_values=frozenset({"--restrict-file-names"}),
    # wget's boolean-mode options: recursive retrieval (and its mirror,
    # span-hosts, and page-requisites companions) changes the destination
    # set, so the classifier gates on them; the scanner must not consume a
    # URL as their operand. wget's -b is --background, also boolean: it
    # takes no operand, so it is absent from short_values.
    boolean_long=frozenset({
        "--mirror",
        "--page-requisites",
        "--recursive",
        "--span-hosts",
    }),
)
# The short letters whose presence in a cluster fires the recursion gate:
# -r/--recursive, -m/--mirror, -p/--page-requisites. The scan resolves
# wget's two-character -n composites first (see _WGET_N_COMPOSITES).
_WGET_RECURSION_SHORT = frozenset({"m", "p", "r"})
# wget's documented two-character short options: the -n prefix forms
# composite flags — -np is --no-parent, -nd --no-directories, -nc
# --no-clobber, -nv --no-verbose, -nH --no-host-directories — so a
# cluster like -np is ONE option, not -n plus -p: the 'p' of -np is
# never --page-requisites.
_WGET_N_COMPOSITES = frozenset({"np", "nd", "nc", "nv", "nH"})

# (a) Redirect-following flags: curl never follows by default, so the flag
# is what makes the destination unknowable. --follow (curl 8.16.0) is the
# spec-shaped variant of --location: it follows redirects too, so the same
# unknowable-destination gate applies.
_CURL_REDIRECT_LONG = frozenset({"--follow", "--location", "--location-trusted"})
_CURL_REDIRECT_SHORT = frozenset({"L"})

# (b) Clients whose DEFAULT behavior follows redirects: wget, unless the
# invocation disables it with --max-redirect=0.
_DEFAULT_REDIRECT_CLIENTS = frozenset({"wget"})
_WGET_REDIRECT_DISABLE = "--max-redirect"

# (c) Destination-overriding or destination-indirecting options: a URL
# naming loopback can connect anywhere behind --resolve/--connect-to, a
# config or input file can name destinations the recognizer cannot inspect,
# a unix(-abstract) socket IS the connection destination (the Docker daemon
# and every other local listener sit behind one), and the Alt-Svc cache can
# select an alternate connection destination the URL does not name.
_CURL_OVERRIDE_LONG = frozenset({
    "--abstract-unix-socket",
    "--alt-svc",
    "--config",
    "--connect-to",
    "--resolve",
    "--unix-socket",
})
_CURL_OVERRIDE_SHORT = frozenset({"K"})
_WGET_OVERRIDE_LONG = frozenset({"--config", "--execute", "--input-file"})
_WGET_OVERRIDE_SHORT = frozenset({"e", "i"})

# (d) The FTP PASV data-connection protection: curl's --ftp-skip-pasv-ip is
# enabled by default and forces the FTP data connection to reuse the
# control connection's address; its single documented disabling spelling
# (curl's long-option parser accepts exact spellings only) lets the server's
# PASV response choose the data-connection address.
_CURL_FTP_SKIP_PASV_IP_DISABLE = "--no-ftp-skip-pasv-ip"

# Options whose values are connection destinations (not merely headers or
# formats): a non-loopback proxy or --url value is egress even when every
# positional URL is loopback.
_CURL_DESTINATION_LONG = ("--proxy", "--url")
_CURL_DESTINATION_SHORT = ("-x",)
_WGET_DESTINATION_LONG = ("--base",)
# wget's -B is --base (destination-taking); -b is --background, a boolean
# with no operand, so only -B keys here.
_WGET_DESTINATION_SHORT = ("-B",)

# Options whose values name local files the client READS (extracted as
# path-inspected operands by the shell policy layer, so workspace and
# sensitive-file authority applies to them like any reader operand).
_CURL_READ_LONG = (
    "--cacert",
    "--capath",
    "--cert",
    "--config",
    "--cookie",
    "--crlfile",
    "--egd-file",
    "--etag-compare",
    "--key",
    "--netrc-file",
    "--random-file",
    "--upload-file",
)
_CURL_READ_SHORT = ("-b", "-E", "-K", "-T")
_WGET_READ_LONG = (
    "--body-file",
    "--ca-certificate",
    "--certificate",
    "--config",
    "--input-file",
    "--load-cookies",
    "--netrc-file",
    "--post-file",
    "--private-key",
)
_WGET_READ_SHORT = ("-i",)

# Options whose values name local files the client WRITES. The HSTS
# options are read AND written (the cache is loaded at startup and saved
# back at exit — curl's --hsts and wget's --hsts-file both say so), and
# --libcurl writes the libcurl-equivalent C source to its file, so all
# three join the write surface: path-inspected and metadata-guarded like
# every other write operand.
_CURL_WRITE_LONG = (
    "--alt-svc",
    "--cookie-jar",
    "--dump-header",
    "--etag-save",
    "--hsts",
    "--libcurl",
    "--output",
    "--output-dir",
    "--trace",
    "--trace-ascii",
)
_CURL_WRITE_SHORT = ("-c", "-D", "-o")
_WGET_WRITE_LONG = (
    "--append-output",
    "--directory-prefix",
    "--hsts-file",
    "--output-document",
    "--output-file",
    "--rejected-log",
    "--save-cookies",
    "--warc-file",
)
_WGET_WRITE_SHORT = ("-a", "-O", "-P", "-o")

# curl's data options read a named local file when the operand starts with
# '@' (-d @file, --data-binary @file); the '@' is not part of the path.
_CURL_DATA_LONG = (
    "--data",
    "--data-ascii",
    "--data-binary",
    "--data-raw",
    "--data-urlencode",
    "--json",
)
_CURL_DATA_SHORT = ("-d",)
# --data-urlencode additionally reads a file through curl's name@filename
# form, where the '@' sits mid-token after the field name.
_CURL_URLDATA_LONG = ("--data-urlencode",)

# curl's header options read headers from a file when the value starts with
# '@' (-H @file, --proxy-header @file); the '@' is not part of the path.
_CURL_HEADER_NAMES = ("--header", "--proxy-header", "-H")

# curl's form options read named local files through '@file' and '<file'
# content operands (-F name=@file).
_CURL_FORM_LONG = ("--form",)
_CURL_FORM_SHORT = ("-F",)

# curl's -O/--remote-name and wget's default download write the file the
# URL names: curl uses "only the file part of the remote file ... the path
# is cut off" and saves it in the working directory (or under
# --output-dir), and wget names the file after the URL's last path
# component — the --default-page value (index.html by default) for a URL
# that ends in a slash — under -P/--directory-prefix (the working
# directory by default). The destination is derivable at resolve time, so
# it joins the write surface exactly like an explicit -o/-O operand.
# curl's -J/--remote-header-name (the server-chosen Content-Disposition
# name) is NOT derivable before the run and stays an accepted scan limit.
# curl pairs its output options with its URLs one-for-one in command-line
# order (tool_getparam.c fills one shared node list: the k-th -o/--output
# or -O/--remote-name occurrence lands on the k-th URL — the man: "the
# first -o is for the first URL and so on"), so the derivation is per URL:
# a URL paired with -O writes its derived name, a URL paired with -o
# writes that file (or stdout for "-"), and only a URL with NO paired
# output takes the --remote-name-all default (a plain -O is itself a
# slot, not a default). The pairing is per OPERATION, not per invocation:
# --next (-:) starts a separate operation whose local options reset (the
# man: "--next resets all local options and only global ones have their
# values survive over to the operation following the --next instruction" —
# the global list is --fail-early, --libcurl, --parallel*, --progress-bar,
# --rate, --show-error, --stderr, --styled-output, --trace* and --verbose),
# so each operation pairs its own URLs with its own output slots, its own
# --remote-name-all default, and its own --output-dir: an earlier
# operation's directory never composes with a later operation's -o
# operand, and vice versa.
_CURL_REMOTE_NAME_LONG = ("--remote-name", "--remote-name-all")
_CURL_REMOTE_NAME_SHORT = frozenset({"O"})
_WGET_DEFAULT_PAGE_LONG = ("--default-page",)

# curl's operation separator, long and short spelling: --next (-:) starts a
# separate operation whose local options reset (see the pairing note
# above), so the download model is per operation, not per invocation.
_CURL_NEXT_TOKENS = frozenset({"--next", "-:"})

_URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


@dataclass(frozen=True)
class EgressClassification:
    """The egress verdict for one modeled network-client invocation.

    ``routine``: every recognized destination is literal loopback and no
    destination-changing behavior gate fired, so the invocation resolves
    run-class. ``ask``: the invocation resolves ask-class (the
    ``network.egress`` dangerous operation); ``reason`` names the behavior
    class or the offending destination.
    """

    kind: Literal["routine", "ask"]
    reason: str | None = None


def parse_literal_url_host(token: str) -> str | None:
    """The host of a literal-URL token, or None when it is not a literal URL.

    Only literal ``scheme://...`` operands are recognized. A schemeless
    destination, an unparseable URL, or a URL without a host returns None,
    and the caller resolves it as non-loopback (fail-closed, never loopback).
    """
    if not _URL_SCHEME.match(token):
        return None
    try:
        host = urlparse(token).hostname
    except ValueError:
        return None
    return host.casefold() if host else None


# The inet_aton shorthand spellings: 1-4 dotted numeric components.
_MIN_ATON_PARTS = 1
_MAX_ATON_PARTS = 4


def is_loopback_host(host: str) -> bool:
    """Whether a URL host is a supported loopback destination.

    The supported forms: the literal name ``localhost`` (with or without the
    root-anchored trailing dot, which bypasses search-domain suffixing), the
    whole ``127.0.0.0/8`` range, and ``::1`` — recognized through
    ``ipaddress`` normalization, so every spelling that normalizes to ``::1``
    (compressed or fully expanded, ``0:0:0:0:0:0:0:1`` included) is loopback —
    plus the ``inet_aton`` shorthand spellings whose every reading (plain
    decimal and octal-aware) parses in range and lands in the loopback range
    or on the unspecified address ``0.0.0.0``, which Linux connects to
    loopback. Everything else — other IP literals, DNS names including
    aliases of loopback, hex or out-of-range components, and multi-label
    names — is not loopback.
    """
    if host in {"localhost", "localhost."}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return _loopback_inet_aton_shorthand(host)
    return address.is_loopback or address == ipaddress.IPv4Address("0.0.0.0")


def _loopback_inet_aton_shorthand(host: str) -> bool:
    """Whether an ``inet_aton`` shorthand spelling is unambiguously loopback.

    A dotted spelling of 1-4 numeric components is recognized only when
    every reading — plain decimal and octal-aware, the two readings
    ``inet_aton`` accepts — parses in range and lands in the loopback range
    or on the unspecified address (which Linux connects to loopback). A
    component whose leading zero makes the octal and decimal readings
    disagree keeps the whole spelling unrecognized (fail-closed).
    """
    parts = host.split(".")
    # ASCII digits only: str.isdigit() alone accepts Unicode digit
    # characters — superscripts crash int() and fullwidth digits parse,
    # which would misclassify a host the client sends to DNS as loopback.
    # Any component that is not plain ASCII digits, or that fails either
    # reading's conversion (an invalid octal component like "08" fails
    # inet_aton too, so the host goes to DNS), keeps the whole spelling
    # unrecognized: fail-closed, never loopback.
    if not _MIN_ATON_PARTS <= len(parts) <= _MAX_ATON_PARTS or not all(
        part.isascii() and part.isdigit() for part in parts
    ):
        return False
    count = len(parts)
    last_bits = 8 * (5 - count)
    for octal_aware in (False, True):
        value = 0
        for index, part in enumerate(parts):
            try:
                if octal_aware and len(part) > 1 and part[:1] == "0":
                    component = int(part, 8)
                else:
                    component = int(part, 10)
            except ValueError:
                return False
            if index == count - 1:
                if component >= 1 << last_bits:
                    return False
                value = (value << last_bits) | component
            else:
                if component >= 1 << 8:
                    return False
                value = (value << 8) | component
        address = ipaddress.IPv4Address(value)
        if not (address.is_loopback or address == ipaddress.IPv4Address("0.0.0.0")):
            return False
    return True


# The proxy-family environment keys the modeled clients honor for their
# connection destination. Matched case-insensitively, which covers the
# uppercase spellings too; the set is the union of what curl and wget honor
# (wget's manpage documents http_proxy, https_proxy, and ftp_proxy) plus the
# SOCKS family curl honors.
_PROXY_ENV_KEYS = frozenset({
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "ftps_proxy",
    "socks_proxy",
    "socks4_proxy",
    "socks5_proxy",
})

# The config-discovery variables of the modeled clients' implicit startup
# config: curl reads "$CURL_HOME/.curlrc" first, then
# "$XDG_CONFIG_HOME/curlrc", then "$HOME/.curlrc"; wget reads SYSTEM_WGETRC
# for its system config and WGETRC (or "$HOME/.wgetrc") for its user config.
# A config file can name destinations
# (proxy =) and write outputs (output =) the recognizer cannot inspect, so
# the invocation's OWN command-line environment carrying one resolves ask.
# The spellings the policy layer checks outright (command-prefix and
# env-wrapper assignments) are denied as dangerous environment assignments;
# this gate covers the spellings that reach the classifier un-denied, the
# modeled sudo -u wrapper's NAME=value operands among them. The injected
# project environment is the user's own trust boundary and is inspected
# for proxy keys only.
_CURL_CONFIG_ENV_KEYS = frozenset({"CURL_HOME", "XDG_CONFIG_HOME", "HOME"})
_WGET_CONFIG_ENV_KEYS = frozenset({"WGETRC", "SYSTEM_WGETRC", "HOME"})


def _proxy_value_loopback(value: str) -> bool:
    """Whether a proxy URL's destination is loopback.

    A schemeless proxy value (``localhost:3128``) is an http proxy to the
    clients, so it is parsed with the scheme prepended. A value that fails
    to parse is non-loopback (fail-closed).
    """
    if not _URL_SCHEME.match(value):
        value = f"http://{value}"
    host = parse_literal_url_host(value)
    return host is not None and is_loopback_host(host)


def _non_loopback_proxy_key(env: Mapping[str, str]) -> str | None:
    """The proxy-family key whose destination is not loopback.

    A loopback-classified invocation that runs with a non-loopback proxy
    egresses through that proxy, so such a key makes the destination
    effectively unestablishable and the invocation resolves ask.
    Case-insensitive key matching covers the uppercase spellings too.
    """
    for key, value in env.items():
        if (
            key.casefold() in _PROXY_ENV_KEYS
            and value
            and not _proxy_value_loopback(value)
        ):
            return key
    return None


def _client_table(client: str) -> _OptionTable:
    return _WGET_TABLE if client == "wget" else _CURL_TABLE


def _curl_operation_args(args: list[str]) -> list[str]:
    """Expose clustered -: separators; curl ignores their remaining suffix."""
    normalized: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            normalized.extend(args[index:])
            break
        consume_next = False
        if token.startswith("--"):
            option = _match_long(token, _CURL_TABLE.long_values)
            consume_next = option is not None and "=" not in token
        elif token.startswith("-"):
            for offset, short in enumerate(token[1:], start=1):
                if short == ":":
                    if offset > 1:
                        normalized.append(token[:offset])
                    token = "-:"
                    break
                if short in _CURL_TABLE.short_values:
                    consume_next = offset == len(token) - 1
                    break
        normalized.append(token)
        index += 1
        if consume_next and index < len(args):
            normalized.append(args[index])
            index += 1
    return normalized


def _destination_candidates(client: str, args: list[str]) -> list[str]:
    """Every operand of the invocation that names a connection destination.

    Positional operands are always candidates (curl and wget treat a
    schemeless positional as a host, so an unparseable one must resolve
    ask). Values of the destination-carrying options are candidates. An
    UNKNOWN option's attached value (``--opt=value``) is a candidate too:
    the recognizer cannot prove it names no destination, so it fails
    closed. Values of known non-destination options (a header, a format
    string, an output file) are not candidates.
    """
    table = _client_table(client)
    parsed = _scan_options(args, table)
    candidates = list(parsed.positionals)
    destination_names = (
        (*_CURL_DESTINATION_LONG, *_CURL_DESTINATION_SHORT)
        if client == "curl"
        else (*_WGET_DESTINATION_LONG, *_WGET_DESTINATION_SHORT)
    )
    candidates.extend(parsed.values_for(*destination_names))
    known_long = table.long_values | table.optional_long_values | table.boolean_long
    for token in parsed.options:
        if not token.startswith("--") or "=" not in token:
            continue
        name = token.partition("=")[0]
        if name not in known_long:
            candidates.append(token.partition("=")[2])
    return candidates


def _wget_recursion_gate_fired(parsed: _ParsedOptions, table: _OptionTable) -> bool:
    """Whether a wget short-option cluster carries a recursion-gate letter.

    wget's documented two-character short options (-np, -nd, -nc, -nv,
    -nH) are atomic: the cluster scan resolves them before testing the
    single letters, so -np's 'p' is --no-parent's, never --page-
    requisites. A value-taking short still ends the cluster (the rest of
    the token is its attached operand, not more options).
    """
    value_shorts = table.short_values | table.optional_short_values
    for token in parsed.options:
        if not token.startswith("-") or token.startswith("--"):
            continue
        chars = token[1:]
        index = 0
        while index < len(chars):
            if chars[index : index + 2] in _WGET_N_COMPOSITES:
                index += 2
                continue
            short = chars[index]
            if short in _WGET_RECURSION_SHORT:
                return True
            if short in value_shorts:
                break
            index += 1
    return False


def classify_network_egress(  # noqa: PLR0911, PLR0912
    tokens: list[str],
    injected_env: Mapping[str, str] | None = None,
    command_env: Mapping[str, str] | None = None,
) -> EgressClassification:
    """Classify one command's argv when it runs a modeled network client.

    Non-client commands classify ``routine``: the egress rule keys on the
    modeled clients only, and every other egress-capable client stays on the
    shipped denylist. The destination-changing behavior classes
    resolve ask regardless of a literal loopback destination; every
    destination candidate must parse as a literal URL with a loopback host;
    a missing or unparseable destination resolves ask (never loopback).

    ``injected_env`` is the trust-gated project environment the bash call's
    subprocess actually receives. ``command_env`` is the invocation's OWN
    command-line environment: the command-prefix assignments and the
    ``env``-wrapper assignments, which the shell applies to this subprocess
    over the inherited environment. Command-line assignments override the
    injected ones (POSIX prefix semantics), so the proxy inspection runs on
    their merge. When the effective environment carries a proxy-family key
    whose value is not a loopback proxy, the loopback destination is
    effectively unestablishable — the connection egresses through that
    proxy — and the invocation resolves ask. Loopback proxies keep the
    current classification. ``command_env`` is also inspected for the
    clients' config-discovery variables (curl's CURL_HOME chain, wget's
    WGETRC): carrying one points the client at an uninspectable startup
    config whose proxy/output lines choose the destination and the writes,
    so the invocation resolves ask. The injected environment is the user's
    own trust boundary and is inspected for proxy keys only.
    """
    if not tokens:
        return EgressClassification("routine")
    client = _command_name(tokens[0])
    if client not in NETWORK_CLIENTS:
        return EgressClassification("routine")
    args = [str(token) for token in tokens[1:]]
    if client == "curl":
        args = _curl_operation_args(args)
    table = _client_table(client)
    parsed = _scan_options(args, table)

    # (a) Redirect-following flags.
    if client == "curl" and (
        _has_long(parsed, *_CURL_REDIRECT_LONG)
        or _has_short(parsed, _CURL_REDIRECT_SHORT, table)
    ):
        return EgressClassification(
            "ask", "the invocation follows redirects (curl -L/--location)"
        )
    # (a2) Recursive retrieval: wget follows the PAGE LINKS of whatever the
    # loopback server serves, so the destination set is not the invoked URL
    # but everything that page references (arbitrary hosts under
    # -H/--span-hosts, same-host content otherwise — still a page the
    # loopback server controls). -p/--page-requisites implies the same
    # retrieval of a page's images and stylesheets. The whole class
    # resolves ask. The short-cluster scan resolves wget's documented
    # two-character -n composites first, so -np (--no-parent) never
    # false-trips the -p gate.
    if client == "wget" and (
        _has_long(parsed, "--mirror", "--page-requisites", "--recursive")
        or _wget_recursion_gate_fired(parsed, table)
    ):
        return EgressClassification(
            "ask",
            "wget -r/--recursive (or -p/--page-requisites) follows page "
            "links to destinations the invocation does not name",
        )
    # (b) Clients whose default follows redirects, unless disabled: the
    # disable holds only when NO redirect-enabling spelling appears, so a
    # later --max-redirect with a non-zero value re-arms the gate.
    if client in _DEFAULT_REDIRECT_CLIENTS:
        redirect_values = parsed.values_for(_WGET_REDIRECT_DISABLE)
        disabled = bool(redirect_values) and all(
            value == "0" for value in redirect_values
        )
        if not disabled:
            return EgressClassification(
                "ask",
                "wget follows redirects by default; disable them with --max-redirect=0",
            )
    # (c) Destination-overriding or destination-indirecting options.
    override_long, override_short = (
        (_CURL_OVERRIDE_LONG, _CURL_OVERRIDE_SHORT)
        if client == "curl"
        else (_WGET_OVERRIDE_LONG, _WGET_OVERRIDE_SHORT)
    )
    if _has_long(parsed, *override_long) or _has_short(parsed, override_short, table):
        return EgressClassification(
            "ask", "the invocation carries a destination-overriding option"
        )
    # (d) FTP PASV data-connection protection disabled: curl's
    # --ftp-skip-pasv-ip is enabled by default (added in 7.74.0) and forces
    # the FTP data connection to reuse the control connection's address, so
    # a loopback control connection keeps the data connection on loopback.
    # Its one documented disabling spelling lets the server's PASV response
    # choose the data-connection address, so the transfer can egress to a
    # server-chosen non-loopback address the invocation never names.
    # --disable-epsv alone is not a trigger (an EPSV response carries no
    # address, and PASV under the default still skips it), and --ftp-port
    # is not either (active mode: the client picks the data address and the
    # server connects inbound). wget has no equivalent gate: --no-passive-ftp
    # switches to that same active mode (inbound data connection, no client
    # egress); wget's DEFAULT passive FTP honors the server's PASV address
    # and is disclosed in the module docstring rather than gated.
    if client == "curl" and _has_long(parsed, _CURL_FTP_SKIP_PASV_IP_DISABLE):
        return EgressClassification(
            "ask",
            "the invocation disables curl's FTP PASV IP protection "
            "(--no-ftp-skip-pasv-ip), so the server's PASV response "
            "chooses the FTP data connection's address",
        )

    candidates = _destination_candidates(client, args)
    if not candidates:
        return EgressClassification("ask", "no literal-URL destination is recognizable")
    for candidate in candidates:
        host = parse_literal_url_host(candidate)
        if host is None:
            return EgressClassification(
                "ask",
                f"the destination '{candidate}' is not a recognizable literal URL",
            )
        if not is_loopback_host(host):
            return EgressClassification(
                "ask", f"the destination '{candidate}' is not loopback"
            )
    # The invocation's effective environment: the trust-gated injected
    # project environment overridden by the command line's OWN assignments
    # (POSIX prefix semantics). Any proxy-family key on a non-loopback
    # proxy makes the loopback destination unestablishable.
    effective_env: dict[str, str] = dict(injected_env or {})
    effective_env.update(command_env or {})
    if effective_env:
        proxy_key = _non_loopback_proxy_key(effective_env)
        if proxy_key is not None:
            return EgressClassification(
                "ask",
                f"the invocation's environment sets {proxy_key} to a "
                "non-loopback proxy",
            )
    # The invocation's OWN environment carrying a config-discovery variable
    # points the client at an uninspectable startup config (curl's
    # .curlrc chain, wget's WGETRC startup file), whose proxy/output
    # lines choose the destination and the writes, so the invocation
    # resolves ask. The injected project environment is the user's own
    # trust boundary and is inspected for proxy keys only.
    config_keys = _CURL_CONFIG_ENV_KEYS if client == "curl" else _WGET_CONFIG_ENV_KEYS
    for key in command_env or {}:
        if key in config_keys:
            return EgressClassification(
                "ask",
                f"the invocation's environment sets {key}, pointing the "
                "client at an uninspectable startup config file",
            )
    return EgressClassification("routine")


def _file_read_operands(value: str) -> list[str]:
    """Local files a data option's operand reads, if any (``-d @file``)."""
    if value.startswith("@") or value.startswith("<"):
        return [value[1:]]
    return []


def _urlencode_read_operands(value: str) -> list[str]:
    """Local files a ``--data-urlencode`` operand reads, if any.

    curl's documented forms include ``name@filename``, where the ``@`` sits
    mid-token after the field name (``payload@.env`` reads ``.env``); a
    ``name=content`` form is literal data and reads nothing. The leading
    ``@``/``<`` forms are handled by :func:`_file_read_operands` as for the
    other data options.
    """
    at = value.find("@")
    equals = value.find("=")
    if at > 0 and (equals == -1 or at < equals):
        path = value[at + 1 :].strip().strip("\"'")
        if path:
            return [path]
    return []


def _header_read_operands(value: str) -> list[str]:
    """Local files a header option's operand reads, if any (``-H @file``)."""
    if value.startswith("@"):
        path = value[1:].strip()
        if path:
            return [path]
    return []


def _form_read_operands(value: str) -> list[str]:
    """Local files a curl form operand reads (``name=@file`` / ``name=<file``).

    curl's documented form syntax separates parameters with ``;`` after the
    ``name=content`` prefix, and the content part reads a file when it starts
    with ``@`` (upload) or ``<`` (text field). An ``@`` content part is a
    COMMA-SEPARATED file list — tool_formparse.c documents
    ``name=@filename,filename2,filename3`` and loops on every comma — and
    every comma piece may carry its own ``;``-parameters, so the pieces are
    split before the parameters. Among the parameters only
    ``headers=@file``/``headers=<file`` read one (``headers="..."`` is a
    literal) — ``type=``, ``filename=`` and ``encoder=`` name literals. The
    content part itself never parses as a parameter, so
    ``-F 'x=headers=@f'`` sends the literal string and reads nothing.
    """
    _, separator, content = value.partition("=")
    if not separator:
        return []
    reads: list[str] = []
    # An '@' content part is a comma-separated file list; the '<' and
    # literal forms take a single word (tool_formparse.c loops on ',' only
    # in the '@' branch).
    file_list = content.startswith("@")
    pieces = content.split(",") if file_list else (content,)
    for piece_index, piece in enumerate(pieces):
        for index, part in enumerate(piece.split(";")):
            part = part.strip()
            if index == 0:
                # The file word: '@file' on the list's first piece, a bare
                # 'file' on the later ones, '<file' for the text-field form.
                if part.startswith("@"):
                    path = part[1:].strip().strip("\"'")
                elif part.startswith("<"):
                    path = part[1:].strip().strip("\"'")
                elif file_list and piece_index > 0:
                    path = part.strip("\"'")
                else:
                    path = ""
                if path:
                    reads.append(path)
            elif part.startswith("headers="):
                header_value = part[len("headers=") :].strip()
                if header_value.startswith(("@", "<")):
                    path = header_value[1:].strip().strip("\"'")
                    if path:
                        reads.append(path)
    return reads


def _url_path(url: str) -> str | None:
    """The URL's path, or None when the operand is not a literal URL."""
    if parse_literal_url_host(url) is None:
        return None
    try:
        return urlparse(url).path
    except ValueError:
        return None


def _scan_ordered(args: list[str], table: _OptionTable) -> list[tuple[str, str, str]]:
    """The scanner's token classification in command-line order.

    A mirror of :func:`_scan_options` that also records WHERE each event
    happened: ``("option", token, "")`` for every option token,
    ``("value", option, value)`` for every consumed operand, and
    ``("positional", token, "")`` for every positional. Reconstructing the
    three scanner outputs from the events reproduces ``_ParsedOptions``
    exactly, which the per-URL download collection relies on: curl fills
    its URL list and its output-option list from one shared node list in
    command-line order, so the pairing needs the interleaved order the
    scanner's split outputs lose.
    """
    events: list[tuple[str, str, str]] = []
    all_long = table.long_values | table.optional_long_values | table.boolean_long
    index = 0
    options_ended = False
    while index < len(args):
        token = args[index]
        if options_ended:
            events.append(("positional", token, ""))
            index += 1
            continue
        if token == "--":
            options_ended = True
            index += 1
            continue
        if token.startswith("--"):
            option = _match_long(token, all_long)
            events.append(("option", option or token, ""))
            if option is None:
                index += 1
                continue
            _, separator, attached = token.partition("=")
            if separator:
                events.append(("value", option, attached))
                index += 1
            elif (
                option in table.long_values
                or (
                    option in table.optional_long_values
                    and table.optional_long_values_consume_next
                )
            ) and index + 1 < len(args):
                events.append(("value", option, args[index + 1]))
                index += 2
            else:
                index += 1
            continue
        if token.startswith("-") and token != "-":
            events.append(("option", token, ""))
            chars = token[1:]
            for offset, short in enumerate(chars):
                if short not in table.short_values | table.optional_short_values:
                    continue
                attached = chars[offset + 1 :]
                if attached:
                    events.append(("value", f"-{short}", attached))
                elif short in table.short_values and index + 1 < len(args):
                    events.append(("value", f"-{short}", args[index + 1]))
                    index += 1
                break
            index += 1
            continue
        events.append(("positional", token, ""))
        index += 1
    return events


@dataclass(frozen=True)
class _DownloadOperation:
    """One download model's URL/output pairing and output directory.

    curl's model is one entry per ``--next``-separated operation: ``urls``
    is the operation's URL operands in command-line order (positionals and
    ``--url`` values alike — curl adds a positional exactly as a ``--url``),
    ``outputs`` the operation's ordered output-option stream (one
    ``("file", name)`` per ``-o``/``--output`` occurrence and one
    ``("remote", "")`` per ``-O``/``--remote-name`` occurrence), and
    ``remote_name_all`` whether the operation's ``--remote-name-all`` arms
    the default for its URLs without a paired output. ``output_dir`` is the
    operation's LAST ``--output-dir`` value ("" when it names none): the
    option is local and resets at ``--next``, so one operation's directory
    never composes with another operation's ``-o`` operand. wget has no
    ``--next``, so its model is a single invocation-wide entry.
    """

    urls: tuple[str, ...]
    outputs: tuple[tuple[str, str], ...]
    remote_name_all: bool
    output_dir: str


def _curl_download_streams(
    args: list[str], table: _OptionTable, parsed: _ParsedOptions
) -> list[_DownloadOperation] | None:
    """curl's per-operation download model: one entry per ``--next`` block.

    Each ``--next`` (or ``-:``) starts a new operation whose local options
    reset, so the URL list, the output-option stream, the
    ``--remote-name-all`` default, and the ``--output-dir`` value are all
    collected per block: the first block ends at the first ``--next``, and
    every later block at the next one (or at the end of the argv).

    Returns None when the ordered mirror disagrees with the scanner (it
    cannot for a faithful mirror), so the caller falls back to the
    scanner's own split outputs rather than trust a divergent walk.
    """
    events = _scan_ordered(args, table)
    if (
        [event[1] for event in events if event[0] == "option"] != list(parsed.options)
        or [(event[1], event[2]) for event in events if event[0] == "value"]
        != list(parsed.values)
        or [event[1] for event in events if event[0] == "positional"]
        != list(parsed.positionals)
    ):
        return None
    operations: list[_DownloadOperation] = []
    urls: list[str] = []
    outputs: list[tuple[str, str]] = []
    remote_name_all = False
    output_dir = ""
    for kind, first, second in events:
        if kind == "positional":
            urls.append(first)
        elif kind == "value":
            if first == "--url":
                urls.append(second)
            elif first in {"--output", "-o"}:
                outputs.append(("file", second))
            elif first == "--output-dir":
                output_dir = second
        elif first in _CURL_NEXT_TOKENS:
            operations.append(
                _DownloadOperation(
                    tuple(urls), tuple(outputs), remote_name_all, output_dir
                )
            )
            urls = []
            outputs = []
            remote_name_all = False
            output_dir = ""
        elif first == "--remote-name-all":
            remote_name_all = True
        elif first == "--remote-name":
            outputs.append(("remote", ""))
        elif first.startswith("-") and not first.startswith("--"):
            # A short cluster: every 'O' is a remote-name slot; the first
            # value-taking short ends the cluster (the rest of the token is
            # its attached operand, not more options).
            for short in first[1:]:
                if short == "O":
                    outputs.append(("remote", ""))
                    continue
                if short in table.short_values | table.optional_short_values:
                    break
    operations.append(
        _DownloadOperation(tuple(urls), tuple(outputs), remote_name_all, output_dir)
    )
    return operations


def _download_write_operands(  # noqa: PLR0912
    client: str, args: list[str], parsed: _ParsedOptions
) -> list[str]:
    """The local files a modeled client's download writes by derivation.

    The write is the URL's file part saved in the working directory —
    curl under -O/--remote-name (a URL with no file part makes curl fail
    before any write, "Remote file name has no length", so nothing is
    derived for it), wget by default (a URL that ends in a slash names
    --default-page's value, index.html by default). An explicit
    -o/--output (curl) or -O/--output-document (wget) operand is already
    the collected write, so no download name is derived alongside it.
    --output-dir (curl) and -P/--directory-prefix (wget) name the
    directory the derived file lands in.

    curl pairs its output options with its URLs one-for-one in
    command-line order (the man: "the first -o is for the first URL and so
    on"), so the derivation is per URL: the k-th URL writes the k-th
    output option's file, and only a URL with no paired output takes the
    --remote-name-all default. A paired ``-o -`` writes to stdout (no
    file), and so does a URL with no paired output and no
    --remote-name-all. The pairing is per OPERATION (curl's --next resets
    the local options), so each operation pairs its own URLs with its own
    output slots, its own --remote-name-all default, and its own
    --output-dir. wget's -O/--output-document concatenates every
    document into its one file, so it suppresses the derived names for the
    whole invocation.
    """
    table = _client_table(client)
    default_page: tuple[str, ...] = ()
    operations: list[_DownloadOperation]
    if client == "curl":
        streams = _curl_download_streams(args, table, parsed)
        if streams is None:
            # The ordered mirror diverged from the scanner (it cannot for a
            # faithful mirror): fall back to the presence-based model,
            # which over-collects rather than under-collects.
            directory = parsed.values_for("--output-dir")
            operations = [
                _DownloadOperation(
                    (*parsed.positionals, *parsed.values_for("--url")),
                    (),
                    _has_long(parsed, *_CURL_REMOTE_NAME_LONG)
                    or _has_short(parsed, _CURL_REMOTE_NAME_SHORT, table),
                    directory[-1] if directory else "",
                )
            ]
        else:
            operations = streams
    else:
        if parsed.values_for("--output-document", "-O"):
            return []
        directory = parsed.values_for("--directory-prefix", "-P")
        default_page = parsed.values_for(*_WGET_DEFAULT_PAGE_LONG)
        operations = [
            _DownloadOperation(
                parsed.positionals, (), False, directory[-1] if directory else ""
            )
        ]
    names: list[str] = []
    for operation in operations:
        for index, url in enumerate(operation.urls):
            if client == "curl":
                if index < len(operation.outputs):
                    kind, _value = operation.outputs[index]
                    if kind != "remote":
                        # A paired -o names the file itself (already collected
                        # as a raw write operand); "-o -" writes to stdout.
                        continue
                elif not operation.remote_name_all:
                    # No paired output and no --remote-name-all: stdout.
                    continue
            path = _url_path(url)
            if path is None:
                continue
            if client == "curl":
                name = path.rsplit("/", 1)[-1]
                if not name:
                    continue
            elif not path or path.endswith("/"):
                name = (default_page or ("index.html",))[-1]
            else:
                name = path.rsplit("/", 1)[-1]
            prefix = operation.output_dir
            names.append(f"{prefix}/{name}" if prefix else name)
    return names


def network_client_path_operands(  # noqa: PLR0912
    client: str, args: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The local-file operands a modeled client reads and writes.

    Returns ``(read_operands, write_operands)`` for the shell policy layer:
    curl's ``-o``/``--output``, ``--output-dir``, ``--dump-header``,
    ``--trace*``, ``--cookie-jar``, ``--etag-save``, ``--alt-svc``,
    ``--hsts`` and ``--libcurl`` name written files; wget's
    ``-O``/``--output-document``, ``-P``/``--directory-prefix``,
    ``--save-cookies``, ``-o``/``--output-file``, ``-a``/``--append-output``,
    ``--rejected-log``, ``--warc-file`` and ``--hsts-file`` do. The
    derived download destinations join the write side too: curl's
    ``-O``/``--remote-name`` and wget's default download write the URL's
    file part in the working directory (under ``--output-dir`` /
    ``-P``), which is derivable at resolve time. The read side covers the
    config, cookie, upload, certificate, data-file (including
    ``--data-urlencode``'s and ``--json``'s ``@file`` forms, and the
    mid-token ``name@file`` form), form and ``@``-file header operands
    (long and short spellings), including the form content part's
    comma-separated ``@file`` list and the ``headers=@file``/
    ``headers=<file`` parameters. curl's ``--output-dir`` also composes
    with its own operation's relative ``-o``/``--output`` operands (curl
    writes ``<output-dir>/<name>``), so the composed path — resolved
    through symlinks and metadata-guarded like every write operand —
    joins the write side alongside the bare operands; the option is local
    and resets at ``--next``, so one operation's directory never composes
    with another operation's ``-o`` operand. The access class comes from
    the resolver's per-command classification, so both lists are
    workspace-scoped and metadata-guarded like every other operand.
    """
    if client == "curl":
        args = _curl_operation_args(args)
    table = _client_table(client)
    parsed = _scan_options(args, table)
    if client == "curl":
        read_names = (*_CURL_READ_LONG, *_CURL_READ_SHORT)
        write_names = (*_CURL_WRITE_LONG, *_CURL_WRITE_SHORT)
        data_names = (*_CURL_DATA_LONG, *_CURL_DATA_SHORT)
        urldata_names = _CURL_URLDATA_LONG
        header_names = _CURL_HEADER_NAMES
        form_names = (*_CURL_FORM_LONG, *_CURL_FORM_SHORT)
    else:
        read_names = (*_WGET_READ_LONG, *_WGET_READ_SHORT)
        write_names = (*_WGET_WRITE_LONG, *_WGET_WRITE_SHORT)
        data_names = ()
        urldata_names = ()
        header_names = ()
        form_names = ()
    reads = list(parsed.values_for(*read_names))
    writes = list(parsed.values_for(*write_names))
    writes.extend(_download_write_operands(client, args, parsed))
    if client == "curl":
        # --output-dir composes with its OWN operation's relative
        # -o/--output operands: curl writes "<output-dir>/<name>"
        # (tool_operate.c), so the composed path is the write the policy
        # must inspect. An absolute -o operand and "-" (stdout) are not
        # composed, and an operation without --output-dir composes nothing.
        streams = _curl_download_streams(args, table, parsed)
        if streams is None:
            # The ordered mirror diverged (it cannot for a faithful
            # mirror): fall back to composing every relative -o operand
            # with the last --output-dir, which over-collects rather than
            # under-collects.
            directory = parsed.values_for("--output-dir")
            if directory:
                prefix = directory[-1]
                for value in parsed.values_for("--output", "-o"):
                    if value and value != "-" and not value.startswith("/"):
                        writes.append(f"{prefix}/{value}")
        else:
            for operation in streams:
                if not operation.output_dir:
                    continue
                for kind, value in operation.outputs:
                    if (
                        kind == "file"
                        and value
                        and value != "-"
                        and not value.startswith("/")
                    ):
                        writes.append(f"{operation.output_dir}/{value}")
    urldata = set(parsed.values_for(*urldata_names))
    for value in parsed.values_for(*data_names):
        reads.extend(_file_read_operands(value))
        if value in urldata:
            reads.extend(_urlencode_read_operands(value))
    for value in parsed.values_for(*header_names):
        reads.extend(_header_read_operands(value))
    for value in parsed.values_for(*form_names):
        reads.extend(_form_read_operands(value))
    return tuple(reads), tuple(writes)

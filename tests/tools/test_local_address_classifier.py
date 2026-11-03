"""Fail-closed address-classification tests (v0.7.1 S4b1a, shared with S4b2a + S4b2b).

``classify_local_address`` in ``chartreux/core/dangerous_commands.py`` owns
the locality verdict for a remote-exec destination host: canonicalization
admits DNS names and IP literals syntactically but never proves a literal
non-local, so this classifier is the fail-closed gate the S4b1b activation
consumes at the resolver seam. ``classify_hostname_resolution`` (S4b2a) owns
the resolve-validate verdict for a DNS-name destination: the system resolver
— which honors ``/etc/hosts`` — must yield exactly one distinct address and
that answer must classify non-local. ``classify_resolution_rebind`` (S4b2b)
owns the changed-answer verdict: a re-observed resolution that differs from
the validated pinned address — in particular a rebind to a local address —
denies, so the changed answer is never contacted.

The ``classify`` fixture is the shared carrier for this file: the S4b1
IP-literal arms below use it now, and the S4b2a hostname-resolution arms use
the ``resolve`` fixture beside it. Every arm runs against a controlled
local-interface enumeration and a controlled resolver, so the suite is
hermetic (no real getaddrinfo, no network).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import ipaddress
import socket
from typing import cast

import pytest

from chartreux.core.dangerous_commands import (
    AddressLocality,
    HostnameRebind,
    HostnameResolution,
    _local_interface_addresses,
    _system_hostname_resolver,
    canonical_remote_destination,
    classify_hostname_resolution,
    classify_local_address,
    classify_resolution_rebind,
)

_DEFAULT_INTERFACES = ("127.0.0.1", "::1", "192.168.10.20", "fe80::1%eth0")

_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@pytest.fixture
def classify() -> Callable[..., AddressLocality]:
    """Classify a host against a controlled local-interface enumeration.

    ``interfaces`` controls the enumeration's data (an unparseable entry
    simulates a malformed enumeration result), ``error`` makes the
    enumeration raise (the getaddrinfo failure case), and ``interfaces=()``
    is the absence-of-data case.
    """

    def _classify(
        host: str,
        *,
        interfaces: Iterable[str] = _DEFAULT_INTERFACES,
        error: Exception | None = None,
    ) -> AddressLocality:
        def enumerate_interfaces() -> frozenset[_IPAddress]:
            if error is not None:
                raise error
            return frozenset(ipaddress.ip_address(text) for text in interfaces)

        return classify_local_address(host, interface_addresses=enumerate_interfaces)

    return _classify


@pytest.fixture
def resolve() -> Callable[..., tuple[HostnameResolution, _IPAddress | None]]:
    """Resolve-validate a hostname with controlled resolution answers.

    ``answers`` is the resolver's distinct address set for the host (the
    /etc/hosts-or-DNS answer the system resolver would give), ``error`` makes
    the resolver raise (the DNS failure case), and ``answers=()`` is the
    empty-answer case. ``interfaces``/``enum_error`` control the locality
    enumeration the answer is classified against, exactly as in ``classify``.
    """

    def _resolve(
        host: str,
        *,
        answers: Iterable[str] = ("93.184.216.34",),
        error: Exception | None = None,
        interfaces: Iterable[str] = _DEFAULT_INTERFACES,
        enum_error: Exception | None = None,
    ) -> tuple[HostnameResolution, _IPAddress | None]:
        def resolver(name: str) -> frozenset[_IPAddress]:
            if error is not None:
                raise error
            return frozenset(ipaddress.ip_address(text) for text in answers)

        def enumerate_interfaces() -> frozenset[_IPAddress]:
            if enum_error is not None:
                raise enum_error
            return frozenset(ipaddress.ip_address(text) for text in interfaces)

        return classify_hostname_resolution(
            host, resolver=resolver, interface_addresses=enumerate_interfaces
        )

    return _resolve


class TestPublicLiteralsClassifyNonLocal:
    """The POSITIVE arm: a refuse-all degenerate classifier fails here."""

    @pytest.mark.parametrize(
        "host", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"]
    )
    def test_global_literal_is_non_local(self, classify, host: str) -> None:
        assert classify(host) == "non_local"

    def test_bracketed_global_v6_is_non_local(self, classify) -> None:
        assert classify("[2606:4700:4700::1111]") == "non_local"

    def test_enumeration_with_data_is_not_refuse_all(self, classify) -> None:
        # A succeeding enumeration that owns other addresses must not turn
        # the classifier refuse-all: a public literal no interface owns is
        # still non-local.
        interfaces = (*_DEFAULT_INTERFACES, "203.0.113.9", "2606:4700:4700::9999")
        assert classify("8.8.8.8", interfaces=interfaces) == "non_local"
        assert classify("1.1.1.1", interfaces=interfaces) == "non_local"


class TestUnspecifiedAndThisNetworkDeny:
    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "0:0:0:0:0:0:0:0", "0.0.0.1", "0.255.255.255", "0.1.2.3"],
    )
    def test_unspecified_and_this_network_deny(self, classify, host: str) -> None:
        assert classify(host) == "local"

    @pytest.mark.parametrize("host", ["0", "0.0", "0x0", "0x7f.0.0.1"])
    def test_inet_aton_shorthand_forms_deny(self, classify, host: str) -> None:
        # ipaddress rejects inet_aton shorthand outright; the parse failure
        # is a classification failure and denies.
        assert classify(host) == "local"


class TestLoopbackDenies:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "127.0.0.2",
            "127.255.255.254",
            "::1",
            "0:0:0:0:0:0:0:1",
            "[::1]",
            "127.1",
        ],
    )
    def test_loopback_denies(self, classify, host: str) -> None:
        assert classify(host) == "local"


class TestLinkLocalDenies:
    @pytest.mark.parametrize(
        "host",
        ["169.254.0.1", "169.254.255.255", "fe80::1", "fe80::1%eth0", "febf::ffff"],
    )
    def test_link_local_denies(self, classify, host: str) -> None:
        assert classify(host) == "local"


class TestUniqueLocalAndPrivateDeny:
    @pytest.mark.parametrize(
        "host",
        [
            "fc00::1",
            "fd12:34::1",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "100.64.0.1",
        ],
    )
    def test_unique_local_and_private_deny(self, classify, host: str) -> None:
        assert classify(host) == "local"


class TestMulticastDenies:
    # Multicast literals are is_global on this Python, so the explicit
    # is_multicast check is what denies them.
    @pytest.mark.parametrize(
        "host", ["224.0.0.1", "239.255.255.255", "ff02::1", "ff0e::1"]
    )
    def test_multicast_denies(self, classify, host: str) -> None:
        assert classify(host) == "local"


class TestMappedAndEmbeddedV6Denies:
    # ::ffff:8.8.8.8 is the load-bearing arm: mapped PUBLIC IPv6 is globally
    # routed as a literal, so only the mapped check denies it.
    @pytest.mark.parametrize(
        "host",
        [
            "::ffff:127.0.0.1",
            "[::ffff:127.0.0.1]",
            "::ffff:8.8.8.8",
            "[::ffff:8.8.8.8]",
            "::ffff:0:8.8.8.8",
            "::8.8.8.8",
            "64:ff9b::808:808",
            "2002:0808:0808::1",
            "2001::808:808",
        ],
    )
    def test_mapped_and_embedded_v6_denies(self, classify, host: str) -> None:
        assert classify(host) == "local"


class TestReservedAndUnassignableDeny:
    @pytest.mark.parametrize(
        "host",
        [
            "240.0.0.1",
            "255.255.255.255",
            "192.0.2.1",
            "198.51.100.1",
            "203.0.113.1",
            "198.18.0.1",
            "2001:db8::1",
            "[2001:db8::1]",
        ],
    )
    def test_reserved_and_unassignable_deny(self, classify, host: str) -> None:
        assert classify(host) == "local"


class TestInterfaceOwnedDenies:
    def test_interface_owned_public_literal_denies(self, classify) -> None:
        assert (
            classify("8.8.8.8", interfaces=(*_DEFAULT_INTERFACES, "8.8.8.8")) == "local"
        )
        assert (
            classify(
                "2606:4700:4700::1111",
                interfaces=(*_DEFAULT_INTERFACES, "2606:4700:4700::1111"),
            )
            == "local"
        )

    def test_default_interface_set_owns_loopback(self, classify) -> None:
        assert classify("127.0.0.1") == "local"
        assert classify("::1") == "local"


class TestEnumerationFailureDenies:
    def test_getaddrinfo_error_denies_even_a_public_literal(self, classify) -> None:
        error = socket.gaierror(-2, "Name or service not known")
        assert classify("8.8.8.8", error=error) == "local"
        assert classify("2606:4700:4700::1111", error=error) == "local"

    def test_generic_enumeration_error_denies(self, classify) -> None:
        assert (
            classify("8.8.8.8", error=OSError("interface enumeration failed"))
            == "local"
        )

    def test_empty_enumeration_denies(self, classify) -> None:
        assert classify("8.8.8.8", interfaces=()) == "local"

    def test_malformed_enumeration_entry_denies(self, classify) -> None:
        assert classify("8.8.8.8", interfaces=("not-an-address",)) == "local"


class TestNonLiteralsDeny:
    """Anything that does not parse as an IP literal is a locality
    classification failure and denies here; DNS names route through the
    S4b2a resolve-validate model (``classify_hostname_resolution``) instead,
    whose arms live in the classes below.
    """

    @pytest.mark.parametrize(
        "host",
        [
            "example.com",
            "deploy.example.com.",
            "localhost",
            "",
            "   ",
            "999.1.1.1",
            "1.2.3.4.5",
            "010.0.0.1",
            "256.256.256.256",
            "::gg::1",
        ],
    )
    def test_non_literal_denies(self, classify, host: str) -> None:
        assert classify(host) == "local"

    def test_non_string_host_denies(self) -> None:
        assert classify_local_address(cast("str", None)) == "local"


class TestLocalInterfaceEnumeration:
    def test_enumeration_parses_getaddrinfo_sockaddrs(self, monkeypatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.20", 0)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1%2", 0, 0, 2)),
            ],
        )
        assert _local_interface_addresses() == frozenset({
            ipaddress.ip_address("192.168.10.20"),
            ipaddress.ip_address("fe80::1%2"),
        })

    def test_enumeration_error_propagates(self, monkeypatch) -> None:
        def fail(host: str, port: object) -> list:
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        with pytest.raises(socket.gaierror):
            _local_interface_addresses()

    def test_enumeration_malformed_sockaddr_propagates(self, monkeypatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("not-an-address", 0))
            ],
        )
        with pytest.raises(ValueError):
            _local_interface_addresses()

    def test_default_enumerator_path_fails_closed(self, monkeypatch) -> None:
        def fail(host: str, port: object) -> list:
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        assert classify_local_address("8.8.8.8") == "local"

    def test_default_enumerator_path_classifies_public(self, monkeypatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.20", 0))
            ],
        )
        assert classify_local_address("8.8.8.8") == "non_local"


class TestDestinationHostIntegration:
    """The classifier consumes the canonical destination identity's host."""

    def test_canonical_destination_hosts_classify(self, classify) -> None:
        public = canonical_remote_destination("deploy@8.8.8.8:2222")
        assert public is not None
        assert classify(public.host) == "non_local"

        test_net = canonical_remote_destination("deploy@192.0.2.7")
        assert test_net is not None
        assert classify(test_net.host) == "local"

        doc_v6 = canonical_remote_destination("deploy@[2001:db8::1]")
        assert doc_v6 is not None
        assert classify(doc_v6.host) == "local"

        hostname = canonical_remote_destination("deploy@example.com")
        assert hostname is not None
        assert classify(hostname.host) == "local"


# --- v0.7.1 S4b2a: hostname resolve-validate arms (shared carrier) ----------
#
# A hostname destination is admitted only when the system resolver — which
# honors /etc/hosts — yields exactly ONE distinct address and that answer
# classifies non-local under the S4b1a classifier. Ambiguity (multi-answer),
# any local answer, and any resolution failure deny.


class TestUniqueNonLocalResolutionAdmits:
    """The POSITIVE arm: a refuse-all resolution model fails here."""

    def test_unique_non_local_answer_admits_with_the_address(self, resolve) -> None:
        verdict, address = resolve("example.com", answers=("93.184.216.34",))
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("93.184.216.34")

    def test_etc_hosts_only_name_admits(self, resolve) -> None:
        # A name only /etc/hosts can answer (no DNS record exists) resolves
        # through the system resolver exactly like a DNS name: the answer the
        # resolver returns is the answer that gets pinned.
        verdict, address = resolve("deploy.corp.internal", answers=("93.184.216.34",))
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("93.184.216.34")

    def test_unique_v6_answer_admits(self, resolve) -> None:
        verdict, address = resolve("v6.example.com", answers=("2606:4700:4700::1111",))
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("2606:4700:4700::1111")

    def test_duplicate_getaddrinfo_rows_dedupe_to_one_answer(self, resolve) -> None:
        # getaddrinfo returns one row per socktype; the same address repeated
        # is still a UNIQUE answer, not ambiguity.
        verdict, address = resolve(
            "example.com", answers=("93.184.216.34", "93.184.216.34")
        )
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("93.184.216.34")

    def test_answer_not_interface_owned_admits(self, resolve) -> None:
        interfaces = (*_DEFAULT_INTERFACES, "203.0.113.9")
        verdict, address = resolve("example.com", interfaces=interfaces)
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("93.184.216.34")


class TestAmbiguousResolutionDenies:
    def test_multi_answer_resolution_denies(self, resolve) -> None:
        verdict, address = resolve(
            "roundrobin.example.com", answers=("93.184.216.34", "1.1.1.1")
        )
        assert verdict == "ambiguous"
        assert address is None

    def test_multi_answer_with_a_local_answer_denies_as_ambiguous(
        self, resolve
    ) -> None:
        verdict, _ = resolve(
            "split.example.com", answers=("93.184.216.34", "127.0.0.1")
        )
        assert verdict == "ambiguous"

    def test_v4_and_v6_answers_deny_as_ambiguous(self, resolve) -> None:
        verdict, _ = resolve(
            "dual.example.com", answers=("93.184.216.34", "2606:4700:4700::1111")
        )
        assert verdict == "ambiguous"


class TestLocalAnswerDenies:
    @pytest.mark.parametrize(
        "answers",
        [("127.0.0.1",), ("10.0.0.5",), ("::1",), ("::ffff:8.8.8.8",), ("0.0.0.0",)],
    )
    def test_single_local_answer_denies(self, resolve, answers) -> None:
        verdict, address = resolve("local.example.com", answers=answers)
        assert verdict == "local"
        assert address == ipaddress.ip_address(answers[0])

    def test_interface_owned_answer_denies(self, resolve) -> None:
        verdict, address = resolve("self.example.com", answers=("192.168.10.20",))
        assert verdict == "local"
        assert address == ipaddress.ip_address("192.168.10.20")

    def test_enumeration_failure_denies_even_a_public_answer(self, resolve) -> None:
        verdict, _ = resolve(
            "example.com", enum_error=socket.gaierror(-2, "Name or service not known")
        )
        assert verdict == "local"

    def test_empty_enumeration_denies_a_public_answer(self, resolve) -> None:
        verdict, _ = resolve("example.com", interfaces=())
        assert verdict == "local"


class TestResolutionFailureDenies:
    def test_getaddrinfo_error_denies(self, resolve) -> None:
        verdict, address = resolve(
            "nx.example.com", error=socket.gaierror(-2, "Name or service not known")
        )
        assert verdict == "failed"
        assert address is None

    def test_generic_resolver_error_denies(self, resolve) -> None:
        verdict, _ = resolve("example.com", error=OSError("resolver down"))
        assert verdict == "failed"

    def test_empty_answer_set_denies(self, resolve) -> None:
        verdict, address = resolve("empty.example.com", answers=())
        assert verdict == "failed"
        assert address is None

    def test_ip_literal_is_not_a_hostname(self, resolve) -> None:
        # Literals route through classify_local_address, never through
        # resolution: the resolve-validate model refuses them.
        verdict, address = resolve("8.8.8.8")
        assert verdict == "failed"
        assert address is None

    def test_non_string_host_denies(self) -> None:
        verdict, address = classify_hostname_resolution(cast("str", None))
        assert verdict == "failed"
        assert address is None


class TestSystemResolverEnumeration:
    """The default resolver IS the system resolver: getaddrinfo, which
    consults the nsswitch chain and honors /etc/hosts ahead of DNS.
    """

    def test_default_resolver_maps_getaddrinfo_answers(self, monkeypatch) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("93.184.216.34", 0)),
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("2606:4700:4700::1111", 0, 0, 0),
                ),
            ],
        )
        assert _system_hostname_resolver("example.com") == frozenset({
            ipaddress.ip_address("93.184.216.34"),
            ipaddress.ip_address("2606:4700:4700::1111"),
        })

    def test_default_resolver_error_propagates(self, monkeypatch) -> None:
        def fail(host: str, port: object) -> list:
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        with pytest.raises(socket.gaierror):
            _system_hostname_resolver("nx.example.com")

    def test_default_resolver_path_classifies_unique_non_local(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda host, port: (
                [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
                if host == "example.com"
                else [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.20", 0))]
            ),
        )
        verdict, address = classify_hostname_resolution("example.com")
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("93.184.216.34")

    def test_default_resolver_path_denies_on_error(self, monkeypatch) -> None:
        def fail(host: str, port: object) -> list:
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        verdict, address = classify_hostname_resolution("nx.example.com")
        assert verdict == "failed"
        assert address is None


class TestHostnameDestinationIntegration:
    """The resolve-validate model consumes the canonical destination's host."""

    def test_canonical_hostname_destination_resolves(self, resolve) -> None:
        destination = canonical_remote_destination("deploy@example.com:2222")
        assert destination is not None
        verdict, address = resolve(destination.host, answers=("93.184.216.34",))
        assert verdict == "unique_non_local"
        assert address == ipaddress.ip_address("93.184.216.34")

    def test_canonical_hostname_destination_ambiguous_denies(self, resolve) -> None:
        destination = canonical_remote_destination("deploy@example.com")
        assert destination is not None
        verdict, _ = resolve(destination.host, answers=("93.184.216.34", "1.1.1.1"))
        assert verdict == "ambiguous"


@pytest.fixture
def rebind() -> Callable[..., tuple[HostnameRebind, _IPAddress | None]]:
    """Re-observe a hostname's resolution against the validated pinned address.

    ``pinned`` is the validated pinned address snapshot (S4b2a's pin),
    ``answers`` is the FRESH resolver answer set the re-observation sees (the
    changed answer), and ``error``/``interfaces``/``enum_error`` control the
    failure modes exactly as in ``resolve``.
    """

    def _rebind(
        host: str,
        pinned: str = "93.184.216.34",
        *,
        answers: Iterable[str] = ("93.184.216.34",),
        error: Exception | None = None,
        interfaces: Iterable[str] = _DEFAULT_INTERFACES,
        enum_error: Exception | None = None,
    ) -> tuple[HostnameRebind, _IPAddress | None]:
        def resolver(name: str) -> frozenset[_IPAddress]:
            if error is not None:
                raise error
            return frozenset(ipaddress.ip_address(text) for text in answers)

        def enumerate_interfaces() -> frozenset[_IPAddress]:
            if enum_error is not None:
                raise enum_error
            return frozenset(ipaddress.ip_address(text) for text in interfaces)

        return classify_resolution_rebind(
            host, pinned, resolver=resolver, interface_addresses=enumerate_interfaces
        )

    return _rebind


# --- v0.7.1 S4b2b: rebinding arms — the changed-answer classifier path ------
#
# ``classify_resolution_rebind`` re-observes a hostname's resolution at the
# spawn and compares it with the validated pinned address: only an unchanged
# unique non-local answer keeps the pin; a changed answer — a rebind to a
# local/loopback/unspecified address, to another address, ambiguity, or a
# failure — denies, so the changed answer is never contacted (HHH).


class TestRebindUnchangedAdmits:
    """The POSITIVE arm: a refuse-all rebinding model fails here."""

    def test_same_unique_non_local_answer_is_unchanged(self, rebind) -> None:
        verdict, address = rebind(
            "example.com", "93.184.216.34", answers=("93.184.216.34",)
        )
        assert verdict == "unchanged"
        assert address == ipaddress.ip_address("93.184.216.34")

    def test_ipv6_spelling_variant_of_the_same_address_is_unchanged(
        self, rebind
    ) -> None:
        # Address equality is by value: an equivalent IPv6 spelling of the
        # pinned address is still the unchanged pin, not a change.
        verdict, address = rebind(
            "v6.example.com",
            "2606:4700:4700::1111",
            answers=("2606:4700:4700:0:0:0:0:1111",),
        )
        assert verdict == "unchanged"
        assert address == ipaddress.ip_address("2606:4700:4700::1111")


class TestRebindChangedDenies:
    def test_changed_to_another_non_local_address(self, rebind) -> None:
        verdict, address = rebind("example.com", "93.184.216.34", answers=("1.1.1.1",))
        assert verdict == "changed"
        assert address == ipaddress.ip_address("1.1.1.1")

    @pytest.mark.parametrize(
        "answers",
        [("127.0.0.1",), ("::1",), ("0.0.0.0",), ("10.0.0.5",), ("::ffff:8.8.8.8",)],
    )
    def test_rebind_to_a_local_answer_keeps_the_local_verdict(
        self, rebind, answers
    ) -> None:
        # A rebind to a local/loopback/unspecified address is the
        # load-bearing arm: the fresh answer classifies local, and the
        # changed answer is never contacted.
        verdict, address = rebind("example.com", "93.184.216.34", answers=answers)
        assert verdict == "local"
        assert address == ipaddress.ip_address(answers[0])

    def test_rebind_to_an_interface_owned_answer_denies(self, rebind) -> None:
        verdict, address = rebind(
            "example.com", "93.184.216.34", answers=("192.168.10.20",)
        )
        assert verdict == "local"
        assert address == ipaddress.ip_address("192.168.10.20")

    def test_rebind_to_ambiguity_denies(self, rebind) -> None:
        verdict, address = rebind(
            "example.com", "93.184.216.34", answers=("93.184.216.34", "1.1.1.1")
        )
        assert verdict == "ambiguous"
        assert address is None

    def test_rebind_to_a_resolution_failure_denies(self, rebind) -> None:
        verdict, address = rebind(
            "example.com",
            "93.184.216.34",
            error=socket.gaierror(-2, "Name or service not known"),
        )
        assert verdict == "failed"
        assert address is None

    def test_rebind_with_an_enumeration_failure_denies(self, rebind) -> None:
        verdict, address = rebind(
            "example.com",
            "93.184.216.34",
            enum_error=socket.gaierror(-2, "Name or service not known"),
        )
        assert verdict == "local"
        assert address == ipaddress.ip_address("93.184.216.34")


class TestRebindFailClosed:
    def test_unparseable_pinned_address_fails_closed(self, rebind) -> None:
        verdict, address = rebind("example.com", "not-an-address")
        assert verdict == "failed"
        assert address is None

    def test_non_string_pinned_address_fails_closed(self) -> None:
        verdict, address = classify_resolution_rebind("example.com", cast("str", None))
        assert verdict == "failed"
        assert address is None

    def test_ip_literal_host_never_routes_through_rebind_resolution(
        self, rebind
    ) -> None:
        # Literals are their own address (the spawn guard skips them); the
        # rebind classifier itself refuses a literal host fail-closed.
        verdict, address = rebind("8.8.8.8", "8.8.8.8")
        assert verdict == "failed"
        assert address is None


class TestRebindStatefulReresolution:
    """The re-resolution arm: the resolver's answer CHANGES between the
    validation observation and the spawn re-observation.
    """

    def test_answer_that_changes_between_observations_denies(self) -> None:
        answers_by_call = [("93.184.216.34",), ("127.0.0.1",)]
        calls: list[str] = []

        def resolver(name: str) -> frozenset[_IPAddress]:
            answers = answers_by_call[min(len(calls), len(answers_by_call) - 1)]
            calls.append(name)
            return frozenset(ipaddress.ip_address(text) for text in answers)

        def enumerate_interfaces() -> frozenset[_IPAddress]:
            return frozenset(ipaddress.ip_address(text) for text in _DEFAULT_INTERFACES)

        # The validation observation: unique non-local, so the address is
        # pinned as the validated snapshot.
        verdict, address = classify_hostname_resolution(
            "example.com", resolver=resolver, interface_addresses=enumerate_interfaces
        )
        assert verdict == "unique_non_local"
        assert address is not None
        pinned = str(address)
        # The spawn re-observation: the answer rebound to loopback — the
        # rebind verdict denies, and the changed answer is never contacted.
        rebind_verdict, rebind_address = classify_resolution_rebind(
            "example.com",
            pinned,
            resolver=resolver,
            interface_addresses=enumerate_interfaces,
        )
        assert rebind_verdict == "local"
        assert rebind_address == ipaddress.ip_address("127.0.0.1")
        assert rebind_address != ipaddress.ip_address(pinned)

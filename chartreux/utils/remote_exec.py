"""Remote-exec wire spellings and the client attestation value object.

This leaf owns the ``network.remote-exec`` operation id, the two
execution-bound client-tool capability spellings, and the frozen attestation
value object. It imports nothing from chartreux at runtime, so the core grant
models and the public app-server/client modules can share these definitions
without the client boundary reaching into ``chartreux.core``.
"""

from __future__ import annotations

from dataclasses import dataclass

REMOTE_EXEC_OPERATION_ID = "network.remote-exec"

# Execution-bound remote-exec client attestation (v0.7.1 S4b1b). The
# app-server wire carries these as ``client_tools`` capability entries on the
# initialize request; the constants name the wire spellings and the dataclass
# below is the value object the execution transports consume. The
# attestation is CLIENT-SELF-REPORTED: it closes accidental misrouting to an
# executor in a different network namespace or filesystem/SSH trust context,
# not a lying client.
REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY = "remote_exec/network_locality"
REMOTE_EXEC_SSH_TRUST_CAPABILITY = "remote_exec/ssh_trust_context"


@dataclass(frozen=True)
class RemoteExecAttestation:
    """What one client attests about the executor that owns its spawns.

    ``network_locality`` attests that the executing client shares this
    server's network namespace, so a server-side locality verdict describes
    the executor too. ``ssh_trust_context`` attests that the executor sees
    the same filesystem paths and SSH trust context (known-hosts files,
    config) the server validates. Only an execution-bound statement — both
    present — admits remote execution through a client-owned transport; an
    entry attesting locality ONLY hard-denies transfers (scp/rsync), whose
    footprint crosses into the filesystem/SSH trust context the locality
    claim does not cover.
    """

    network_locality: bool = False
    ssh_trust_context: bool = False

    @property
    def execution_bound(self) -> bool:
        """Whether the attestation admits remote execution at all."""
        return self.network_locality and self.ssh_trust_context

    @property
    def locality_only(self) -> bool:
        """Whether the entry attests network locality without the trust context."""
        return self.network_locality and not self.ssh_trust_context

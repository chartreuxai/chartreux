#!/usr/bin/env bash
# Fail-closed isolation gate for the spawned (stdio / PTY) test legs.
#
# Every spawned leg of the stdio topology runs THROUGH this script. It proves
# the kernel network-namespace mechanism works, and only then launches the given
# test command inside it. If any precondition fails it refuses and exits
# non-zero WITHOUT running the command: a spawned leg never runs unisolated.
#
# Usage: bash tests/pty/run_gate.sh "<pytest invocation>"
#
# Mechanism (B1 contract of record): bubblewrap, `bwrap --unshare-net`, a
# userns-backed network namespace. Inside the sandbox only `lo` exists, so the
# in-gate deterministic provider (tests/e2e/mock_server.py bound to 127.0.0.1)
# is reachable and every other destination is kernel-denied -- for the spawned
# root agent, its python tree and any native executable it spawns. The
# in-process egress guard (tests/differential/egress.py) stays in force as the
# second, in-process layer.
#
# The repository is bound read-only at its own path, so a test that tries to
# write inside the tree fails loudly instead of dirtying the checkout; scratch
# writes land on the in-namespace tmpfs mounted at /tmp.
#
# Environment knobs (all optional):
#   GATE_BWRAP                       sandbox binary (failure-injection/testing only)
#   PYTEST_CHARTREUX_RECEIPT_GATE    C5b executed-receipt gate transport
#   PYTEST_CHARTREUX_RECEIPT_JOURNAL receipt journal path; defaults to the tmpfs
#
# Exit status: 1 when the gate refuses or the launch cannot start, otherwise the
# launched command's own status.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BWRAP="${GATE_BWRAP:-bwrap}"
LAUNCH="${1:-}"
JOURNAL="${PYTEST_CHARTREUX_RECEIPT_JOURNAL:-/tmp/chartreux-gate-receipts.jsonl}"

refuse() {
    echo "GATE REFUSED (fail-closed): $1" >&2
    echo "the launch command was NOT run" >&2
    exit 1
}

if [ -z "$LAUNCH" ]; then
    refuse 'usage: bash tests/pty/run_gate.sh "<pytest invocation>"'
fi

# The sandbox prelude shared by the probes and the launch: the whole host root
# read-only, real /dev and /proc, a private writable /tmp, the repository
# re-bound read-only at its own path (it lives under /tmp for a session
# worktree), and a fresh network namespace.
sandbox() {
    "$BWRAP" \
        --ro-bind / / \
        --dev /dev \
        --proc /proc \
        --tmpfs /tmp \
        --ro-bind "$REPO_ROOT" "$REPO_ROOT" \
        --die-with-parent \
        --unshare-net \
        "$@"
}

echo "[gate 1/4] isolation mechanism present: $BWRAP" >&2
command -v "$BWRAP" >/dev/null 2>&1 ||
    refuse "isolation mechanism '$BWRAP' is missing; install bubblewrap (see .github/workflows/ci.yml)"

EGRESS_PROBE="$(
    cat <<'PY'
import socket
import sys

probe = socket.socket()
probe.settimeout(2)
try:
    probe.connect(("192.0.2.1", 80))
except OSError:
    sys.exit(0)
sys.exit("EGRESS-HAPPENED")
PY
)"

LOOPBACK_PROBE="$(
    cat <<'PY'
import socket
import threading

server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
port = server.getsockname()[1]


def serve() -> None:
    connection, _ = server.accept()
    connection.send(b"ok")
    connection.close()


threading.Thread(target=serve, daemon=True).start()
client = socket.socket()
client.settimeout(2)
try:
    client.connect(("127.0.0.1", port))
    if client.recv(2) != b"ok":
        raise SystemExit(1)
except OSError:
    raise SystemExit(1)
PY
)"

echo "[gate 2/4] namespace creation startup probe" >&2
sandbox -- true ||
    refuse "namespace creation failed (kernel/user-namespace restriction) - refusing to run unisolated"

echo "[gate 3/4] in-sandbox egress-negation self-check (non-loopback must be denied)" >&2
sandbox -- python3 -c "$EGRESS_PROBE" ||
    refuse "a non-loopback connection succeeded from inside the sandbox - isolation is not in effect"

echo "[gate 4/4] in-sandbox loopback self-check (the in-gate provider must be reachable)" >&2
sandbox -- python3 -c "$LOOPBACK_PROBE" ||
    refuse "loopback is unreachable inside the sandbox - the in-gate provider would be dead"

echo "[gate] preconditions passed - launching isolated" >&2
sandbox \
    --setenv PYTHONDONTWRITEBYTECODE 1 \
    --setenv UV_CACHE_DIR /tmp/chartreux-gate-uv-cache \
    --setenv PYTEST_CHARTREUX_PTY_GATE 1 \
    --setenv PYTEST_CHARTREUX_RECEIPT_JOURNAL "$JOURNAL" \
    --setenv PYTEST_ADDOPTS "${PYTEST_ADDOPTS:-} --basetemp=/tmp/chartreux-gate-basetemp -p no:cacheprovider" \
    --chdir "$REPO_ROOT" \
    -- bash -c "$LAUNCH"
status=$?
echo "[gate] launch finished with status $status" >&2
exit "$status"

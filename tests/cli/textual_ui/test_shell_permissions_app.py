"""Shell Permissions panel: inspection sections and the revoke flow.

The panel distinguishes live session grants, saved project grants (with the
disk revision), and blocking explicit denylist entries rendered as an absolute
veto. Revocation is per operation with the expected live generation and disk
revision from the last inspection; live and saved outcomes are reported
separately, a conflict marks the list stale and asks for a refresh instead of
claiming an unconditional revocation, and endpoint errors are surfaced
verbatim. Remote-exec authority is destination-keyed: each destination is its
own row under the live, accepted-config, or disk-only section and revokes
exactly that destination, while the one whole-scope row needs an explicit
confirmation because it removes every destination. The service layer is
protocol-only: these tests drive it through a fake resource, never a live
provider.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

import pytest
from textual.app import App
from textual.widgets import OptionList

from chartreux.app_server.protocol import (
    AppServerResponseError,
    DangerousCommandAuthorityReadResponse,
    DangerousCommandDenylistBlockWire,
    DangerousCommandGrantScopeWire,
    DangerousCommandGrantWire,
    DangerousCommandRevocationOutcome,
    DangerousCommandRevokeResponse,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.cli.textual_ui.widgets.shell_permissions_app import (
    ShellPermissionsApp,
    ShellPermissionsOptionList,
    ShellPermissionsResult,
)
from chartreux.ui.shell_permissions import ShellPermissionsService
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

PROJECT = "/home/example/projects/shop"
REPO = "/home/example/projects/shop"
GIT_DIR = "/home/example/projects/shop/.git"
REMOTE_EXEC = "network.remote-exec"
DESTINATION_A = "deploy@203.0.113.10:22"
DESTINATION_B = "deploy@203.0.113.20:2222"
DESTINATION_C = "ops@198.51.100.7:2200"


def _grant(
    operation_id: str, *, work_tree: str | None = REPO, destination: str | None = None
) -> DangerousCommandGrantWire:
    return DangerousCommandGrantWire(
        operation_id=operation_id,
        scope=DangerousCommandGrantScopeWire(git_dir=GIT_DIR, work_tree=work_tree),
        destination=destination,
    )


def _inspection(
    *,
    live: list[DangerousCommandGrantWire] | None = None,
    saved: list[DangerousCommandGrantWire] | None = None,
    saved_operations: list[str] | None = None,
    saved_destinations: list[str] | None = None,
    disk_revision: str | None = "rev-1",
    blocking: list[DangerousCommandDenylistBlockWire] | None = None,
    generation: int = 3,
) -> DangerousCommandAuthorityReadResponse:
    return DangerousCommandAuthorityReadResponse(
        project=PROJECT,
        live_authority_generation=generation,
        live_grants=live or [],
        accepted_saved_grants=saved or [],
        saved_operations=saved_operations,
        saved_destinations=saved_destinations,
        disk_revision=disk_revision,
        blocking_denylist=blocking or [],
    )


RevocationStatus = Literal["removed", "conflict", "skipped", "not_applicable", "failed"]


def _revoke(
    *,
    live: RevocationStatus = "removed",
    saved: RevocationStatus = "removed",
    generation: int = 4,
    live_generation: int | None = 4,
    warning: str | None = None,
) -> DangerousCommandRevokeResponse:
    return DangerousCommandRevokeResponse(
        live=DangerousCommandRevocationOutcome(status=live, generation=live_generation),
        saved=DangerousCommandRevocationOutcome(status=saved),
        live_authority_generation=generation,
        warning=warning,
    )


class FakeResource:
    """Protocol double for the P5 client resource methods."""

    def __init__(
        self,
        inspections: list[DangerousCommandAuthorityReadResponse],
        revoke: DangerousCommandRevokeResponse | None = None,
        *,
        revoke_error: Exception | None = None,
        inspect_error: Exception | None = None,
    ) -> None:
        self.inspections = list(inspections)
        self.revoke_response = revoke
        self.revoke_error = revoke_error
        self.inspect_error = inspect_error
        self.inspect_calls = 0
        self.revoke_calls: list[dict[str, object]] = []
        self.revoke_gate: asyncio.Event | None = None

    async def read_dangerous_command_authority(
        self,
    ) -> DangerousCommandAuthorityReadResponse:
        self.inspect_calls += 1
        if self.inspect_error is not None:
            raise self.inspect_error
        if len(self.inspections) > 1:
            return self.inspections.pop(0)
        return self.inspections[0]

    async def revoke_dangerous_command_grant(
        self,
        *,
        operation_id: str,
        expected_live_generation: int,
        expected_disk_revision: str | None,
        user_initiated: bool,
        destination: str | None = None,
    ) -> DangerousCommandRevokeResponse:
        self.revoke_calls.append({
            "operation_id": operation_id,
            "expected_live_generation": expected_live_generation,
            "expected_disk_revision": expected_disk_revision,
            "user_initiated": user_initiated,
            "destination": destination,
        })
        if self.revoke_gate is not None:
            await self.revoke_gate.wait()
        if self.revoke_error is not None:
            raise self.revoke_error
        assert self.revoke_response is not None
        return self.revoke_response


class PanelHarness(App[None]):
    def __init__(self, service: ShellPermissionsService) -> None:
        super().__init__()
        self.service = service
        self.result: object = None

    def on_mount(self) -> None:
        self.push_screen(ShellPermissionsApp(self.service), self._receive)

    def _receive(self, result: object) -> None:
        self.result = result


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for panel state")
        await pilot.pause(0.01)


def _panel(app: PanelHarness) -> ShellPermissionsApp:
    screen = app.screen
    assert isinstance(screen, ShellPermissionsApp)
    return screen


def _row_ids(screen: ShellPermissionsApp) -> list[str]:
    grants = screen.query_one("#shell-permissions-grants", OptionList)
    return [str(row.id) for row in grants.options if row.id is not None]


def _row_prompt(screen: ShellPermissionsApp, option_id: str) -> str:
    grants = screen.query_one("#shell-permissions-grants", OptionList)
    for row in grants.options:
        if row.id == option_id:
            return str(row.prompt)
    raise AssertionError(f"row not found: {option_id}")


def _feedback(screen: ShellPermissionsApp) -> str:
    return str(screen.query_one("#shell-permissions-feedback", NoMarkupStatic).content)


def _hint(screen: ShellPermissionsApp) -> str:
    return str(screen.query_one("#shell-permissions-hint", NoMarkupStatic).content)


def _project_line(screen: ShellPermissionsApp) -> str:
    return str(screen.query_one("#shell-permissions-project", NoMarkupStatic).content)


@pytest.fixture
def inspection() -> DangerousCommandAuthorityReadResponse:
    return _inspection(
        live=[_grant("git.push")],
        saved=[_grant("git.push")],
        saved_operations=["git.push"],
        blocking=[
            DangerousCommandDenylistBlockWire(
                pattern="git push",
                operations=["git.push", "git.push.force", "git.push.delete"],
            )
        ],
    )


@pytest.fixture
def harness(
    inspection: DangerousCommandAuthorityReadResponse,
) -> tuple[PanelHarness, FakeResource]:
    resource = FakeResource([inspection], revoke=_revoke())
    return PanelHarness(ShellPermissionsService(resource)), resource


class TestInspectionSections:
    @pytest.mark.asyncio
    async def test_sections_distinguish_live_saved_and_blocking(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, resource = harness
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            ids = _row_ids(screen)
            assert ids == [
                "head:live",
                "live:git.push",
                "head:saved",
                "saved:git.push",
                "head:denylist",
                "deny:git push",
            ]
            headings = {
                row.id: str(row.prompt)
                for row in screen.query_one(
                    "#shell-permissions-grants", OptionList
                ).options
                if row.id is not None and str(row.id).startswith("head:")
            }
            assert "Live session grants" in headings["head:live"]
            assert "Saved project grants" in headings["head:saved"]
            assert "Blocked by explicit denylist" in headings["head:denylist"]
            # Live and saved rows are labelled apart, with their scope.
            assert "live session" in _row_prompt(screen, "live:git.push")
            assert REPO in _row_prompt(screen, "live:git.push")
            assert "saved for this project" in _row_prompt(screen, "saved:git.push")
            assert REPO in _row_prompt(screen, "saved:git.push")
            # The blocking entry is an absolute veto, never a grant to revoke.
            veto = _row_prompt(screen, "deny:git push")
            assert "absolute veto" in veto
            assert "git.push.force" in veto
            # The project line carries the disk revision of the saved read.
            project = _project_line(screen)
            assert PROJECT in project
            assert "rev-1" in project
            assert resource.inspect_calls == 1

    @pytest.mark.asyncio
    async def test_disk_only_saved_operation_reported_apart(self) -> None:
        resource = FakeResource(
            [_inspection(saved=[], saved_operations=["git.push"])], revoke=_revoke()
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            ids = _row_ids(screen)
            assert "disk:git.push" in ids
            assert "saved:git.push" not in ids
            prompt = _row_prompt(screen, "disk:git.push")
            assert "on disk only" in prompt
            assert "not accepted by this session" in prompt

    @pytest.mark.asyncio
    async def test_empty_sections_render_none_rows(self) -> None:
        resource = FakeResource([_inspection(saved_operations=[])], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            ids = _row_ids(_panel(app))
            assert ids == [
                "head:live",
                "state:live",
                "head:saved",
                "state:saved",
                "head:denylist",
                "state:denylist",
            ]

    @pytest.mark.asyncio
    async def test_unavailable_saved_disk_read_is_reported(self) -> None:
        resource = FakeResource(
            [_inspection(saved_operations=None, disk_revision=None)], revoke=_revoke()
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            ids = _row_ids(_panel(app))
            assert "state:saved-unavailable" in ids
            assert "rev-1" not in _project_line(_panel(app))


class TestRevokeFlow:
    @pytest.mark.asyncio
    async def test_revoke_reports_live_and_saved_separately(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, resource = harness
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            # The live grant row is the first selectable row.
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            highlighted = grants.highlighted_option
            assert highlighted is not None
            assert str(highlighted.id) == "live:git.push"
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            call = resource.revoke_calls[0]
            assert call == {
                "operation_id": "git.push",
                "expected_live_generation": 3,
                "expected_disk_revision": "rev-1",
                "user_initiated": True,
                "destination": None,
            }
            feedback = _feedback(screen)
            # Live and saved outcomes are reported apart, not as one claim.
            assert "live session authority removed" in feedback
            assert "saved project grant removed" in feedback
            assert "git.push" in feedback

    @pytest.mark.asyncio
    async def test_revoke_from_disk_only_row_reports_sides_apart(self) -> None:
        inspection = _inspection(live=[], saved=[], saved_operations=["git.push"])
        refreshed = _inspection(live=[], saved=[], saved_operations=[])
        resource = FakeResource(
            [inspection, refreshed], revoke=_revoke(live="skipped", saved="removed")
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            ids = _row_ids(screen)
            grants.highlighted = ids.index("disk:git.push")
            await pilot.pause()
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            # The disk-only row revokes with the inspection's expected revisions.
            assert resource.revoke_calls == [
                {
                    "operation_id": "git.push",
                    "expected_live_generation": 3,
                    "expected_disk_revision": "rev-1",
                    "user_initiated": True,
                    "destination": None,
                }
            ]
            feedback = _feedback(screen)
            # Nothing live existed, so the live side is skipped, not removed;
            # the saved side is reported apart.
            assert "live removal skipped" in feedback
            assert "saved project grant removed" in feedback
            widget = screen.query_one("#shell-permissions-feedback", NoMarkupStatic)
            assert not widget.has_class("warning")
            assert not widget.has_class("error")
            # The reconciled list no longer carries the on-disk-only operation.
            assert "disk:git.push" not in _row_ids(screen)

    @pytest.mark.asyncio
    async def test_revoke_without_disk_revision_skips_saved_side(self) -> None:
        inspection = _inspection(
            live=[_grant("git.push")], saved=[], saved_operations=[], disk_revision=None
        )
        refreshed = _inspection(live=[], saved=[], saved_operations=[])
        resource = FakeResource(
            [inspection, refreshed], revoke=_revoke(saved="skipped")
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            highlighted = grants.highlighted_option
            assert highlighted is not None
            assert str(highlighted.id) == "live:git.push"
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            # No disk revision was read, so None is passed through.
            assert resource.revoke_calls == [
                {
                    "operation_id": "git.push",
                    "expected_live_generation": 3,
                    "expected_disk_revision": None,
                    "user_initiated": True,
                    "destination": None,
                }
            ]
            feedback = _feedback(screen)
            assert "live session authority removed" in feedback
            # The saved side was not attempted, reported without error or warning.
            assert "no saved removal attempted" in feedback
            widget = screen.query_one("#shell-permissions-feedback", NoMarkupStatic)
            assert not widget.has_class("warning")
            assert not widget.has_class("error")

    @pytest.mark.asyncio
    async def test_conflict_marks_stale_and_requires_refresh(
        self, inspection: DangerousCommandAuthorityReadResponse
    ) -> None:
        refreshed = _inspection(live=[], saved=[], saved_operations=[], generation=7)
        resource = FakeResource(
            [inspection, refreshed],
            revoke=_revoke(live="conflict", saved="skipped", live_generation=7),
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: screen._stale)
            feedback = _feedback(screen)
            # No unconditional "revoked": the conflict is reported as such.
            assert feedback.startswith("! Warning:")
            assert "nothing was revoked" in feedback
            assert "press R to refresh" in feedback
            assert "authority removed" not in feedback
            # The pre-revocation list is kept and marked stale.
            assert "live:git.push" in _row_ids(screen)
            assert "stale" in _project_line(screen)
            assert screen.query_one(
                "#shell-permissions-feedback", NoMarkupStatic
            ).has_class("warning")
            # The reconciliation flow: R re-inspects and clears the staleness.
            await pilot.press("r")
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            await _wait_until(pilot, lambda: not screen._stale)
            assert "stale" not in _project_line(screen)
            assert "live:git.push" not in _row_ids(screen)

    @pytest.mark.asyncio
    async def test_saved_failure_warns_authority_can_return(
        self, inspection: DangerousCommandAuthorityReadResponse
    ) -> None:
        warning = (
            "Removing the saved git.push grant for /repo did not succeed, so the "
            "authority can return on the next config reload or session start."
        )
        resource = FakeResource(
            [inspection], revoke=_revoke(saved="failed", warning=warning)
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(
                pilot, lambda: "saved removal failed" in _feedback(screen)
            )
            feedback = _feedback(screen)
            assert "live session authority removed" in feedback
            assert "saved removal failed" in feedback
            assert "can return on the next config reload" in feedback
            assert screen.query_one(
                "#shell-permissions-feedback", NoMarkupStatic
            ).has_class("warning")

    @pytest.mark.asyncio
    async def test_consequential_saved_side_is_not_applicable(
        self, inspection: DangerousCommandAuthorityReadResponse
    ) -> None:
        resource = FakeResource([inspection], revoke=_revoke(saved="not_applicable"))
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: "never saved" in _feedback(screen))
            feedback = _feedback(screen)
            assert "live session authority removed" in feedback
            assert (
                "never saved (consequential operations are not persisted)" in feedback
            )

    @pytest.mark.asyncio
    async def test_denylist_row_explains_veto_without_revoking(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, resource = harness
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            ids = _row_ids(screen)
            grants.highlighted = ids.index("deny:git push")
            await pilot.pause()
            assert "Explain veto" in _hint(screen)
            await pilot.press("enter")
            await pilot.pause()
            assert resource.revoke_calls == []
            feedback = _feedback(screen)
            assert "absolute veto" in feedback
            assert "No live or saved grant overrides it" in feedback
            assert "denylist" in feedback

    @pytest.mark.asyncio
    async def test_busy_revocation_blocks_second_mutation(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, resource = harness
        resource.revoke_gate = asyncio.Event()
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            assert screen._busy
            assert "Running: Revoking git.push" in _hint(screen)
            # While busy, Enter and R are refused.
            await pilot.press("enter")
            await pilot.press("r")
            await pilot.pause()
            assert len(resource.revoke_calls) == 1
            assert resource.inspect_calls == 1
            resource.revoke_gate.set()
            await _wait_until(pilot, lambda: not screen._busy)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            # A second revocation is accepted once the first finished.
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 2)

    @pytest.mark.asyncio
    async def test_endpoint_error_is_surfaced_verbatim(
        self, inspection: DangerousCommandAuthorityReadResponse
    ) -> None:
        error = AppServerResponseError(
            ProtocolError(
                code=ProtocolErrorCode.NOT_FOUND,
                message="Dangerous-command revocation requires a live registered session",
            )
        )
        resource = FakeResource([inspection], revoke_error=error)
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: _feedback(screen).startswith("Failed:"))
            feedback = _feedback(screen)
            assert "Revoking git.push failed" in feedback
            assert "requires a live registered session" in feedback
            assert screen.query_one(
                "#shell-permissions-feedback", NoMarkupStatic
            ).has_class("error")
            # The list is unchanged: the failed revocation removed nothing.
            assert "live:git.push" in _row_ids(screen)
            assert not screen._stale

    @pytest.mark.asyncio
    async def test_inspect_failure_reports_and_retries(self) -> None:
        error = AppServerResponseError(
            ProtocolError(
                code=ProtocolErrorCode.NOT_FOUND,
                message="Dangerous-command authority requires a live registered session",
            )
        )
        resource = FakeResource([_inspection()], revoke=_revoke(), inspect_error=error)
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._load_error is not None)
            screen = _panel(app)
            assert "state:unavailable" in _row_ids(screen)
            feedback = _feedback(screen)
            assert feedback.startswith("Failed: Could not inspect")
            assert "requires a live registered session" in feedback
            assert "unknown" in _project_line(screen)
            # R retries the inspection and reconciles.
            resource.inspect_error = None
            await pilot.press("r")
            await _wait_until(pilot, lambda: screen._inspection is not None)
            assert "head:live" in _row_ids(screen)
            assert _feedback(screen) == ""


class TestDestinationRevoke:
    """Remote-exec destination rows: selective revoke and the batch confirmation."""

    @pytest.mark.asyncio
    async def test_each_destination_is_labelled_by_its_entry_kind(self) -> None:
        resource = FakeResource(
            [
                _inspection(
                    live=[_grant(REMOTE_EXEC, destination=DESTINATION_A)],
                    saved=[
                        _grant(REMOTE_EXEC, destination=DESTINATION_A),
                        _grant(REMOTE_EXEC, destination=DESTINATION_B),
                    ],
                    saved_operations=[],
                    saved_destinations=[DESTINATION_A, DESTINATION_B, DESTINATION_C],
                )
            ],
            revoke=_revoke(),
        )
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            ids = _row_ids(screen)
            # Each exact destination is its own row, under the section that
            # holds that authority; the disk-only one is reported apart.
            assert ids == [
                "head:live",
                "whole:network.remote-exec",
                f"dest:live:{DESTINATION_A}",
                "head:saved",
                f"dest:saved:{DESTINATION_A}",
                f"dest:saved:{DESTINATION_B}",
                f"dest:disk:{DESTINATION_C}",
                "head:denylist",
                "state:denylist",
            ]
            # Three labelled entry kinds: live session, accepted config, disk.
            live_row = _row_prompt(screen, f"dest:live:{DESTINATION_A}")
            assert REMOTE_EXEC in live_row
            assert DESTINATION_A in live_row
            assert "live session" in live_row
            saved_row = _row_prompt(screen, f"dest:saved:{DESTINATION_B}")
            assert DESTINATION_B in saved_row
            assert "saved for this project (accepted config)" in saved_row
            disk_row = _row_prompt(screen, f"dest:disk:{DESTINATION_C}")
            assert DESTINATION_C in disk_row
            assert "on disk only; not accepted by this session" in disk_row
            # A destination row is never shown as authority for the whole
            # operation: the accepted section carries both rows separately.
            assert f"dest:saved:{DESTINATION_A}" != f"dest:saved:{DESTINATION_B}"
            # The whole-scope action states what it removes, and exists once.
            whole = _row_prompt(screen, "whole:network.remote-exec")
            assert "all destinations in every scope" in whole
            assert "confirm required" in whole
            assert ids.count("whole:network.remote-exec") == 1

    @pytest.mark.asyncio
    async def test_destination_row_revokes_exactly_that_destination(self) -> None:
        inspection = _inspection(
            live=[
                _grant(REMOTE_EXEC, destination=DESTINATION_A),
                _grant(REMOTE_EXEC, destination=DESTINATION_B),
            ],
            saved_operations=[],
            saved_destinations=[DESTINATION_A, DESTINATION_B],
        )
        refreshed = _inspection(saved_operations=[], saved_destinations=[])
        resource = FakeResource([inspection, refreshed], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            grants.highlighted = _row_ids(screen).index(f"dest:live:{DESTINATION_A}")
            await pilot.pause()
            assert "Revoke destination" in _hint(screen)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            # The control names the exact destination: never the whole scope.
            assert resource.revoke_calls == [
                {
                    "operation_id": REMOTE_EXEC,
                    "expected_live_generation": 3,
                    "expected_disk_revision": "rev-1",
                    "user_initiated": True,
                    "destination": DESTINATION_A,
                }
            ]
            feedback = _feedback(screen)
            assert f"Revoking {REMOTE_EXEC} ({DESTINATION_A})" in feedback
            assert "live session authority removed" in feedback
            assert DESTINATION_A in feedback

    @pytest.mark.asyncio
    async def test_whole_scope_row_requires_confirmation_before_revoking_all(
        self,
    ) -> None:
        inspection = _inspection(
            live=[_grant(REMOTE_EXEC, destination=DESTINATION_A)],
            saved=[_grant(REMOTE_EXEC, destination=DESTINATION_B)],
            saved_operations=[],
            saved_destinations=[DESTINATION_A, DESTINATION_B],
        )
        refreshed = _inspection(saved_operations=[], saved_destinations=[])
        resource = FakeResource([inspection, refreshed], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            grants.highlighted = _row_ids(screen).index("whole:network.remote-exec")
            await pilot.pause()
            assert "Revoke all" in _hint(screen)
            await pilot.press("enter")
            await pilot.pause()
            # Nothing is revoked until the user confirms the batch removal.
            assert resource.revoke_calls == []
            feedback = _feedback(screen)
            assert feedback.startswith("! Confirm:")
            assert "EVERY remote-exec destination" in feedback
            assert "every saved destination of this project" in feedback
            assert "y confirm" in _hint(screen)
            assert screen._confirm
            # `n` cancels the confirmation and changes no authority.
            await pilot.press("n")
            await pilot.pause()
            assert screen._confirm is False
            assert resource.revoke_calls == []
            assert _feedback(screen) == ""
            # Confirming performs the whole-scope removal (no destination).
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            assert resource.revoke_calls == [
                {
                    "operation_id": REMOTE_EXEC,
                    "expected_live_generation": 3,
                    "expected_disk_revision": "rev-1",
                    "user_initiated": True,
                    "destination": None,
                }
            ]
            assert f"Revoking {REMOTE_EXEC}:" in _feedback(screen)

    @pytest.mark.asyncio
    async def test_escape_cancels_the_confirmation_before_closing(self) -> None:
        inspection = _inspection(
            live=[_grant(REMOTE_EXEC, destination=DESTINATION_A)],
            saved_operations=[],
            saved_destinations=[DESTINATION_A],
        )
        resource = FakeResource([inspection], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            grants.highlighted = _row_ids(screen).index("whole:network.remote-exec")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert screen._confirm
            # The first Escape cancels the confirmation and keeps the panel.
            await pilot.press("escape")
            await pilot.pause()
            assert screen._confirm is False
            assert app.screen is screen
            assert resource.revoke_calls == []
            # The second Escape dismisses the panel.
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is not screen
            assert isinstance(app.result, ShellPermissionsResult)
            assert app.result.feedback == ""
            assert resource.revoke_calls == []

    @pytest.mark.asyncio
    async def test_pending_confirmation_refuses_a_second_revocation(self) -> None:
        inspection = _inspection(
            live=[_grant("git.push"), _grant(REMOTE_EXEC, destination=DESTINATION_A)],
            saved_operations=[],
            saved_destinations=[],
        )
        resource = FakeResource([inspection], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            grants.highlighted = _row_ids(screen).index("whole:network.remote-exec")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert screen._confirm
            # Selecting an ordinary operation row behind the prompt is refused.
            grants.highlighted = _row_ids(screen).index("live:git.push")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert resource.revoke_calls == []
            assert screen._confirm
            # Refresh abandons the pending confirmation instead of acting.
            await pilot.press("r")
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            assert resource.revoke_calls == []
            assert screen._confirm is False

    @pytest.mark.asyncio
    async def test_disk_only_destination_reported_and_revoked(self) -> None:
        inspection = _inspection(
            live=[], saved=[], saved_operations=[], saved_destinations=[DESTINATION_C]
        )
        refreshed = _inspection(saved_operations=[], saved_destinations=[])
        resource = FakeResource([inspection, refreshed], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            ids = _row_ids(screen)
            assert "dest:disk:" + DESTINATION_C in ids
            assert f"dest:saved:{DESTINATION_C}" not in ids
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            grants.highlighted = ids.index("dest:disk:" + DESTINATION_C)
            await pilot.pause()
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            assert resource.revoke_calls == [
                {
                    "operation_id": REMOTE_EXEC,
                    "expected_live_generation": 3,
                    "expected_disk_revision": "rev-1",
                    "user_initiated": True,
                    "destination": DESTINATION_C,
                }
            ]
            assert "dest:disk:" + DESTINATION_C not in _row_ids(screen)

    @pytest.mark.asyncio
    async def test_destination_error_names_the_destination_and_keeps_the_list(
        self,
    ) -> None:
        inspection = _inspection(
            live=[_grant(REMOTE_EXEC, destination=DESTINATION_A)],
            saved_operations=[],
            saved_destinations=[],
        )
        error = AppServerResponseError(
            ProtocolError(
                code=ProtocolErrorCode.INVALID_PARAMS,
                message="A destination-scoped revocation requires network.remote-exec",
            )
        )
        resource = FakeResource([inspection], revoke_error=error)
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            grants.highlighted = _row_ids(screen).index(f"dest:live:{DESTINATION_A}")
            await pilot.pause()
            await pilot.press("enter")
            await _wait_until(pilot, lambda: _feedback(screen).startswith("Failed:"))
            feedback = _feedback(screen)
            # The failed operation is named with the destination, verbatim.
            assert f"Revoking {DESTINATION_A} failed" in feedback
            assert "A destination-scoped revocation requires" in feedback
            assert screen.query_one(
                "#shell-permissions-feedback", NoMarkupStatic
            ).has_class("error")
            # Nothing was claimed and the list still shows the authority.
            assert f"dest:live:{DESTINATION_A}" in _row_ids(screen)
            assert not screen._stale

    @pytest.mark.asyncio
    async def test_destination_rows_are_usable_at_80x24(self) -> None:
        inspection = _inspection(
            live=[
                _grant(REMOTE_EXEC, destination=DESTINATION_A),
                _grant(REMOTE_EXEC, destination=DESTINATION_B),
            ],
            saved_operations=[],
            saved_destinations=[DESTINATION_A, DESTINATION_B],
        )
        resource = FakeResource([inspection], revoke=_revoke())
        app = PanelHarness(ShellPermissionsService(resource))
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            content = screen.query_one("#shell-permissions-content")
            assert content.has_class("fullscreen")
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            assert grants.has_focus
            # The compact floor keeps every destination reachable by j/k, and
            # each row's rendering stays inside the panel width.
            reached: list[str] = []
            for _ in range(6):
                option = grants.highlighted_option
                assert option is not None and option.id is not None
                reached.append(str(option.id))
                await pilot.press("j")
            assert f"dest:live:{DESTINATION_A}" in reached
            assert f"dest:live:{DESTINATION_B}" in reached
            assert "whole:network.remote-exec" in reached
            grants.highlighted = _row_ids(screen).index(f"dest:live:{DESTINATION_B}")
            await pilot.pause()
            assert "Revoke destination" in _hint(screen)
            assert screen.query_one("#shell-permissions-hint").display
            assert screen.query_one("#shell-permissions-feedback").display


class TestPanelConventions:
    @pytest.mark.asyncio
    async def test_80x24_renders_with_focus_and_accurate_hints(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, _resource = harness
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            content = screen.query_one("#shell-permissions-content")
            assert content.has_class("fullscreen")
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            assert grants.has_focus
            hint = _hint(screen)
            assert "navigate" in hint
            assert "Revoke" in hint
            assert "refresh" in hint
            assert "Esc" in hint
            # The focused row carries the block cursor: a background style, so
            # the current row stays visible without color.
            assert grants.highlighted is not None
            strip = grants.render_line(grants.highlighted)
            assert any(
                segment.style is not None
                and segment.style.bgcolor is not None
                and segment.style.bgcolor.type is not None
                for segment in strip
            )
            assert screen.query_one("#shell-permissions-hint").display
            assert screen.query_one("#shell-permissions-feedback").display

    @pytest.mark.asyncio
    @pytest.mark.parametrize("size", [(60, 20), (120, 36), (84, 28)])
    async def test_sizes_render_and_escape_closes(
        self, harness: tuple[PanelHarness, FakeResource], size: tuple[int, int]
    ) -> None:
        app, _resource = harness
        async with app.run_test(size=size) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            content = screen.query_one("#shell-permissions-content")
            fullscreen = size[0] < 84 or size[1] < 28
            assert content.has_class("fullscreen") is fullscreen
            assert (content.border_title == "Shell Permissions") is not fullscreen
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.result, ShellPermissionsResult)
            assert app.screen is not screen

    @pytest.mark.asyncio
    async def test_jk_and_tab_groups_and_back_action(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, _resource = harness
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            screen = _panel(app)
            grants = screen.query_one(
                "#shell-permissions-grants", ShellPermissionsOptionList
            )
            actions = screen.query_one("#shell-permissions-actions", OptionList)

            def highlighted_id() -> str:
                option = grants.highlighted_option
                assert option is not None
                return str(option.id)

            assert highlighted_id() == "live:git.push"
            await pilot.press("j")
            assert highlighted_id() == "saved:git.push"
            await pilot.press("k")
            assert highlighted_id() == "live:git.push"
            # Movement is bounded: it never lands on a heading.
            await pilot.press("k")
            assert highlighted_id() == "live:git.push"
            await pilot.press("tab")
            assert actions.has_focus
            assert "Refresh" in _hint(screen)
            await pilot.press("shift+tab")
            assert grants.has_focus
            # Enter on Back dismisses the panel with the outcome feedback.
            await pilot.press("tab")
            actions.highlighted = 1
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is not screen

    @pytest.mark.asyncio
    async def test_dismiss_returns_outcome_feedback(
        self, harness: tuple[PanelHarness, FakeResource]
    ) -> None:
        app, resource = harness
        async with app.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: _panel(app)._inspection is not None)
            await pilot.press("enter")
            await _wait_until(pilot, lambda: len(resource.revoke_calls) == 1)
            await _wait_until(pilot, lambda: resource.inspect_calls == 2)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.result, ShellPermissionsResult)
            assert "live session authority removed" in app.result.feedback
            assert "saved project grant removed" in app.result.feedback


class TestShellPermissionsService:
    @pytest.mark.asyncio
    async def test_inspect_maps_wire_to_panel_state(self) -> None:
        resource = FakeResource([
            _inspection(
                live=[_grant("git.push"), _grant("git.push", work_tree="/other")],
                saved=[_grant("git.push")],
                saved_operations=["git.push", "git.push.mirror"],
                blocking=[
                    DangerousCommandDenylistBlockWire(
                        pattern="git push", operations=["git.push"]
                    )
                ],
            )
        ])
        service = ShellPermissionsService(resource)
        state = await service.inspect()
        assert state.project == PROJECT
        assert state.live_authority_generation == 3
        assert state.disk_revision == "rev-1"
        assert [grant.operation_id for grant in state.live_grants] == [
            "git.push",
            "git.push",
        ]
        assert [grant.repository for grant in state.live_grants] == [REPO, "/other"]
        assert [grant.source for grant in state.live_grants] == ["live", "live"]
        assert [grant.source for grant in state.saved_grants] == ["saved"]
        # git.push.mirror is on disk but not accepted: reported apart.
        assert state.disk_only_operations == ("git.push.mirror",)
        assert not state.saved_operations_unavailable
        assert state.blocking_denylist[0].pattern == "git push"
        assert state.blocking_denylist[0].operations == ("git.push",)

    @pytest.mark.asyncio
    async def test_revoke_passes_user_initiated_and_expected_revisions(self) -> None:
        resource = FakeResource([_inspection()], revoke=_revoke())
        service = ShellPermissionsService(resource)
        outcome = await service.revoke(
            operation_id="git.push",
            expected_live_generation=3,
            expected_disk_revision="rev-1",
        )
        assert resource.revoke_calls == [
            {
                "operation_id": "git.push",
                "expected_live_generation": 3,
                "expected_disk_revision": "rev-1",
                "user_initiated": True,
                "destination": None,
            }
        ]
        assert outcome.live_status == "removed"
        assert outcome.saved_status == "removed"
        assert outcome.live_authority_generation == 4
        assert not outcome.conflicted

    @pytest.mark.asyncio
    async def test_revoke_conflict_is_an_outcome_not_an_error(self) -> None:
        resource = FakeResource(
            [_inspection()],
            revoke=_revoke(live="conflict", saved="skipped", live_generation=9),
        )
        service = ShellPermissionsService(resource)
        outcome = await service.revoke(
            operation_id="git.push",
            expected_live_generation=3,
            expected_disk_revision="rev-1",
        )
        assert outcome.conflicted
        assert outcome.live_status == "conflict"
        assert outcome.saved_status == "skipped"
        assert outcome.live_generation == 9

    @pytest.mark.asyncio
    async def test_inspect_projects_the_three_destination_entry_kinds(self) -> None:
        resource = FakeResource([
            _inspection(
                live=[_grant(REMOTE_EXEC, destination=DESTINATION_A)],
                saved=[_grant(REMOTE_EXEC, destination=DESTINATION_B)],
                saved_operations=[],
                saved_destinations=[DESTINATION_B, DESTINATION_C],
            )
        ])
        state = await ShellPermissionsService(resource).inspect()
        # The exact destination rides on the projection, never a bare operation.
        assert [
            (grant.operation_id, grant.destination, grant.source)
            for grant in state.live_grants
        ] == [(REMOTE_EXEC, DESTINATION_A, "live")]
        assert [
            (grant.operation_id, grant.destination, grant.source)
            for grant in state.saved_grants
        ] == [(REMOTE_EXEC, DESTINATION_B, "saved")]
        # Only the destination the accepted config does not hold is disk-only.
        assert state.disk_only_destinations == (DESTINATION_C,)
        assert state.disk_only_operations == ()

    @pytest.mark.asyncio
    async def test_revoke_passes_the_exact_destination(self) -> None:
        resource = FakeResource([_inspection()], revoke=_revoke())
        outcome = await ShellPermissionsService(resource).revoke(
            operation_id=REMOTE_EXEC,
            expected_live_generation=3,
            expected_disk_revision="rev-1",
            destination=DESTINATION_A,
        )
        assert resource.revoke_calls == [
            {
                "operation_id": REMOTE_EXEC,
                "expected_live_generation": 3,
                "expected_disk_revision": "rev-1",
                "user_initiated": True,
                "destination": DESTINATION_A,
            }
        ]
        assert outcome.destination == DESTINATION_A
        assert not outcome.conflicted

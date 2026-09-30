from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.config import ProxySettingsView
from chartreux.app_server.models import (
    QuestionChoice,
    UserQuestion,
    UserQuestionRequest,
)
from chartreux.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class QuestionLayoutHarness(App[None]):
    CSS_PATH = "../../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield QuestionApp(
            UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question=(
                            "Choose the features for this project and review every "
                            "available option."
                        ),
                        options=[
                            QuestionChoice(
                                label=f"Feature {index}",
                                description=(
                                    "A deliberately long description that wraps at a "
                                    "narrow terminal width."
                                ),
                            )
                            for index in range(1, 9)
                        ],
                        multi_select=True,
                    )
                ]
            )
        )


class ProxyLayoutHarness(App[None]):
    CSS_PATH = "../../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield ProxySetupApp(
            ProxySettingsView(
                values={f"PROXY_{index}": None for index in range(1, 6)},
                descriptions={
                    f"PROXY_{index}": (
                        "A long proxy setting description that wraps at a narrow "
                        "terminal width."
                    )
                    for index in range(1, 6)
                },
            )
        )


class CompactQuestionLayoutHarness(App[None]):
    CSS_PATH = "../../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield QuestionApp(
            UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Choose the features for this project.",
                        options=[
                            QuestionChoice(label=f"Feature {index}", description="")
                            for index in range(1, 4)
                        ],
                        multi_select=True,
                    )
                ]
            )
        )


class CompactProxyLayoutHarness(App[None]):
    CSS_PATH = "../../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield ProxySetupApp(
            ProxySettingsView(
                values={"HTTP_PROXY": None, "HTTPS_PROXY": None},
                descriptions={"HTTP_PROXY": "", "HTTPS_PROXY": ""},
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_question_panel_keeps_help_visible_and_scrolls_options(
    size: tuple[int, int],
) -> None:
    async with QuestionLayoutHarness().run_test(size=size) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(QuestionApp)
        body = app.query_one("#question-body")
        help_row = app.query_one(".question-help", NoMarkupStatic)

        assert app.region.bottom <= pilot.app.size.height
        assert app.region.height <= pilot.app.size.height // 2
        assert help_row.display
        assert help_row.region.bottom <= app.region.bottom

        await pilot.press("down", "down", "down", "down", "down", "down", "down")
        await pilot.pause()

        assert app.selected_option == 7
        if size == (80, 24):
            assert body.scroll_y > 0
            assert body.max_scroll_y > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        "Warning: Answer delivery is uncertain after the remote service timed out; "
        "your answer may still be processing.",
        "Failed: Answer submission was rejected because the remote service returned "
        "an invalid response and could not accept your answer.",
    ],
)
async def test_question_submission_status_keeps_all_help_lines_inside_panel(
    status: str,
) -> None:
    async with QuestionLayoutHarness().run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(QuestionApp)
        app.set_submission_status(status)
        await pilot.pause()

        help_row = app.query_one(".question-help", NoMarkupStatic)
        assert help_row.display
        assert help_row.region.y >= app.region.y
        assert help_row.region.bottom <= app.region.bottom
        assert "Answer" in str(help_row.content)
        assert help_row.virtual_size.height <= help_row.region.height


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_proxy_panel_keeps_help_inside_panel_at_narrow_size(
    size: tuple[int, int],
) -> None:
    async with ProxyLayoutHarness().run_test(size=size) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(ProxySetupApp)
        body = app.query_one("#proxysetup-form")
        help_row = app.query_one("#proxysetup-help", NoMarkupStatic)

        assert app.region.bottom <= pilot.app.size.height
        assert app.region.height <= pilot.app.size.height // 2
        assert help_row.display
        assert help_row.region.bottom <= app.region.bottom
        assert body.max_scroll_y > 0


@pytest.mark.asyncio
async def test_compact_question_panel_uses_natural_height_at_120x36() -> None:
    async with CompactQuestionLayoutHarness().run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(QuestionApp)
        body = app.query_one("#question-body")

        assert app.region.height < pilot.app.size.height // 2
        assert body.max_scroll_y == 0


@pytest.mark.asyncio
async def test_compact_question_panel_reaches_other_and_submit_at_80x24() -> None:
    async with CompactQuestionLayoutHarness().run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(QuestionApp)

        await pilot.press("down", "down", "down")
        await pilot.pause()
        assert app.selected_option == app._other_option_idx

        await pilot.press("down")
        await pilot.pause()
        assert app.selected_option == app._submit_option_idx


@pytest.mark.asyncio
async def test_compact_proxy_panel_uses_natural_height_at_120x36() -> None:
    async with CompactProxyLayoutHarness().run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(ProxySetupApp)
        body = app.query_one("#proxysetup-form")

        assert app.region.height < pilot.app.size.height // 2
        assert body.max_scroll_y == 0


@pytest.mark.asyncio
async def test_proxy_fields_reach_apply_at_80x24() -> None:
    async with ProxyLayoutHarness().run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(ProxySetupApp)
        inputs = list(app.inputs.values())

        app.focus()
        await pilot.pause()
        assert app.screen.focused is inputs[0]
        for input_widget in inputs[1:]:
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen.focused is input_widget

        await pilot.press("enter")
        await pilot.pause()

        assert app.screen.focused is app.query_one("#proxysetup-save")


@pytest.mark.asyncio
async def test_proxy_error_visibility_transitions_between_form_and_field_errors() -> (
    None
):
    async with CompactProxyLayoutHarness().run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app = pilot.app.query_one(ProxySetupApp)
        form_error = app.query_one("#proxysetup-error", NoMarkupStatic)
        field_error = app.query_one("#proxy-error-HTTP_PROXY", NoMarkupStatic)

        assert not form_error.display
        app.show_error("The proxy settings could not be saved right now")
        await pilot.pause()
        assert form_error.display
        assert not field_error.display

        app.show_error("HTTP_PROXY must use a valid URL")
        await pilot.pause()
        assert not form_error.display
        assert field_error.display
        assert field_error.styles.height is not None
        assert field_error.styles.height.unit.name == "AUTO"

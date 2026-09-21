from __future__ import annotations

from chartreux.app_server.config import ConfigView, ModelConfigView


def build_test_app_config(*, show_thinking_nodes: bool = False) -> ConfigView:
    return ConfigView(
        active_model=ModelConfigView(
            name="test-model",
            alias="test-model",
            thinking="off",
            supports_images=False,
            display_name="test-model",
        ),
        active_model_pinned=True,
        default_model_alias="test-model",
        theme="textual-dark",
        log_level=None,
        disable_welcome_banner_animation=False,
        autocopy_to_clipboard=True,
        file_watcher_for_autocomplete=False,
        ask_confirmation_on_exit=True,
        show_greeting=True,
        show_thinking_nodes=show_thinking_nodes,
        enable_notifications=True,
        models=[
            ModelConfigView(
                name="test-model",
                alias="test-model",
                thinking="off",
                supports_images=False,
                display_name="test-model",
            )
        ],
        validation_warnings=[],
    )

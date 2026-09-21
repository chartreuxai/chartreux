from __future__ import annotations

import pytest

from chartreux.core.tools.builtins.bash import Bash, BashArgs, CapturedShellResult


@pytest.mark.parametrize(
    ("tool_class", "args_model", "result_model"),
    [(Bash, BashArgs, CapturedShellResult)],
)
def test_shell_tools_declare_their_argument_and_result_models(
    tool_class, args_model, result_model
):
    assert tool_class._get_tool_args_results() == (args_model, result_model)

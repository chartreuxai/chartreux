from __future__ import annotations

from pathlib import Path


class TestBashPromptCheckoutExemptionContract:
    def test_bash_prompt_documents_sanctioned_delegated_switch_shapes(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/bash.md"
        ).read_text()

        assert "Clean-tree checkout exemption" in prompt
        assert "`git checkout <existing-local-branch>`" in prompt
        assert "`git checkout -b <new-branch> [<start-point>]`" in prompt
        assert "sanctioned delegated switch shapes" in prompt
        assert "run WITHOUT the dangerous-command approval flow" in prompt

    def test_bash_prompt_documents_whole_call_eligibility(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/bash.md"
        ).read_text()

        assert "the entire command is exactly one git invocation" in prompt
        assert "no inline `-c` options" in prompt
        assert "display-only globals such as `--no-pager` are fine" in prompt
        assert "Anything unestablishable fails closed to the approval flow" in prompt

    def test_bash_prompt_documents_state_guardrails(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/bash.md"
        ).read_text()

        assert "clean work tree (no unstaged, staged, or untracked changes)" in prompt
        assert (
            "no in-progress operation markers (merge, rebase, cherry-pick, "
            "revert)" in prompt
        )
        assert "no program-selecting config such as `core.hooksPath`" in prompt

    def test_bash_prompt_documents_injected_no_overwrite_ignore(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/bash.md"
        ).read_text()

        assert "--no-overwrite-ignore" in prompt
        assert (
            "an injected `--no-overwrite-ignore` immediately after the "
            "`checkout` token" in prompt
        )
        assert (
            "the command as you wrote it stays unchanged for display and "
            "results" in prompt
        )
        assert (
            "an ignored-file collision fails closed — checkout refuses — "
            "instead of silently overwriting local content" in prompt
        )

    def test_bash_prompt_keeps_user_owned_switches_user_owned(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/bash.md"
        ).read_text()

        assert (
            "Branch merging, switches to remote-tracking branches, and "
            "operations on branches other than the current working branch "
            "remain user-owned" in prompt
        )


class TestAdrDocumentsCheckoutExemption:
    @staticmethod
    def _adr_text() -> str:
        # The ADR prose is hard-wrapped; collapse whitespace so pinned
        # phrases match across line breaks.
        adr = (
            Path(__file__).parents[2] / "docs/adr/0019-dangerous-git-permissions.md"
        ).read_text()
        return " ".join(adr.split())

    def test_adr_documents_state_aware_exemption(self) -> None:
        adr = self._adr_text()

        assert "### State-aware clean-tree checkout exemption" in adr
        assert "no pathspec, no `--`" in adr
        assert "--recurse-submodules`, no `-B`" in adr
        assert "every candidate repository scope passes the state guardrails" in adr
        assert "a sanitized rev-parse probe" in adr

    def test_adr_documents_injected_flag_and_race_adjudication(self) -> None:
        adr = self._adr_text()

        assert "`--no-overwrite-ignore`" in adr
        assert (
            "the original string stays original for display and result surfaces" in adr
        )
        assert "preservation of concurrent user changes" in adr

    def test_adr_documents_classifier_divergence_and_preserved_classes(self) -> None:
        adr = self._adr_text()

        assert "Classifier divergence, deliberate and pinned" in adr
        assert "the git_pure corpus bytes stay frozen" in adr
        assert "The approval classes are preserved" in adr

    def test_adr_documents_sanctioned_switch_shapes_and_deferred_alternatives(
        self,
    ) -> None:
        adr = self._adr_text()

        assert "Sanctioned delegated switch shapes" in adr
        assert "remain USER-OWNED" in adr
        assert "git show :2:<path>`/`git show :3:<path>`" in adr
        assert "update-ref stays a hard guard" in adr

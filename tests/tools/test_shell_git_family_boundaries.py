"""Boundary contracts for the W3b git-family split (stage 4 close).

The git family is seven modules under ``chartreux/core/tools/builtins`` --
``_shell_argv``, ``_shell_executor_grammar``, ``_shell_policy_types``,
``_shell_git_support``, ``_shell_git_policy``, ``_shell_git_classifier``, and
``_shell_git_inspection`` -- composed by the ``_shell_command_policy`` facade.
This module pins the split's structural contracts: the pure/I/O module
boundary (only ``_shell_git_inspection`` touches the filesystem), the
no-facade-import acyclicity of every family module, the facade placement of
``git_repository_config_risk`` next to its ``analyze_shell_command_policy``
callee (the cycle-break), and the three classification properties the
design's differential verification calls for: hard_denied verdicts are never
grantable, grantable verdicts resolve their operations through the registry,
and reader subcommands opt into repository-config inspection.
"""

from __future__ import annotations

import ast
import inspect
import shlex

from chartreux.core.dangerous_commands import OPERATION_REGISTRY, operations_for_ids
from chartreux.core.tools.builtins import (
    _shell_argv,
    _shell_command_policy as policy,
    _shell_executor_grammar,
    _shell_git_classifier,
    _shell_git_inspection,
    _shell_git_policy,
    _shell_git_support,
    _shell_policy_types,
)

FAMILY_MODULES = (
    _shell_argv,
    _shell_executor_grammar,
    _shell_policy_types,
    _shell_git_support,
    _shell_git_policy,
    _shell_git_classifier,
    _shell_git_inspection,
)
IO_MODULE = _shell_git_inspection

# The filesystem-touch tokens the pure modules must never contain.
IO_TOKENS = ("read_text", ".stat(", "Path.cwd", ".is_dir(", ".is_file(", ".exists()")


def _top_level_defs(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _imports_facade(module) -> bool:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.endswith("._shell_command_policy"):
                return True
    return False


def _doc(module) -> str:
    doc = inspect.getdoc(module)
    assert doc is not None, module.__name__
    return doc


def test_git_family_io_module_boundary() -> None:
    """Only _shell_git_inspection performs filesystem I/O; the family is acyclic."""
    for module in FAMILY_MODULES:
        source = inspect.getsource(module)
        hits = [token for token in IO_TOKENS if token in source]
        if module is IO_MODULE:
            assert hits, "the family's I/O module must contain filesystem touches"
        else:
            assert not hits, f"{module.__name__} must stay pure, found {hits}"

    # Each split module's docstring declares its pure/I/O status.
    assert "only I/O module" in _doc(IO_MODULE)
    assert "The module is pure" in _doc(_shell_git_policy)
    assert "pure" in _doc(_shell_git_classifier)

    # No family module imports the facade, so the import graph stays acyclic
    # (this also pins the retired _shell_executor_grammar deferred import).
    for module in FAMILY_MODULES:
        assert not _imports_facade(module), f"{module.__name__} imports the facade"

    # git_repository_config_risk stays in the facade, next to its
    # analyze_shell_command_policy callee: the intra-module call is what
    # breaks the inspection<->facade cycle (ASG "breaking the :4182 edge").
    facade_defs = _top_level_defs(policy)
    assert "git_repository_config_risk" in facade_defs
    assert "analyze_shell_command_policy" in facade_defs
    inspection_defs = _top_level_defs(IO_MODULE)
    assert "git_repository_config_risk" not in inspection_defs
    assert "analyze_shell_command_policy" not in inspection_defs


def _classify(command: str):
    return policy.classify_dangerous_git_command(shlex.split(command))


# Hard guards, unscannable globals, aliases, unmodeled builtins, and
# unmodeled push refspecs: every one fails closed with no grantable operation.
HARD_DENIED_COMMANDS = (
    "git reset --hard",
    "git reset --hard HEAD",
    "git reset --ha HEAD",
    "git reset -q --hard HEAD",
    "git clean -f",
    "git clean -fd",
    "git clean --f",
    "git clean -x -f",
    "git clean -e .ign -f",
    "git clean --force --dry-run --no-dry-run",
    "git clean -fd --no-dry-run",
    "git -c foo=bar push",
    "git -cfoo=bar push",
    "git --config-env foo=bar push",
    "git --config-env=foo=bar push",
    "git fixture-alias",
    "git fsck",
    "git gc --auto",
    "git bundle list-heads",
    "git push :main",
    "git push origin 'ma*in'",
    "git -C",
)


def test_hard_denied_classifications_are_never_grantable() -> None:
    """A hard_denied verdict never carries or implies a grantable operation."""
    assert len(HARD_DENIED_COMMANDS) >= 15
    for command in HARD_DENIED_COMMANDS:
        classification = _classify(command)
        assert classification.kind == "hard_denied", command
        # Never grantable: no proposal payload, so no operation is offered
        # and no grant choice can authorize the call.
        assert classification.grantable is None, command
        assert classification.reason, command


# Grantable forms with the exact operation IDs the classifier must name.
GRANTABLE_COMMANDS = (
    ("git push", ("git.push",)),
    ("git push origin main", ("git.push",)),
    ("git push v1.0", ("git.push",)),
    ("git push --all origin", ("git.push",)),
    ("git push -u origin main", ("git.push",)),
    ("git push --force origin main", ("git.push.force",)),
    ("git push --delete origin main", ("git.push.delete",)),
    ("git push --mirror origin", ("git.push.mirror",)),
    ("git push --prune --force origin main", ("git.push.force", "git.push.delete")),
    ("git push --prune --mirror origin", ("git.push.delete", "git.push.mirror")),
    ("git checkout main", ("git.checkout",)),
    ("git checkout -b feat", ("git.checkout",)),
    ("git restore file", ("git.restore",)),
    ("git restore --worktree --staged file", ("git.restore",)),
    ("git switch -f main", ("git.switch.discard",)),
    ("git switch --discard-changes main", ("git.switch.discard",)),
    ("git stash drop", ("git.stash.drop",)),
    ("git stash clear", ("git.stash.clear",)),
    ("git reflog expire --all", ("git.reflog.expire",)),
    ("git reflog delete main@{0}", ("git.reflog.delete",)),
)


def test_grantable_classifications_resolve_operations() -> None:
    """A grantable verdict's operation IDs resolve through the registry."""
    assert len(GRANTABLE_COMMANDS) >= 15
    for command, operation_ids in GRANTABLE_COMMANDS:
        classification = _classify(command)
        assert classification.kind == "grantable", command
        assert classification.grantable is not None, command
        # The payload names exactly the expected operations...
        assert classification.grantable.operation_ids == operation_ids, command
        # ...and every named operation resolves in the registry, in order.
        resolved = operations_for_ids(classification.grantable.operation_ids)
        assert resolved == classification.grantable.operations, command
        assert all(
            operation.operation_id in OPERATION_REGISTRY for operation in resolved
        ), command
        assert classification.grantable.consequence, command
        assert classification.grantable.call.subcommand, command


def test_reader_subcommands_set_inspect_git_repository() -> None:
    """Reader subcommands opt into repository-config inspection; others do not."""
    readers = _shell_git_support._GIT_READER_SUBCOMMANDS
    assert len(readers) >= 20
    for subcommand in sorted(readers):
        result = policy.analyze_shell_command_policy(["git", subcommand])
        assert result.inspect_git_repository is True, subcommand

    # Recognized non-readers stay out of the inspection flag.
    non_readers = _shell_git_support._GIT_NONREADER_SUBCOMMANDS
    assert len(non_readers) >= 20
    for subcommand in sorted(non_readers):
        result = policy.analyze_shell_command_policy(["git", subcommand])
        assert result.inspect_git_repository is False, subcommand

    # Unmodeled builtins that are not readers stay out too.
    unmodeled = _shell_git_support._GIT_UNMODELED_BUILTINS - readers
    assert unmodeled
    for subcommand in sorted(unmodeled):
        result = policy.analyze_shell_command_policy(["git", subcommand])
        assert result.inspect_git_repository is False, subcommand

    # Unrecognized, bare, and non-git invocations stay out as well.
    for tokens in (["git", "fixture-alias"], ["git"], ["ls"], []):
        result = policy.analyze_shell_command_policy(tokens)
        assert result.inspect_git_repository is False, tokens


# symbolic-ref is modeled by mode (v0.5.9 P2-d), never blanket-read: the
# read form (exactly one operand, no write marker) resolves like a reader;
# the write forms (a ref update or deletion) are not granted by the read
# modeling, and neither are bare or unscannable forms.
SYMBOLIC_REF_READ_COMMANDS = (
    "git symbolic-ref HEAD",
    "git symbolic-ref -q HEAD",
    "git symbolic-ref --short HEAD",
    "git symbolic-ref --recurse HEAD",
    "git symbolic-ref --no-recurse HEAD",
    "git -C . symbolic-ref HEAD",
    "git symbolic-ref -- HEAD",
)

SYMBOLIC_REF_WRITE_COMMANDS = (
    "git symbolic-ref HEAD refs/heads/main",
    "git symbolic-ref -m reason HEAD refs/heads/main",
    "git symbolic-ref --message reason HEAD refs/heads/main",
    "git symbolic-ref -m reason HEAD",
    "git symbolic-ref -d HEAD",
    "git symbolic-ref --delete HEAD",
    "git symbolic-ref -d -q HEAD",
    "git symbolic-ref HEAD refs/heads/main extra",
)

SYMBOLIC_REF_UNMODELED_COMMANDS = (
    "git symbolic-ref",
    "git symbolic-ref -q",
    "git symbolic-ref --recurse",
    "git symbolic-ref --bogus HEAD",
    "git symbolic-ref -x HEAD",
)


def test_symbolic_ref_is_modeled_not_table_classified() -> None:
    """symbolic-ref leaves the flat tables: its mode decides, not membership."""
    assert "symbolic-ref" not in _shell_git_support._GIT_UNMODELED_BUILTINS
    assert "symbolic-ref" not in _shell_git_support._GIT_READER_SUBCOMMANDS
    assert "symbolic-ref" not in _shell_git_support._GIT_NONREADER_SUBCOMMANDS


def test_symbolic_ref_read_forms_resolve_like_readers() -> None:
    """The read mode carries no dangerous operation and inspects the repo."""
    for command in SYMBOLIC_REF_READ_COMMANDS:
        classification = _classify(command)
        assert classification.kind == "routine", command
        assert classification.grantable is None, command
        result = policy.analyze_shell_command_policy(shlex.split(command))
        assert result.requires_approval is False, command
        assert result.inspect_git_repository is True, command
        assert result.positional_values == ("HEAD",), command


def test_symbolic_ref_write_forms_are_never_grantable() -> None:
    """The write mode is not granted by the read modeling: never run-class."""
    for command in SYMBOLIC_REF_WRITE_COMMANDS:
        classification = _classify(command)
        assert classification.kind == "hard_denied", command
        assert classification.grantable is None, command
        assert "write form" in (classification.reason or ""), command
        result = policy.analyze_shell_command_policy(shlex.split(command))
        assert result.requires_approval is True, command
        assert result.inspect_git_repository is False, command
        # The policy layer's denial text carries the classifier's reason.
        assert result.denial_reason is not None, command
        assert result.denial_reason.startswith(classification.reason or ""), command


def test_symbolic_ref_unmodeled_forms_fail_closed() -> None:
    """Bare, operand-less, and unscannable forms keep the form denial."""
    for command in SYMBOLIC_REF_UNMODELED_COMMANDS:
        classification = _classify(command)
        assert classification.kind == "hard_denied", command
        assert classification.grantable is None, command
        assert "write form" not in (classification.reason or ""), command
        result = policy.analyze_shell_command_policy(shlex.split(command))
        assert result.requires_approval is True, command
        assert result.denial_reason is not None, command
        assert result.denial_reason.startswith(classification.reason or ""), command

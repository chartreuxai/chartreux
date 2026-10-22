# Instructions and skills

Chartreux composes the agent's instructions from three selections — an
**identity**, a **workflow**, and an **orchestration strategy** — layered with
project instructions and optional skills. Project-provided material is loaded
only from trusted folders; see [Tools and safety](tools-safety.md). The
selection model, its precedence, and the frozen per-conversation lifecycle are
documented in the [identity architecture](../reference/identity-architecture.md).

## Project instructions: `AGENTS.md`

Use `~/.chartreux/AGENTS.md` for instructions that apply to your projects. Put `AGENTS.md` in a project directory for repository or subtree-specific guidance. Chartreux collects project files from the trust root toward the working directory, so more local instructions take precedence. Instructions attached to a file in a subdirectory are considered when that file is used.

`AGENTS.md` does not replace the composed agent instructions. Its content is appended as a higher-priority instructions section. This makes it suitable for repository conventions, required checks, and directory-scoped constraints.

## Custom identities, workflows, and strategies

The three selections resolve from shipped built-ins or from custom files in
your user directories:

| Resource | User directory | Selection |
| --- | --- | --- |
| Identity prompt | `~/.chartreux/prompts/` | `identity = "user:foo"` or bare `"foo"` → `prompts/foo.md` |
| Workflow body | `~/.chartreux/workflows/` | `workflow = "user:foo"` or bare `"foo"` → `workflows/foo.md` |
| Orchestration strategy | `~/.chartreux/orchestration_strategies/` | `orchestration_strategy = "user:foo"` or bare `"foo"` → `orchestration_strategies/foo.md` |

A new prompt file in `~/.chartreux/prompts/` becomes a selectable identity:
`identity = "my-reviewer"` (or `"user:my-reviewer"`) resolves to
`prompts/my-reviewer.md`. The prompt is the whole contract for a custom
identity — no metadata, no enforced posture, and no `extends` or inheritance
between identities.

```toml
[agent]
identity = "user:release-auditor"     # prompts/release-auditor.md
workflow = "ceremony"                 # built-in workflow
orchestration_strategy = "none"       # no strategy body
```

```toml
[agent]
workflow = "user:ship-it"             # workflows/ship-it.md
orchestration_strategy = "user:two-phase"  # orchestration_strategies/two-phase.md
```

New-conversation precedence is CLI flag (`--identity`, `--workflow`,
`--orchestration-strategy`) > environment (`CHARTREUX_AGENT__IDENTITY`,
`CHARTREUX_AGENT__WORKFLOW`, `CHARTREUX_AGENT__ORCHESTRATION_STRATEGY`) >
`config.toml` > shipped default, and the Settings > Identity & Orchestration
group edits the same keys. Selection **and content** freeze per conversation:
saved changes apply to new conversations only, and `/reload` reports them as
pending.

One strict loading contract governs all three kinds:

- Ids are plain file stems — letters, digits, `_`, and `-`, starting with a
  letter or digit. No traversal, nested, or extra-colon forms; malformed
  qualifications fail loudly.
- Files must be non-empty UTF-8 no larger than 64 KiB.
- `builtin:` and `user:` qualification must be well-formed.
- Reserved shipped names — the five built-in identities (`coder`,
  `orchestrator`, `researcher`, `reviewer`, `advisor`), the workflows
  (`minimal`, `ceremony`), and the strategy (`default`) — cannot be shadowed:
  a custom file reusing a shipped name is rejected with a rename instruction,
  never silently preferred.
- Custom files exist only in the user directories above. Project-level
  resource directories do not supply identities, workflows, or strategies;
  a project `.chartreux/prompts/` directory does not make a prompt selectable,
  and retired directories are detected and reported, never consumed.

Identity selection precedence: a bare built-in name, a `builtin:`-qualified
built-in, a `user:`-qualified custom id, or — for identity only, where the
prompt file is the whole contract — a bare custom name that resolves to a
prompt file in the user prompts directory. Unknown or malformed selections fail
at conversation start with actionable errors, and `chartreux doctor` resolves
them offline.

The retired `system_prompt_id` key no longer selects the composed prompt — the
`[agent]` selections do. Full-skeleton prompt replacement is replaced by a
custom identity plus workflow/strategy composition; `chartreux doctor` reports
the key with successor guidance.

## Custom compaction prompts

Compaction uses the built-in `compact` prompt by default; a custom id remains
available and resolves to a `.md` file in the user prompts directory
(`~/.chartreux/prompts/`), through the same strict loader as identities —
reserved built-in names cannot be shadowed, and project prompt directories no
longer supply prompts:

```toml
compaction_prompt_id = "project-summary"
```

Place `project-summary.md` in `~/.chartreux/prompts/`. Extra text passed to `/compact` is appended after the configured compaction prompt.

## Skills

A skill packages reusable instructions and workflow guidance. It can be discoverable by the model or invocable as a slash command, but it does not add tools. Chartreux reads the [Agent Skills specification](https://agentskills.io/specification).

Create one directory per skill with `SKILL.md`:

```markdown
---
name: code-review
description: Review a change for correctness, tests, and regressions.
user-invocable: true
allowed-tools:
  - read_file
  - grep
---

# Code review

Inspect the requested change, identify concrete findings, and report them by severity.
```

The `name` must be lowercase letters, numbers, and hyphens. `description` tells the agent when to use the skill. `user-invocable` defaults to `true`; set it to `false` for model-only skills. `allowed-tools` is experimental metadata for pre-approved tools, not a mechanism for adding tools or bypassing policy.

### Discovery and precedence

Chartreux first loads Python built-in skills, whose names are reserved and cannot be overridden. It then searches configured `skill_paths`, skills shipped with Chartreux, trusted project directories `.chartreux/skills/` and `.agents/skills/`, and finally `~/.chartreux/skills/` and `~/.agents/skills/`. Configured, project, and user skills shadow a shipped skill; among those local directories, the first definition of a name wins.

```toml
skill_paths = ["/work/shared/chartreux-skills"]
```

Manage the discovered set with exact names, globs, or `re:` regular expressions. If `enabled_skills` is nonempty, it is an allowlist; otherwise `disabled_skills` removes matches.

```toml
enabled_skills = ["code-review", "test-*"]
# disabled_skills = ["experimental-*"]
```

### Slash commands

A user-invocable skill becomes `/skill-name`. Text after the command is supplied as extra instructions, so `/code-review focus on error handling` loads the skill and adds that focus. Slash commands appear with built-in commands in completion.

For tool availability and permission controls, see [Tools and safety](tools-safety.md). For a specialized child agent instead of a reusable instruction bundle, see [Subagents](subagents.md).

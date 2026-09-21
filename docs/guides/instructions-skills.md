# Instructions and skills

Chartreux combines a selected system prompt with project instructions and optional skills. Project-provided material is loaded only from trusted folders; see [Tools and safety](tools-safety.md).

## Project instructions: `AGENTS.md`

Use `~/.chartreux/AGENTS.md` for instructions that apply to your projects. Put `AGENTS.md` in a project directory for repository or subtree-specific guidance. Chartreux collects project files from the trust root toward the working directory, so more local instructions take precedence. Instructions attached to a file in a subdirectory are considered when that file is used.

`AGENTS.md` does not replace the selected system prompt. Its content is appended as a higher-priority instructions section. This makes it suitable for repository conventions, required checks, and directory-scoped constraints.

## Custom system prompts

The built-in system prompt IDs are `cli`, `explore`, `tests`, and `minimal`. Select one, or select a custom bare filename:

```toml
system_prompt_id = "my-review-prompt"
```

Store custom prompt files as `my-review-prompt.md` in `.chartreux/prompts/` for a trusted project or in `~/.chartreux/prompts/` for the user. Project directories are searched before user directories, and custom prompt files are considered before built-in IDs; a local `cli.md`, for example, shadows the built-in `cli` prompt.

## Custom compaction prompts

Compaction uses the built-in `compact` prompt by default; custom prompt IDs remain available. Prompt lookup and precedence are the same as for system prompts.

```toml
compaction_prompt_id = "project-summary"
```

Place `project-summary.md` in one of the prompt directories above. Extra text passed to `/compact` is appended after the configured compaction prompt.

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

Chartreux searches configured `skill_paths`, then trusted project directories `.chartreux/skills/` and `.agents/skills/`, then `~/.chartreux/skills/` and `~/.agents/skills/`. The first discovered definition of a name wins; built-in skill names are reserved.

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

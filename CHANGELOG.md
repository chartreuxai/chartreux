# Changelog

All notable changes to Chartreux are documented in this file.

## 0.1.1 — Unreleased

Initial pre-release of Chartreux. This is the first Chartreux release, forked from
[Mistral Vibe](https://github.com/mistralai/mistral-vibe) v2.25.5; the changes
below describe Chartreux-specific differences rather than inherited capabilities.

### Added

- A provider-agnostic `models.toml` catalog with linked deployments, priority
  failover, model tags and roles, and per-model thinking-level collapse.
- Background subagents with non-blocking execution, fan-out, completion
  notifications, retention, reuse and retasking with per-run model, thinking,
  and permission overrides, plus transcript browsing and an agent sidebar in
  the TUI.
- Provider-configurable `web_search` and image-capable `read_image` tools.
- Fan-out launches runnable role members and reports unavailable or forbidden
  members as skipped.

### Changed

- Updated the shipped catalog to the gpt-6 lineup with published prices and
  compaction thresholds; raised glm-5-3's default thinking to high and removed
  the former medium worker and reviewer roles.
- Rebound the built-in reviewer profile to `small-reviewer` and reworked the
  main-review tiers around a blocking parallel Deep review fan-out.
- An independent, local-first harness: no sign-in/sign-out or provider accounts;
  provider API keys are supplied through `.env`, onboarding writes a default
  `config.toml`, and configuration is file-based rather than a settings UI.
- A simplified provider layer with generic OpenAI-style and Anthropic-style APIs,
  alongside Mistral and Codex/Responses adapters; Vertex and the reasoning adapter
  were removed.
- A Chartreux-inspired light/dark TUI palette.
- Tool permissions and the shared app-server boundary for the Textual CLI, ACP,
  and programmatic clients were reworked; Chartreux sends no product telemetry.

# Getting started

Chartreux is a local terminal coding harness. Run it from a project you trust so it can inspect and work with that project.

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- A current Mistral API key for the default provider

Linux is supported. macOS is not validated and Windows is unsupported.

## Install

Install the command-line tool directly from GitHub with uv:

```bash
uv tool install git+https://github.com/chartreuxai/chartreux
```

This installs `chartreux` (as well as `chartreux-acp` and
`chartreux-app-server`). Ensure `uv tool dir --bin` is on your `PATH`.

For development from a checkout, synchronize the project environment instead:

```bash
git clone https://github.com/chartreuxai/chartreux.git
cd chartreux
uv sync
uv run chartreux
```

## First launch

Change to the project you want to work in, then start Chartreux:

```bash
cd /path/to/project
chartreux
```

On its first run, Chartreux creates `~/.chartreux/config.toml` for your
selections; built-in defaults remain effective until you add overrides. It also
creates `~/.chartreux/.env`. Interactive onboarding presents a welcome and theme
selection, then uses the shared provider-management flow to select a provider
and enter its credentials; Mistral is the default route. It then probes the
provider for available chat models and lets you choose from a searchable list,
with already-configured models preselected and non-chat models such as embeddings
filtered out. Finally, select the active model; the current selection is
preselected, and `Default` is available when none is set. Entered credentials are
stored in that `.env` file. You can run the onboarding explicitly with:

```bash
chartreux --setup
```

A non-empty `MISTRAL_API_KEY` already present in the process environment takes
precedence over the value in `.env`. You can set it in the shell instead of
using onboarding:

```bash
export MISTRAL_API_KEY="your-api-key"
```

## Next steps

Type a request at the interactive prompt, for example: `Find the test that
covers this function.` For terminal interaction, sessions, and workspace
options, see the [terminal guide](guides/terminal.md). For configuration and
model selection, see the [configuration reference](reference/configuration.md).

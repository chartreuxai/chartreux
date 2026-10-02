# Getting started

Chartreux is a local terminal coding harness. Run it from a project you trust so it can inspect and work with that project.

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- An API key for a provider you want to use, or access to a keyless provider

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

On its first run, Chartreux creates `~/.chartreux/config.toml` for ordinary
settings and `~/.chartreux/.env` for credentials. The saved main and subagent
presets live in `~/.chartreux/models.toml`. Interactive onboarding presents a
welcome, then follows **Connect provider → Configure models → Add another
provider or continue → Choose default presets → [optional Web search] →
Finish**. Mistral is the initial model-provider route. Discovery filters out
non-chat models such as embeddings; you can also configure a model manually.
Each provider and model save advances without requiring a return to an earlier
screen. Presets assign
one model and thinking level to the main assistant and each subagent role.
Ready preset choices stay selected. When a current preset is not runnable,
setup suggests a model from a configured provider only when its deployment is
enabled, its supported thinking level is valid, and the credential resolver
finds any required key. Saving presets persists that choice. If web search is
already ready, onboarding preserves the current search choice and skips that
step. Otherwise the optional Web search step offers Exa, Brave, and DuckDuckGo;
it does not offer `auto` or a second Mistral choice. Standalone Settings shows
one Mistral choice; `auto` remains a supported configuration alias for Mistral.
The Web search step lets you save settings and finish, go back to presets, or
choose **Skip for now**. Skipping leaves web-search settings unchanged and does
not disable the tool. Finishing checks required presets and, if the search
step is shown, web-search configuration and credentials; it does not test live
connectivity. Entered
credentials are saved in the `.env` file when possible; setup reports if a key
is available only for the current session. You can run onboarding explicitly
with:

```bash
chartreux --setup
```

A non-empty `MISTRAL_API_KEY` already present in the process environment takes
precedence over the value in `.env`. You can set it in the shell instead of
using onboarding:

```bash
export MISTRAL_API_KEY="your-api-key"
```

## Check setup

Run `chartreux doctor` from the project directory to validate local
configuration and model selection without network requests or subprocesses.
Bare doctor does not read `.env` or keyring credentials, so an `unverified`
credential is not necessarily missing. For opt-in live checks and billable
provider smoke probes, see [Troubleshooting](guides/troubleshooting.md#run-diagnostics).

## Next steps

Type a request at the interactive prompt, for example: `Find the test that
covers this function.` For terminal interaction, sessions, and workspace
options, see the [terminal guide](guides/terminal.md). For configuration and
model selection, see the [configuration reference](reference/configuration.md).

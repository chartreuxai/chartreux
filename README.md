# Chartreux

Chartreux is an oppinionated independent fork of [Mistral Vibe](https://github.com/mistralai/mistral-vibe).

Chartreux is pre-release alpha software with no stability guarantees; breaking changes may occur without migration paths.

```
  ▄▄                ▄▄
  ███▄▄▄        ▄▄▄███
  ████████████████████
  ████████████████████
 ██████████████████████
█████▀▀▀█▀████▀▀▀█▀█████
█████   █ ████   █ █████
████████████████████████
▀▄▄▄███████  ███████▄▄▄▀
 ▄▄▄█████▄▄▄▄▄▄█████▄▄▄
   ▄▄██████████████▄▄
      ▀▀▀▀████▀▀▀▀
```

## Highlights

- Async subagents with retention, reuse, and retasking
- Provider-agnostic model catalog with deployment failover
- Local-first sessions and workspace support
- Skills, hooks, and MCP extensibility
- Credentials loaded from `.env` files or the process environment
- Light, dark, and automatic terminal themes

## Install

Chartreux requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/chartreuxai/chartreux
```

For development from a checkout:

```bash
git clone https://github.com/chartreuxai/chartreux.git
cd chartreux
uv sync
```

Ensure `uv tool dir --bin` is on your `PATH`.

## Launch

From a project you trust:

```bash
chartreux
```

Configuration is stored in `~/.chartreux/config.toml`; provider credentials can be stored in `~/.chartreux/.env`. See the [configuration guide](https://chartreuxai.github.io/chartreux/docs/guides/configuration/).

## Documentation

- [Getting started](https://chartreuxai.github.io/chartreux/docs/getting-started/)
- [Guides](https://chartreuxai.github.io/chartreux/docs/guides/)
- [Reference](https://chartreuxai.github.io/chartreux/docs/reference/)
- [Architecture decision records](https://chartreuxai.github.io/chartreux/docs/adr/)
- [Full documentation](https://chartreuxai.github.io/chartreux/docs/)

## Contributing

This is a personal fork for my own use. Bug reports, Pull requests and even feature requests are wellcome, but any kind of entitled behaviour isn't. This project is for my own needs, you are encouraged to fork it if you don't like how I run things, it's easy to maintain a fork with LLMs these days.

### Running tests

Run the suite with `uv run pytest`; `pyproject.toml` configures xdist parallelism. Avoid `-o addopts=''`: it disables the configured addopts, including xdist parallelism. The installed-contract tests build a wheel and sdist and install them offline, so set `UV_FIND_LINKS` to a pre-provisioned local wheelhouse.

## License

Chartreux is licensed under [Apache-2.0](LICENSE) and is a fork of Mistral Vibe. See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for attribution and [CHANGELOG.md](CHANGELOG.md) for changes.

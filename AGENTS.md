Chartreux is an independent fork of Mistral Vibe.

## Development cycle

- `main` is the release trunk: one commit per release, tagged `vX.Y.Z`. No direct commits to `main` (the v0.2.0 release commit is the one manual exception).
- `development` is the integration branch: all work happens on feature branches cut from `development` and is squash-merged into `development`.
- Release procedure: on `development`, bump the version in `pyproject.toml` and finalize the CHANGELOG entry; squash-merge `development` into `main`; tag the release commit on `main`; push `main` and the tag; continue `development` from the merged state.
- The project is pre-release alpha: no stability guarantees, breaking changes allowed without migration paths.

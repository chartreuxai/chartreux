"""Shared fixtures for the WP6 differential replay skeleton (WP6 [6A]).

Later differential traces ([6B]-[6D]) take their seed and script from here so
one campaign seed drives every topology comparison.
"""

from __future__ import annotations

import pytest

from tests.differential.replay import ReplayScript, build_replay_script

DEFAULT_REPLAY_SEED = 20_261_110


@pytest.fixture
def replay_seed() -> int:
    return DEFAULT_REPLAY_SEED


@pytest.fixture
def replay_script(replay_seed: int) -> ReplayScript:
    return build_replay_script(replay_seed)

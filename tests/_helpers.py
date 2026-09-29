"""Shared helpers for the offline regression suite.

Everything in `tests/` runs without any LLM call: the point is to lock down the
"stage -1" safety/determinism fixes so they cannot silently regress.

`DataManager` resolves every path relative to the process working directory
(`data/<world>/persona/<char>/...`), so each test runs inside a throwaway
temporary directory and never touches real run data.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

# Imported while the process still runs from the repository root on purpose:
# prompt/config modules load `data/<world>/worldview.json` relative to the CWD
# at import time, so importing them after a chdir would fail.
from src.agents.data_manager import DataManager
from src.world.clock import Clock, Stage

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_WORLD = "regression_world"
TEST_CHAR = "测试角色"


@contextlib.contextmanager
def temp_workspace() -> Iterator[Path]:
    """Yield a fresh temporary working directory, restoring the old one after."""
    prev = os.getcwd()
    tmp = Path(tempfile.mkdtemp(prefix="agentopia-regression-"))
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(prev)
        shutil.rmtree(tmp, ignore_errors=True)


def make_datamanager(
    char: str = TEST_CHAR,
    world: str = TEST_WORLD,
    *,
    year: int = 2020,
    week: int = 1,
    stage: Stage = Stage.BEGIN,
    clock: Clock | None = None,
) -> Tuple[DataManager, Clock]:
    """Build a DataManager (and its clock) inside the current working directory.

    Pass an existing `clock` to model several personas that share one world
    clock, which is how `World` runs them.
    """
    if clock is None:
        clock = Clock(start_year=year, start_week=week)
        if stage != Stage.BEGIN:
            clock.set_stage(stage)
    dm = DataManager(char=char, world=world, clock=clock, model="regression-model")
    return dm, clock


def persona_root(char: str = TEST_CHAR, world: str = TEST_WORLD) -> Path:
    return Path("data") / world / "persona" / char

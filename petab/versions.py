"""Handling of PEtab version numbers."""
from __future__ import annotations

from pathlib import Path

import petab
from petab.v1 import Problem as V1Problem
from petab.v1.C import FORMAT_VERSION
from petab.v1.yaml import load_yaml

__all__ = [
    "get_major_version",
]


def get_major_version(
    problem: str | dict | Path | V1Problem | petab.v2.Problem,
) -> int:
    """Get the major version number of the given problem."""
    version = None

    if isinstance(problem, str | Path):
        yaml_config = load_yaml(problem)
        version = yaml_config.get(FORMAT_VERSION)
    elif isinstance(problem, dict):
        version = problem.get(FORMAT_VERSION)

    if version is not None:
        version = str(version)
        return int(version.split(".")[0])

    if isinstance(problem, V1Problem):
        return 1

    from . import v2

    if isinstance(problem, v2.Problem):
        return 2

    raise ValueError(f"Unsupported argument type: {type(problem)}")

"""Handling of PEtab version numbers."""
from __future__ import annotations

from pathlib import Path

from petab.C import FORMAT_VERSION
from petab.v1 import Problem as V1Problem
from petab.v1.yaml import load_yaml
from petab.v2 import Problem as V2Problem

__all__ = [
    "get_major_version",
]


def get_major_version(
    problem: str | dict | Path | V1Problem | V2Problem,
) -> int:
    """Get the major version number of the given problem."""
    if isinstance(problem, V1Problem):
        return 1

    if isinstance(problem, V2Problem):
        return 2

    if isinstance(problem, str | Path):
        yaml_config = load_yaml(problem)
        version = yaml_config.get(FORMAT_VERSION)
    elif isinstance(problem, dict):
        version = problem.get(FORMAT_VERSION)
    else:
        raise ValueError(f"Unsupported argument type: {type(problem)}")

    version = str(version)
    return int(version.split(".")[0])

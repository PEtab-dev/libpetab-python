"""Handling of PEtab version numbers."""
from __future__ import annotations

from pathlib import Path

from petab.C import FORMAT_VERSION
from petab.v1 import Problem as V1Problem
from petab.v1.yaml import load_yaml
from petab.v2 import Problem as V2Problem

__all__ = [
    "is_v1_problem",
    "is_v2_problem",
]


def is_v1_problem(problem: str | dict | Path | V1Problem | V2Problem) -> bool:
    """Check if the given problem is a PEtab v1 problem."""
    if isinstance(problem, V1Problem):
        return True

    if isinstance(problem, V2Problem):
        return False

    if isinstance(problem, str | Path):
        yaml_config = load_yaml(problem)
        version = yaml_config.get(FORMAT_VERSION)
    elif isinstance(problem, dict):
        version = problem.get(FORMAT_VERSION)
    else:
        raise ValueError(f"Unsupported argument type: {type(problem)}")

    version = str(version)
    if version.startswith("1.") or version == "1":
        return True


def is_v2_problem(problem: str | dict | Path | V1Problem | V2Problem) -> bool:
    """Check if the given problem is a PEtab v2 problem."""
    if isinstance(problem, V1Problem):
        return False

    if isinstance(problem, V2Problem):
        return True

    if isinstance(problem, str | Path):
        yaml_config = load_yaml(problem)
        version = yaml_config.get(FORMAT_VERSION)
    elif isinstance(problem, dict):
        version = problem.get(FORMAT_VERSION)
    else:
        raise ValueError(f"Unsupported argument type: {type(problem)}")

    version = str(version)
    if version.startswith("2.") or version == "2":
        return True

"""Handling of PEtab version numbers."""

from __future__ import annotations

import re
from pathlib import Path

import petab

__all__ = [
    "get_major_version",
    "parse_version",
]
from . import v1

# version regex pattern
_version_pattern = (
    r"(?P<major>\d+)(?:\.(?P<minor>\d+))?"
    r"(?:\.(?P<patch>\d+))?(?P<suffix>[\w.]+)?"
)
_version_re = re.compile(_version_pattern)


def parse_version(version: str | int) -> tuple[int, int, int, str]:
    """Parse a version string into a tuple of integers and suffix."""
    if isinstance(version, int):
        return version, 0, 0, ""

    version = str(version)
    match = _version_re.match(version)
    if match is None:
        raise ValueError(f"Invalid version string: {version}")

    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    patch = int(match.group("patch") or 0)
    suffix = match.group("suffix") or ""

    return major, minor, patch, suffix


def get_major_version(
    problem: str | dict | Path | petab.v1.Problem | petab.v2.Problem,
) -> int:
    """Get the major version number of the given problem."""
    version = None

    if isinstance(problem, str | Path):
        from petab.v1.yaml import load_yaml

        yaml_config = load_yaml(problem)
        version = yaml_config.get(v1.C.FORMAT_VERSION)
    elif isinstance(problem, dict):
        version = problem.get(v1.C.FORMAT_VERSION)

    if version is not None:
        version = str(version)
        return int(version.split(".")[0])

    if isinstance(problem, petab.v1.Problem):
        return 1

    from . import v2

    if isinstance(problem, v2.Problem):
        return 2

    raise ValueError(f"Unsupported argument type: {type(problem)}")

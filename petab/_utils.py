"""Private, version-independent utility functions for PEtab."""

from pathlib import Path

from pydantic import AnyUrl, TypeAdapter

PathOrUrlAdapter = TypeAdapter(AnyUrl | Path)


def _generate_path(
    file_path: str | Path | AnyUrl,
    base_path: Path | str | AnyUrl | None = None,
) -> str:
    if base_path is None:
        return file_path

    file_path = PathOrUrlAdapter.validate_python(file_path)
    if isinstance(file_path, AnyUrl):
        # if URL, this is absolute
        return str(file_path)

    base_path = PathOrUrlAdapter.validate_python(base_path)
    if isinstance(base_path, Path):
        # if file_path is absolute, base_path will be ignored
        return str(base_path / file_path)

    # combine URL parts
    return f"{base_path}/{file_path}"

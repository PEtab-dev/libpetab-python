"""Base classes shared across petab.v2 to avoid circular imports."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, get_args

import pandas as pd
from pydantic import AnyUrl, BaseModel, Field

from .._utils import _generate_path

if TYPE_CHECKING:
    from .core import Problem

logger = logging.getLogger(__name__)


class ValidationIssueSeverity(IntEnum):
    """The severity of a validation issue."""

    INFO = 10
    WARNING = 20
    ERROR = 30
    CRITICAL = 40


@dataclass
class ValidationIssue:
    """The result of a validation task."""

    level: ValidationIssueSeverity
    message: str
    task: str | None = None

    def __post_init__(self):
        if not isinstance(self.level, ValidationIssueSeverity):
            raise TypeError(
                "`level` must be an instance of ValidationIssueSeverity."
            )

    def __str__(self):
        return f"{self.level.name}: {self.message}"

    @staticmethod
    def _get_task_name() -> str | None:
        """Get the name of the ValidationTask that raised this error."""
        import inspect

        for frame_info in inspect.stack():
            frame = frame_info.frame
            if "self" in frame.f_locals:
                task = frame.f_locals["self"]
                if isinstance(task, ValidationTask):
                    return task.__class__.__name__
        return None


@dataclass
class ValidationError(ValidationIssue):
    """A validation result with level ERROR."""

    level: ValidationIssueSeverity = field(
        default=ValidationIssueSeverity.ERROR, init=False
    )

    def __post_init__(self):
        if self.task is None:
            self.task = self._get_task_name()


@dataclass
class ValidationWarning(ValidationIssue):
    """A validation result with level WARNING."""

    level: ValidationIssueSeverity = field(
        default=ValidationIssueSeverity.WARNING, init=False
    )

    def __post_init__(self):
        if self.task is None:
            self.task = self._get_task_name()


class ValidationResultList(list):
    """A list of validation results."""

    def log(
        self,
        *,
        logger: logging.Logger = logger,
        min_level: ValidationIssueSeverity = ValidationIssueSeverity.INFO,
        max_level: ValidationIssueSeverity = ValidationIssueSeverity.CRITICAL,
    ):
        """Log the validation results."""
        for result in self:
            if result.level < min_level or result.level > max_level:
                continue
            msg = f"{result.level.name}: {result.message} [{result.task}]"
            if result.level == ValidationIssueSeverity.INFO:
                logger.info(msg)
            elif result.level == ValidationIssueSeverity.WARNING:
                logger.warning(msg)
            elif result.level >= ValidationIssueSeverity.ERROR:
                logger.error(msg)

        if not self:
            logger.info("PEtab format check completed successfully.")

    def has_errors(self) -> bool:
        """Check if there are any errors in the validation results."""
        return any(
            result.level >= ValidationIssueSeverity.ERROR for result in self
        )


class ValidationTask(ABC):
    """A task to validate a PEtab problem."""

    @abstractmethod
    def run(self, problem: Problem) -> ValidationIssue | None:
        """Run the validation task."""
        ...

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


T = TypeVar("T", bound=BaseModel)


class BaseTable(BaseModel, Generic[T]):
    """Base class for PEtab tables."""

    #: The table elements
    elements: list[T]
    #: The path to the table file, if applicable.
    #: Relative to the base path, if the base path is set and rel_path is not
    #: an absolute path.
    rel_path: AnyUrl | Path | None = Field(exclude=True, default=None)
    #: The base path for the table file, if applicable.
    #: This is usually the directory of the PEtab YAML file.
    base_path: AnyUrl | Path | None = Field(exclude=True, default=None)

    def __init__(self, elements: list[T] = None, **kwargs) -> None:
        """Initialize the BaseTable with a list of elements."""
        if elements is None:
            elements = []
        super().__init__(elements=elements, **kwargs)

    def __getitem__(self, id_: str) -> T:
        """Get an element by ID.

        :param id_: The ID of the element to retrieve.
        :return: The element with the given ID.
        :raises KeyError: If no element with the given ID exists.
        :raises NotImplementedError:
            If the element type does not have an ID attribute.
        """
        if "id" not in self._element_class().model_fields:
            raise NotImplementedError(
                f"__getitem__ is not implemented for {self.__class__.__name__}"
            )

        for element in self.elements:
            if element.id == id_:
                return element

        raise KeyError(f"{T.__name__} ID {id_} not found")

    @classmethod
    @abstractmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> BaseTable[T]:
        """Create a table from a DataFrame."""
        pass

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        """Convert the table to a DataFrame."""
        pass

    @classmethod
    def from_tsv(
        cls, file_path: str | Path, base_path: str | Path | None = None
    ) -> BaseTable[T]:
        """Create table from a TSV file."""
        df = pd.read_csv(_generate_path(file_path, base_path), sep="\t")
        return cls.from_df(df, rel_path=file_path, base_path=base_path)

    def to_tsv(self, file_path: str | Path = None) -> None:
        """Write the table to a TSV file."""
        df = self.to_df()
        df.to_csv(
            file_path or _generate_path(self.rel_path, self.base_path),
            sep="\t",
            index=not isinstance(df.index, pd.RangeIndex),
        )

    @classmethod
    def _element_class(cls) -> type[T]:
        """Get the class of the elements in the table."""
        return get_args(cls.model_fields["elements"].annotation)[0]

    def __add__(self, other: T) -> BaseTable[T]:
        """Add an item to the table."""
        if not isinstance(other, self._element_class()):
            raise TypeError(
                f"Can only add {self._element_class().__name__} "
                f"to {self.__class__.__name__}"
            )
        return self.__class__(elements=self.elements + [other])

    def __iadd__(self, other: T) -> BaseTable[T]:
        """Add an item to the table in place."""
        if not isinstance(other, self._element_class()):
            raise TypeError(
                f"Can only add {self._element_class().__name__} "
                f"to {self.__class__.__name__}"
            )
        self.elements.append(other)
        return self

"""Types around the PEtab object model."""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import traceback
from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum
from itertools import chain
from math import nan
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Self,
    TypeVar,
    get_args,
)

import numpy as np
import pandas as pd
import sympy as sp
from pydantic import (
    AfterValidator,
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from .._utils import _generate_path
from ..v1 import (
    validate_yaml_syntax,
    yaml,
)
from ..v1.distributions import *
from ..v1.lint import is_valid_identifier
from ..v1.math import petab_math_str, sympify_petab
from ..v1.models.model import Model, model_factory
from ..v1.yaml import get_path_prefix
from ..versions import parse_version
from . import C, get_observable_df

if TYPE_CHECKING:
    from ..v2.lint import ValidationResultList, ValidationTask


__all__ = [
    "Problem",
    "ProblemConfig",
    "Observable",
    "ObservableTable",
    "NoiseDistribution",
    "Change",
    "Condition",
    "ConditionTable",
    "ExperimentPeriod",
    "Experiment",
    "ExperimentTable",
    "Measurement",
    "MeasurementTable",
    "Mapping",
    "MappingTable",
    "Parameter",
    "ParameterScale",
    "ParameterTable",
    "PriorDistribution",
]

logger = logging.getLogger(__name__)


def _is_finite_or_neg_inf(v: float, info: ValidationInfo) -> float:
    if not np.isfinite(v) and v != -np.inf:
        raise ValueError(
            f"{info.field_name} value must be finite or -inf but got {v}"
        )
    return v


def _is_finite_or_pos_inf(v: float, info: ValidationInfo) -> float:
    if not np.isfinite(v) and v != np.inf:
        raise ValueError(
            f"{info.field_name} value must be finite or inf but got {v}"
        )
    return v


def _not_nan(v: float, info: ValidationInfo) -> float:
    if np.isnan(v):
        raise ValueError(f"{info.field_name} value must not be nan.")
    return v


def _convert_nan_to_none(v):
    """Convert NaN or "" to None."""
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str) and v == "":
        return None
    return v


def _valid_petab_id(v: str) -> str:
    """Field validator for PEtab IDs."""
    if not v:
        raise ValueError("ID must not be empty.")
    if not is_valid_identifier(v):
        raise ValueError(f"Invalid ID: {v}")
    return v


def _valid_petab_id_or_none(v: str) -> str | None:
    """Field validator for optional PEtab IDs."""
    if not v:
        return None
    if not is_valid_identifier(v):
        raise ValueError(f"Invalid ID: {v}")
    return v


class ParameterScale(str, Enum):
    """Parameter scales.

    Parameter scales as used in the PEtab parameter table.
    """

    LIN = C.LIN
    LOG = C.LOG
    LOG10 = C.LOG10


class NoiseDistribution(str, Enum):
    """Noise distribution types.

    Noise distributions as used in the PEtab observable table.
    """

    #: Normal distribution
    NORMAL = C.NORMAL
    #: Laplace distribution
    LAPLACE = C.LAPLACE
    #: Log-normal distribution
    LOG_NORMAL = C.LOG_NORMAL
    #: Log-Laplace distribution
    LOG_LAPLACE = C.LOG_LAPLACE


class PriorDistribution(str, Enum):
    """Prior types.

    Prior types as used in the PEtab parameter table.
    """

    #: Cauchy distribution.
    CAUCHY = C.CAUCHY
    #: Chi-squared distribution.
    CHI_SQUARED = C.CHI_SQUARED
    #: Exponential distribution.
    EXPONENTIAL = C.EXPONENTIAL
    #: Gamma distribution.
    GAMMA = C.GAMMA
    #: Laplace distribution.
    LAPLACE = C.LAPLACE
    #: Log-Laplace distribution
    LOG_LAPLACE = C.LOG_LAPLACE
    #: Log-normal distribution.
    LOG_NORMAL = C.LOG_NORMAL
    #: Log-uniform distribution.
    LOG_UNIFORM = C.LOG_UNIFORM
    #: Normal distribution.
    NORMAL = C.NORMAL
    #: Rayleigh distribution.
    RAYLEIGH = C.RAYLEIGH
    #: Uniform distribution.
    UNIFORM = C.UNIFORM


assert set(C.PRIOR_DISTRIBUTIONS) == {e.value for e in PriorDistribution}, (
    "PriorDistribution enum does not match C.PRIOR_DISTRIBUTIONS "
    f"{set(C.PRIOR_DISTRIBUTIONS)} vs { {e.value for e in PriorDistribution} }"
)

_prior_to_cls = {
    PriorDistribution.CAUCHY: Cauchy,
    PriorDistribution.CHI_SQUARED: ChiSquare,
    PriorDistribution.EXPONENTIAL: Exponential,
    PriorDistribution.GAMMA: Gamma,
    PriorDistribution.LAPLACE: Laplace,
    PriorDistribution.LOG_LAPLACE: Laplace,
    PriorDistribution.LOG_NORMAL: Normal,
    PriorDistribution.LOG_UNIFORM: LogUniform,
    PriorDistribution.NORMAL: Normal,
    PriorDistribution.RAYLEIGH: Rayleigh,
    PriorDistribution.UNIFORM: Uniform,
}

assert not (_mismatch := set(PriorDistribution) ^ set(_prior_to_cls)), (
    "PriorDistribution enum does not match _prior_to_cls. "
    f"Mismatches: {_mismatch}"
)


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


class Observable(BaseModel):
    """Observable definition."""

    #: Observable ID.
    id: Annotated[str, AfterValidator(_valid_petab_id)] = Field(
        alias=C.OBSERVABLE_ID
    )
    #: Observable name.
    name: str | None = Field(alias=C.OBSERVABLE_NAME, default=None)
    #: Observable formula.
    formula: sp.Basic | None = Field(alias=C.OBSERVABLE_FORMULA, default=None)
    #: Noise formula.
    noise_formula: sp.Basic | None = Field(alias=C.NOISE_FORMULA, default=None)
    #: Noise distribution.
    noise_distribution: NoiseDistribution = Field(
        alias=C.NOISE_DISTRIBUTION, default=NoiseDistribution.NORMAL
    )
    #: Placeholder symbols for the observable formula.
    observable_placeholders: list[sp.Symbol] = Field(
        alias=C.OBSERVABLE_PLACEHOLDERS, default=[]
    )
    #: Placeholder symbols for the noise formula.
    noise_placeholders: list[sp.Symbol] = Field(
        alias=C.NOISE_PLACEHOLDERS, default=[]
    )

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
    )

    @field_validator(
        "name",
        "formula",
        "noise_formula",
        "noise_distribution",
        mode="before",
    )
    @classmethod
    def _convert_nan_to_default(cls, v, info: ValidationInfo):
        if isinstance(v, float) and np.isnan(v):
            return cls.model_fields[info.field_name].default
        return v

    @field_validator("formula", "noise_formula", mode="before")
    @classmethod
    def _sympify(cls, v):
        if v is None or isinstance(v, sp.Basic):
            return v
        if isinstance(v, float) and np.isnan(v):
            return None

        return sympify_petab(v)

    @field_validator(
        "observable_placeholders", "noise_placeholders", mode="before"
    )
    @classmethod
    def _sympify_id_list(cls, v):
        if v is None:
            return []

        if isinstance(v, float) and np.isnan(v):
            return []

        if isinstance(v, str):
            v = v.split(C.PARAMETER_SEPARATOR)
        elif not isinstance(v, Sequence):
            v = [v]

        v = [pid.strip() for pid in v]
        return [sympify_petab(_valid_petab_id(pid)) for pid in v if pid]


class ObservableTable(BaseTable[Observable]):
    """PEtab observable table."""

    @property
    def observables(self) -> list[Observable]:
        """List of observables."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> ObservableTable:
        """Create an ObservableTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        df = get_observable_df(df)
        observables = [
            Observable(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]
        return cls(observables, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the ObservableTable to a DataFrame."""
        records = self.model_dump(by_alias=True)["elements"]
        for record in records:
            obs = record[C.OBSERVABLE_FORMULA]
            noise = record[C.NOISE_FORMULA]
            record[C.OBSERVABLE_FORMULA] = petab_math_str(obs)
            record[C.NOISE_FORMULA] = petab_math_str(noise)
            record[C.OBSERVABLE_PLACEHOLDERS] = C.PARAMETER_SEPARATOR.join(
                map(str, record[C.OBSERVABLE_PLACEHOLDERS])
            )
            record[C.NOISE_PLACEHOLDERS] = C.PARAMETER_SEPARATOR.join(
                map(str, record[C.NOISE_PLACEHOLDERS])
            )
        return pd.DataFrame(records).set_index([C.OBSERVABLE_ID])


class Change(BaseModel):
    """A change to the model or model state.

    A change to the model or model state, corresponding to an individual
    row of the PEtab condition table.

    >>> Change(
    ...     target_id="k1",
    ...     target_value="10",
    ... )  # doctest: +NORMALIZE_WHITESPACE
    Change(target_id='k1', target_value=10.0000000000000)
    """

    #: The ID of the target entity to change.
    target_id: Annotated[str, AfterValidator(_valid_petab_id)] = Field(
        alias=C.TARGET_ID
    )
    #: The value to set the target entity to.
    target_value: sp.Basic = Field(alias=C.TARGET_VALUE)

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        extra="allow",
        validate_assignment=True,
    )

    @field_validator("target_value", mode="before")
    @classmethod
    def _sympify(cls, v):
        if v is None or isinstance(v, sp.Basic):
            return v
        if isinstance(v, float) and np.isnan(v):
            return None

        return sympify_petab(v)


class Condition(BaseModel):
    """A set of changes to the model or model state.

    A set of simultaneously occurring changes to the model or model state,
    corresponding to a perturbation of the underlying system. This corresponds
    to all rows of the PEtab condition table with the same condition ID.

    >>> Condition(
    ...     id="condition1",
    ...     changes=[
    ...         Change(
    ...             target_id="k1",
    ...             target_value="10",
    ...         )
    ...     ],
    ... )  # doctest: +NORMALIZE_WHITESPACE
    Condition(id='condition1',
    changes=[Change(target_id='k1', target_value=10.0000000000000)])
    """

    #: The condition ID.
    id: Annotated[str, AfterValidator(_valid_petab_id)] = Field(
        alias=C.CONDITION_ID
    )
    #: The changes associated with this condition.
    changes: list[Change]

    #: :meta private:
    model_config = ConfigDict(
        populate_by_name=True, extra="allow", validate_assignment=True
    )

    def __add__(self, other: Change) -> Condition:
        """Add a change to the set."""
        if not isinstance(other, Change):
            raise TypeError("Can only add Change to Condition")
        return Condition(id=self.id, changes=self.changes + [other])

    def __iadd__(self, other: Change) -> Condition:
        """Add a change to the set in place."""
        if not isinstance(other, Change):
            raise TypeError("Can only add Change to Condition")
        self.changes.append(other)
        return self


class ConditionTable(BaseTable[Condition]):
    """PEtab condition table."""

    @property
    def conditions(self) -> list[Condition]:
        """List of conditions."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> ConditionTable:
        """Create a ConditionTable from a DataFrame."""
        if df is None or df.empty:
            return cls(**kwargs)

        conditions = []
        for condition_id, sub_df in df.groupby(C.CONDITION_ID):
            changes = [Change(**row) for row in sub_df.to_dict("records")]
            conditions.append(Condition(id=condition_id, changes=changes))

        return cls(conditions, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the ConditionTable to a DataFrame."""
        records = [
            {C.CONDITION_ID: condition.id, **change.model_dump(by_alias=True)}
            for condition in self.conditions
            for change in condition.changes
        ]
        for record in records:
            record[C.TARGET_VALUE] = (
                float(record[C.TARGET_VALUE])
                if record[C.TARGET_VALUE].is_number
                else str(record[C.TARGET_VALUE])
            )
        return (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=C.CONDITION_DF_REQUIRED_COLS)
        )

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        """Get all free symbols in the condition table.

        This includes all free symbols in the target values of the changes,
        independently of whether it is referenced by any experiment, or
        (indirectly) by any measurement.
        """
        return set(
            chain.from_iterable(
                change.target_value.free_symbols
                for condition in self.conditions
                for change in condition.changes
                if change.target_value is not None
            )
        )


class ExperimentPeriod(BaseModel):
    """A period of a timecourse or experiment defined by a start time
    and a list of condition IDs.

    This corresponds to a row of the PEtab experiment table.
    """

    #: The start time of the period in time units as defined in the model.
    time: Annotated[float, AfterValidator(_is_finite_or_neg_inf)] = Field(
        alias=C.TIME
    )
    #: The IDs of the conditions to be applied at the start time.
    condition_ids: list[str] = Field(default_factory=list)

    #: :meta private:
    model_config = ConfigDict(
        populate_by_name=True, extra="allow", validate_assignment=True
    )

    @field_validator("condition_ids", mode="before")
    @classmethod
    def _validate_ids(cls, condition_ids):
        if condition_ids in [None, "", [], [""]]:
            # unspecified, or "use-model-as-is"
            return []

        for condition_id in condition_ids:
            # The empty condition ID for "use-model-as-is" has been handled
            #  above. Having a combination of empty and non-empty IDs is an
            #  error, since the targets of conditions to be combined must be
            #  disjoint.
            if not is_valid_identifier(condition_id):
                raise ValueError(f"Invalid {C.CONDITION_ID}: `{condition_id}'")
        return condition_ids

    @property
    def is_preequilibration(self) -> bool:
        """Check if this period is a preequilibration period."""
        return self.time == C.TIME_PREEQUILIBRATION


class Experiment(BaseModel):
    """An experiment or a timecourse defined by an ID and a set of different
    periods.

    Corresponds to a group of rows of the PEtab experiment table with the same
    experiment ID.
    """

    #: The experiment ID.
    id: Annotated[str, AfterValidator(_valid_petab_id)] = Field(
        alias=C.EXPERIMENT_ID
    )
    #: The periods of the experiment.
    periods: list[ExperimentPeriod] = []

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
    )

    def __add__(self, other: ExperimentPeriod) -> Experiment:
        """Add a period to the experiment."""
        if not isinstance(other, ExperimentPeriod):
            raise TypeError("Can only add ExperimentPeriod to Experiment")
        return Experiment(id=self.id, periods=self.periods + [other])

    def __iadd__(self, other: ExperimentPeriod) -> Experiment:
        """Add a period to the experiment in place."""
        if not isinstance(other, ExperimentPeriod):
            raise TypeError("Can only add ExperimentPeriod to Experiment")
        self.periods.append(other)
        return self

    @property
    def has_preequilibration(self) -> bool:
        """Check if the experiment has preequilibration enabled."""
        return any(period.is_preequilibration for period in self.periods)

    @property
    def sorted_periods(self) -> list[ExperimentPeriod]:
        """Get the periods of the experiment sorted by time."""
        return sorted(self.periods, key=lambda period: period.time)

    def sort_periods(self) -> None:
        """Sort the periods of the experiment by time."""
        self.periods.sort(key=lambda period: period.time)


class ExperimentTable(BaseTable[Experiment]):
    """PEtab experiment table."""

    @property
    def experiments(self) -> list[Experiment]:
        """List of experiments."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> ExperimentTable:
        """Create an ExperimentTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        experiments = []
        for experiment_id, cur_exp_df in df.groupby(C.EXPERIMENT_ID):
            periods = []
            for timepoint in cur_exp_df[C.TIME].unique():
                condition_ids = [
                    cid
                    for cid in cur_exp_df.loc[
                        cur_exp_df[C.TIME] == timepoint, C.CONDITION_ID
                    ]
                    if not pd.isna(cid)
                ]
                periods.append(
                    ExperimentPeriod(
                        time=timepoint,
                        condition_ids=condition_ids,
                    )
                )
            experiments.append(Experiment(id=experiment_id, periods=periods))

        return cls(experiments, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the ExperimentTable to a DataFrame."""
        records = [
            {
                C.EXPERIMENT_ID: experiment.id,
                C.TIME: period.time,
                C.CONDITION_ID: condition_id,
            }
            for experiment in self.experiments
            for period in experiment.periods
            for condition_id in period.condition_ids or [""]
        ]
        return (
            pd.DataFrame(records)
            if records
            else pd.DataFrame(columns=C.EXPERIMENT_DF_REQUIRED_COLS)
        )


class Measurement(BaseModel):
    """A measurement.

    A measurement of an observable at a specific time point in a specific
    experiment.
    """

    #: The model ID.
    model_id: Annotated[
        str | None, BeforeValidator(_valid_petab_id_or_none)
    ] = Field(alias=C.MODEL_ID, default=None)
    #: The observable ID.
    observable_id: Annotated[str, BeforeValidator(_valid_petab_id)] = Field(
        alias=C.OBSERVABLE_ID
    )
    #: The experiment ID.
    experiment_id: Annotated[
        str | None, BeforeValidator(_valid_petab_id_or_none)
    ] = Field(alias=C.EXPERIMENT_ID, default=None)
    #: The time point of the measurement in time units as defined in the model.
    time: Annotated[float, AfterValidator(_is_finite_or_pos_inf)] = Field(
        alias=C.TIME
    )
    #: The measurement value.
    measurement: Annotated[float, AfterValidator(_not_nan)] = Field(
        alias=C.MEASUREMENT
    )
    #: Values for placeholder parameters in the observable formula.
    observable_parameters: list[sp.Basic] = Field(
        alias=C.OBSERVABLE_PARAMETERS, default_factory=list
    )
    #: Values for placeholder parameters in the noise formula.
    noise_parameters: list[sp.Basic] = Field(
        alias=C.NOISE_PARAMETERS, default_factory=list
    )

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
        validate_assignment=True,
    )

    @field_validator(
        "experiment_id",
        "observable_parameters",
        "noise_parameters",
        mode="before",
    )
    @classmethod
    def convert_nan_to_none(cls, v, info: ValidationInfo):
        if isinstance(v, float) and np.isnan(v):
            return cls.model_fields[info.field_name].default
        return v

    @field_validator(
        "observable_parameters", "noise_parameters", mode="before"
    )
    @classmethod
    def _sympify_list(cls, v):
        if v is None:
            return []

        if isinstance(v, float) and np.isnan(v):
            return []

        if isinstance(v, str):
            v = v.split(C.PARAMETER_SEPARATOR)
        elif not isinstance(v, Sequence):
            v = [v]

        return [sympify_petab(x) for x in v]


class MeasurementTable(BaseTable[Measurement]):
    """PEtab measurement table."""

    @property
    def measurements(self) -> list[Measurement]:
        """List of measurements."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> MeasurementTable:
        """Create a MeasurementTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        if C.MODEL_ID in df.columns:
            df[C.MODEL_ID] = df[C.MODEL_ID].apply(_convert_nan_to_none)

        measurements = [
            Measurement(
                **row.to_dict(),
            )
            for _, row in df.reset_index().iterrows()
        ]

        return cls(measurements, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the MeasurementTable to a DataFrame."""
        records = self.model_dump(by_alias=True)["elements"]
        for record in records:
            record[C.OBSERVABLE_PARAMETERS] = C.PARAMETER_SEPARATOR.join(
                map(str, record[C.OBSERVABLE_PARAMETERS])
            )
            record[C.NOISE_PARAMETERS] = C.PARAMETER_SEPARATOR.join(
                map(str, record[C.NOISE_PARAMETERS])
            )

        return pd.DataFrame(records)


class Mapping(BaseModel):
    """Mapping PEtab entities to model entities."""

    #: PEtab entity ID.
    petab_id: Annotated[str, AfterValidator(_valid_petab_id)] = Field(
        alias=C.PETAB_ENTITY_ID
    )
    #: Model entity ID.
    model_id: Annotated[str | None, BeforeValidator(_convert_nan_to_none)] = (
        Field(alias=C.MODEL_ENTITY_ID, default=None)
    )
    #: Arbitrary name
    name: Annotated[str | None, BeforeValidator(_convert_nan_to_none)] = Field(
        alias=C.NAME, default=None
    )

    #: :meta private:
    model_config = ConfigDict(
        populate_by_name=True, extra="allow", validate_assignment=True
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if (
            self.model_id
            and self.model_id != self.petab_id
            and is_valid_identifier(self.model_id)
        ):
            raise ValueError(
                "Aliasing of entities that already have a valid identifier "
                "is not allowed. Simplify your PEtab problem by removing the "
                f"mapping entry for `{self.petab_id} -> {self.model_id}`, "
                f"and replacing all occurrences of `{self.petab_id}` with "
                f"`{self.model_id}`."
            )
        return self


class MappingTable(BaseTable[Mapping]):
    """PEtab mapping table."""

    @property
    def mappings(self) -> list[Mapping]:
        """List of mappings."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> MappingTable:
        """Create a MappingTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        mappings = [
            Mapping(**row.to_dict()) for _, row in df.reset_index().iterrows()
        ]
        return cls(mappings, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the MappingTable to a DataFrame."""
        res = (
            pd.DataFrame(self.model_dump(by_alias=True)["elements"])
            if self.mappings
            else pd.DataFrame(columns=C.MAPPING_DF_REQUIRED_COLS)
        )
        return res.set_index([C.PETAB_ENTITY_ID])

    def __getitem__(self, petab_id: str) -> Mapping:
        """Get a mapping by PEtab ID."""
        for mapping in self.mappings:
            if mapping.petab_id == petab_id:
                return mapping
        raise KeyError(f"PEtab ID {petab_id} not found")

    def get(self, petab_id, default=None):
        """Get a mapping by PEtab ID or return a default value."""
        try:
            return self[petab_id]
        except KeyError:
            return default


class Parameter(BaseModel):
    """Parameter definition."""

    #: Parameter ID.
    id: Annotated[str, BeforeValidator(_valid_petab_id)] = Field(
        alias=C.PARAMETER_ID
    )
    #: Lower bound.
    lb: Annotated[float | None, BeforeValidator(_convert_nan_to_none)] = Field(
        alias=C.LOWER_BOUND, default=None
    )
    #: Upper bound.
    ub: Annotated[float | None, BeforeValidator(_convert_nan_to_none)] = Field(
        alias=C.UPPER_BOUND, default=None
    )
    #: Nominal value.
    nominal_value: Annotated[
        float | None, BeforeValidator(_convert_nan_to_none)
    ] = Field(alias=C.NOMINAL_VALUE, default=None)
    #: Is the parameter to be estimated?
    estimate: bool = Field(alias=C.ESTIMATE, default=True)
    #: Type of parameter prior distribution.
    prior_distribution: Annotated[
        PriorDistribution | None, BeforeValidator(_convert_nan_to_none)
    ] = Field(alias=C.PRIOR_DISTRIBUTION, default=None)
    #: Prior distribution parameters.
    prior_parameters: list[float] = Field(
        alias=C.PRIOR_PARAMETERS, default_factory=list
    )

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=True,
        extra="allow",
        validate_assignment=True,
    )

    @field_validator("prior_parameters", mode="before")
    @classmethod
    def _validate_prior_parameters(
        cls, v: str | list[str] | float | None | np.ndarray
    ):
        if v is None:
            return []

        if isinstance(v, float) and np.isnan(v):
            return []

        if isinstance(v, str):
            if v == "":
                return []
            v = v.split(C.PARAMETER_SEPARATOR)
        elif not isinstance(v, Sequence):
            v = [v]

        return [float(x) for x in v]

    @field_validator("estimate", mode="before")
    @classmethod
    def _validate_estimate_before(cls, v: bool | str):
        if isinstance(v, bool):
            return v

        if isinstance(v, str):
            v = v.strip().lower()
            if v == "true":
                return True
            if v == "false":
                return False

        raise ValueError(
            f"Invalid value for estimate: {v}. Must be `true` or `false`."
        )

    @field_serializer("estimate")
    def _serialize_estimate(self, estimate: bool, _info):
        return str(estimate).lower()

    @field_serializer("prior_distribution")
    def _serialize_prior_distribution(
        self, prior_distribution: PriorDistribution | None, _info
    ):
        if prior_distribution is None:
            return ""
        return str(prior_distribution)

    @field_serializer("prior_parameters")
    def _serialize_prior_parameters(
        self, prior_parameters: list[float], _info
    ) -> str:
        return C.PARAMETER_SEPARATOR.join(map(str, prior_parameters))

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.estimate and self.nominal_value is None:
            raise ValueError(
                "Non-estimated parameter must have a nominal value"
            )

        if self.estimate and (self.lb is None or self.ub is None):
            raise ValueError(
                "Estimated parameter must have lower and upper bounds set"
            )

        if self.lb is not None and self.ub is not None and self.lb > self.ub:
            raise ValueError(
                "Lower bound must be less than or equal to upper bound."
            )

        # NOTE: priorType and priorParameters are currently checked in
        #   `CheckPriorDistribution`

        return self

    @property
    def prior_dist(self) -> Distribution | None:
        """Get the prior distribution of the parameter.

        :return: The prior distribution of the parameter, or None if no prior
            distribution is set.
        """
        if not self.estimate:
            raise ValueError(f"Parameter `{self.id}' is not estimated.")

        if self.prior_distribution is None:
            return None

        if not (cls := _prior_to_cls.get(self.prior_distribution)):
            raise ValueError(
                f"Prior distribution `{self.prior_distribution}' not "
                "supported."
            )

        if str(self.prior_distribution).startswith("log-"):
            log = True
        elif str(self.prior_distribution).startswith("log10-"):
            log = 10
        else:
            log = False

        if cls == Exponential:
            # `Exponential.__init__` does not accept the `log` parameter
            if log is not False:
                raise ValueError(
                    "Exponential distribution does not support log "
                    "transformation."
                )
            return cls(*self.prior_parameters, trunc=[self.lb, self.ub])

        if cls == Uniform:
            # `Uniform.__init__` does not accept the `trunc` parameter
            low = max(self.prior_parameters[0], self.lb)
            high = min(self.prior_parameters[1], self.ub)
            return cls(low, high)

        if cls == LogUniform:
            # Mind the different interpretation of distribution parameters for
            #  Uniform(..., log=True) and LogUniform!!
            return cls(*self.prior_parameters, trunc=[self.lb, self.ub])

        return cls(*self.prior_parameters, log=log, trunc=[self.lb, self.ub])


class ParameterTable(BaseTable[Parameter]):
    """PEtab parameter table."""

    @property
    def parameters(self) -> list[Parameter]:
        """List of parameters."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> ParameterTable:
        """Create a ParameterTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        parameters = [
            Parameter(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(parameters, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the ParameterTable to a DataFrame."""
        return pd.DataFrame(
            self.model_dump(by_alias=True)["elements"]
        ).set_index([C.PARAMETER_ID])

    @property
    def n_estimated(self) -> int:
        """Number of estimated parameters."""
        return sum(p.estimate for p in self.parameters)


class Problem:
    """
    PEtab parameter estimation problem

    A PEtab parameter estimation problem as defined by

    - models
    - condition tables
    - experiment tables
    - measurement tables
    - parameter tables
    - observable tables
    - mapping tables

    See also :doc:`petab:v2/documentation_data_format`.
    """

    def __init__(
        self,
        models: list[Model] = None,
        condition_tables: list[ConditionTable] = None,
        experiment_tables: list[ExperimentTable] = None,
        observable_tables: list[ObservableTable] = None,
        measurement_tables: list[MeasurementTable] = None,
        parameter_tables: list[ParameterTable] = None,
        mapping_tables: list[MappingTable] = None,
        config: ProblemConfig = None,
    ):
        from ..v2.lint import default_validation_tasks

        self.config = config
        self.models: list[Model] = models or []
        self.validation_tasks: list[ValidationTask] = (
            default_validation_tasks.copy()
        )

        self.observable_tables = observable_tables or [ObservableTable()]
        self.condition_tables = condition_tables or [ConditionTable()]
        self.experiment_tables = experiment_tables or [ExperimentTable()]
        self.measurement_tables = measurement_tables or [MeasurementTable()]
        self.mapping_tables = mapping_tables or [MappingTable()]
        self.parameter_tables = parameter_tables or [ParameterTable()]

    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id!r}>"

    def __str__(self):
        pid = repr(self.id) if self.id else "without ID"

        model = f"with models {self.models}" if self.model else "without model"

        ne = len(self.experiments)
        experiments = f"{ne} experiments"

        nc = len(self.conditions)
        conditions = f"{nc} conditions"

        no = len(self.observables)
        observables = f"{no} observables"

        nm = len(self.measurements)
        measurements = f"{nm} measurements"

        nest = sum(pt.n_estimated for pt in self.parameter_tables)
        parameters = f"{nest} estimated parameters"

        return (
            f"PEtab Problem {pid} {model}, {conditions}, {experiments}, "
            f"{observables}, {measurements}, {parameters}"
        )

    def __getitem__(
        self, key
    ) -> (
        Condition | Experiment | Observable | Measurement | Parameter | Mapping
    ):
        """Get PEtab entity by ID.

        This allows accessing PEtab entities such as conditions, experiments,
        observables, and parameters by their ID.

        Accessing model entities is not currently not supported.
        """
        for table_list in (
            self.condition_tables,
            self.experiment_tables,
            self.observable_tables,
            self.measurement_tables,
            self.parameter_tables,
            self.mapping_tables,
        ):
            for table in table_list:
                try:
                    return table[key]
                except (KeyError, NotImplementedError):
                    pass

        raise KeyError(
            f"Entity with ID '{key}' not found in the PEtab problem"
        )

    @staticmethod
    def from_yaml(
        yaml_config: dict | Path | str, base_path: str | Path = None
    ) -> Problem:
        """
        Factory method to load model and tables as specified by YAML file.

        Arguments:
            yaml_config: PEtab configuration as dictionary or YAML file name
            base_path: Base directory or URL to resolve relative paths
        """
        if isinstance(yaml_config, Path):
            yaml_config = str(yaml_config)

        if isinstance(yaml_config, str):
            yaml_file = yaml_config
            if base_path is None:
                base_path = get_path_prefix(yaml_file)
            yaml_config = yaml.load_yaml(yaml_file)
        else:
            yaml_file = None

        validate_yaml_syntax(yaml_config)

        if (format_version := parse_version(yaml_config[C.FORMAT_VERSION]))[
            0
        ] != 2:
            # If we got a path to a v1 yaml file, try to auto-upgrade
            from tempfile import TemporaryDirectory

            from .petab1to2 import petab1to2

            if format_version[0] == 1 and yaml_file:
                logger.debug(
                    "Auto-upgrading problem from PEtab 1.0 to PEtab 2.0"
                )
                with TemporaryDirectory() as tmpdirname:
                    try:
                        petab1to2(yaml_file, output_dir=tmpdirname)
                    except Exception as e:
                        raise ValueError(
                            "Failed to auto-upgrade PEtab 1.0 problem to "
                            "PEtab 2.0"
                        ) from e
                    return Problem.from_yaml(
                        Path(tmpdirname) / Path(yaml_file).name
                    )
            raise ValueError(
                "Provided PEtab files are of unsupported version "
                f"{yaml_config[C.FORMAT_VERSION]}."
            )

        config = ProblemConfig(
            **yaml_config, base_path=base_path, filepath=yaml_file
        )

        parameter_tables = [
            ParameterTable.from_tsv(f, base_path=base_path)
            for f in config.parameter_files
        ]

        models = [
            model_factory(
                model_info.location,
                base_path=base_path,
                model_language=model_info.language,
                model_id=model_id,
            )
            for model_id, model_info in (config.model_files or {}).items()
        ]

        measurement_tables = (
            [
                MeasurementTable.from_tsv(f, base_path)
                for f in config.measurement_files
            ]
            if config.measurement_files
            else None
        )

        condition_tables = (
            [
                ConditionTable.from_tsv(f, base_path)
                for f in config.condition_files
            ]
            if config.condition_files
            else None
        )

        experiment_tables = (
            [
                ExperimentTable.from_tsv(f, base_path)
                for f in config.experiment_files
            ]
            if config.experiment_files
            else None
        )

        observable_tables = (
            [
                ObservableTable.from_tsv(f, base_path)
                for f in config.observable_files
            ]
            if config.observable_files
            else None
        )

        mapping_tables = (
            [MappingTable.from_tsv(f, base_path) for f in config.mapping_files]
            if config.mapping_files
            else None
        )

        return Problem(
            config=config,
            models=models,
            condition_tables=condition_tables,
            experiment_tables=experiment_tables,
            observable_tables=observable_tables,
            measurement_tables=measurement_tables,
            parameter_tables=parameter_tables,
            mapping_tables=mapping_tables,
        )

    @staticmethod
    def from_combine(filename: Path | str) -> Problem:
        """Read PEtab COMBINE archive (http://co.mbine.org/documents/archive).

        See also :py:func:`petab.v2.create_combine_archive`.

        Arguments:
            filename: Path to the PEtab-COMBINE archive

        Returns:
            A :py:class:`petab.v2.Problem` instance.
        """
        # function-level import, because module-level import interfered with
        # other SWIG interfaces
        try:
            import libcombine
        except ImportError as e:
            raise ImportError(
                "To use PEtab's COMBINE functionality, libcombine "
                "(python-libcombine) must be installed."
            ) from e

        archive = libcombine.CombineArchive()
        if archive.initializeFromArchive(str(filename)) is None:
            raise ValueError(f"Invalid Combine Archive: {filename}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            archive.extractTo(tmpdirname)
            problem = Problem.from_yaml(
                os.path.join(tmpdirname, archive.getMasterFile().getLocation())
            )
        archive.cleanUp()

        return problem

    @staticmethod
    def get_problem(problem: str | Path | Problem) -> Problem:
        """Get a PEtab problem from a file or a problem object.

        Arguments:
            problem: Path to a PEtab problem file or a PEtab problem object.

        Returns:
            A PEtab problem object.
        """
        if isinstance(problem, Problem):
            return problem

        if isinstance(problem, str | Path):
            return Problem.from_yaml(problem)

        raise TypeError(
            "The argument `problem` must be a path to a PEtab problem file "
            "or a PEtab problem object."
        )

    def to_files(self, base_path: str | Path | None) -> None:
        """Write the PEtab problem to files.

        Writes the model, condition, experiment, measurement, parameter,
        observable, and mapping tables to their respective files as specified
        by the `rel_path` and `base_path` of their respective objects.

        This expects that all objects have their `rel_path` and `base_path`
        set correctly, which is usually done by :meth:`Problem.from_yaml`.

        :param base_path:
            The base path the yaml file and tables will be written to.
            If ``None``, the `base_path` of the individual tables and
            :obj:`Problem.config.base_path` will be used.
        """
        config = copy.deepcopy(self.config) or ProblemConfig(
            format_version="2.0.0"
        )

        for model in self.models:
            model.to_file(
                _generate_path(model.rel_path, base_path or model.base_path)
            )

        config.model_files = {
            model.model_id: ModelFile(
                location=model.rel_path, language=model.type_id
            )
            for model in self.models
        }

        config.condition_files = [
            table.rel_path for table in self.condition_tables if table.rel_path
        ]
        config.experiment_files = [
            table.rel_path
            for table in self.experiment_tables
            if table.rel_path
        ]
        config.observable_files = [
            table.rel_path
            for table in self.observable_tables
            if table.rel_path
        ]
        config.measurement_files = [
            table.rel_path
            for table in self.measurement_tables
            if table.rel_path
        ]
        config.parameter_files = [
            table.rel_path for table in self.parameter_tables if table.rel_path
        ]
        config.mapping_files = [
            table.rel_path for table in self.mapping_tables if table.rel_path
        ]

        for table in chain(
            self.condition_tables,
            self.experiment_tables,
            self.observable_tables,
            self.measurement_tables,
            self.parameter_tables,
            self.mapping_tables,
        ):
            if table.rel_path:
                table.to_tsv(
                    _generate_path(
                        table.rel_path, base_path or table.base_path
                    )
                )

        config.to_yaml(
            _generate_path(
                Path(str(config.filepath)).name, base_path or config.base_path
            )
        )

    @property
    def model(self) -> Model | None:
        """The model of the problem.

        This is a convenience property for `Problem`s with only one single
        model.

        :return:
            The model of the problem, or None if no model is defined.
        :raises:
            ValueError: If the problem has more than one model defined.
        """
        if len(self.models) == 1:
            return self.models[0]

        if len(self.models) == 0:
            return None

        raise ValueError(
            "Problem contains more than one model. "
            "Use `Problem.models` to access all models."
        )

    @model.setter
    def model(self, value: Model):
        """Set the model of the problem.

        This is a convenience setter for `Problem`s with only one single
        model. This will replace any existing models in the problem with the
        provided model.
        """
        self.models = [value]

    @property
    def condition_df(self) -> pd.DataFrame | None:
        """Combined condition tables as DataFrame."""
        return (
            ConditionTable(conditions).to_df()
            if (conditions := self.conditions)
            else None
        )

    @condition_df.setter
    def condition_df(self, value: pd.DataFrame):
        self.condition_tables = [ConditionTable.from_df(value)]

    @property
    def experiment_df(self) -> pd.DataFrame | None:
        """Experiment table as DataFrame."""
        return (
            ExperimentTable(experiments).to_df()
            if (experiments := self.experiments)
            else None
        )

    @experiment_df.setter
    def experiment_df(self, value: pd.DataFrame):
        self.experiment_tables = [ExperimentTable.from_df(value)]

    @property
    def measurement_df(self) -> pd.DataFrame | None:
        """Combined measurement tables as DataFrame."""
        return (
            MeasurementTable(measurements).to_df()
            if (measurements := self.measurements)
            else None
        )

    @measurement_df.setter
    def measurement_df(self, value: pd.DataFrame):
        self.measurement_tables = [MeasurementTable.from_df(value)]

    @property
    def parameter_df(self) -> pd.DataFrame | None:
        """Combined parameter tables as DataFrame."""
        return (
            ParameterTable(parameters).to_df()
            if (parameters := self.parameters)
            else None
        )

    @parameter_df.setter
    def parameter_df(self, value: pd.DataFrame):
        self.parameter_tables = [ParameterTable.from_df(value)]

    @property
    def observable_df(self) -> pd.DataFrame | None:
        """Combined observable tables as DataFrame."""
        return (
            ObservableTable(observables).to_df()
            if (observables := self.observables)
            else None
        )

    @observable_df.setter
    def observable_df(self, value: pd.DataFrame):
        self.observable_tables = [ObservableTable.from_df(value)]

    @property
    def mapping_df(self) -> pd.DataFrame | None:
        """Combined mapping tables as DataFrame."""
        return (
            MappingTable(mappings).to_df()
            if (mappings := self.mappings)
            else None
        )

    @mapping_df.setter
    def mapping_df(self, value: pd.DataFrame):
        self.mapping_tables = [MappingTable.from_df(value)]

    @property
    def conditions(self) -> list[Condition]:
        """List of conditions in the condition table(s)."""
        return list(
            chain.from_iterable(ct.conditions for ct in self.condition_tables)
        )

    @property
    def experiments(self) -> list[Experiment]:
        """List of experiments in the experiment table(s)."""
        return list(
            chain.from_iterable(
                et.experiments for et in self.experiment_tables
            )
        )

    @property
    def observables(self) -> list[Observable]:
        """List of observables in the observable table(s)."""
        return list(
            chain.from_iterable(
                ot.observables for ot in self.observable_tables
            )
        )

    @property
    def measurements(self) -> list[Measurement]:
        """List of measurements in the measurement table(s)."""
        return list(
            chain.from_iterable(
                mt.measurements for mt in self.measurement_tables
            )
        )

    @property
    def parameters(self) -> list[Parameter]:
        """List of parameters in the parameter table(s)."""
        return list(
            chain.from_iterable(pt.parameters for pt in self.parameter_tables)
        )

    @property
    def mappings(self) -> list[Mapping]:
        """List of mappings in the mapping table(s)."""
        return list(
            chain.from_iterable(mt.mappings for mt in self.mapping_tables)
        )

    @property
    def id(self) -> str | None:
        """The ID of the PEtab problem if set, ``None`` otherwise."""
        return self.config.id if self.config else None

    @id.setter
    def id(self, value: str):
        """Set the ID of the PEtab problem."""
        if self.config is None:
            self.config = ProblemConfig(format_version="2.0.0")
        self.config.id = value

    def get_optimization_parameters(self) -> list[str]:
        """
        Get the list of optimization parameter IDs from parameter table.

        Returns:
            A list of IDs of parameters selected for optimization
            (i.e., those with estimate = True).
        """
        return [p.id for p in self.parameters if p.estimate]

    def get_observable_ids(self) -> list[str]:
        """
        Returns dictionary of observable ids.
        """
        return [o.id for o in self.observables]

    def _apply_mask(self, v: list, free: bool = True, fixed: bool = True):
        """Apply mask of only free or only fixed values.

        Parameters
        ----------
        v:
            The full vector the mask is to be applied to.
        free:
            Whether to return free parameters, i.e., parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e., parameters not to
            estimate.

        Returns
        -------
        The reduced vector with applied mask.
        """
        if not free and not fixed:
            return []
        if not free:
            return [v[ix] for ix in self.x_fixed_indices]
        if not fixed:
            return [v[ix] for ix in self.x_free_indices]
        return v

    def get_x_ids(self, free: bool = True, fixed: bool = True):
        """Generic function to get parameter ids.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.

        Returns
        -------
        The parameter IDs.
        """
        v = [p.id for p in self.parameters]
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def x_ids(self) -> list[str]:
        """Parameter table parameter IDs"""
        return self.get_x_ids()

    @property
    def x_free_ids(self) -> list[str]:
        """Parameter table parameter IDs, for free parameters."""
        return self.get_x_ids(fixed=False)

    @property
    def x_fixed_ids(self) -> list[str]:
        """Parameter table parameter IDs, for fixed parameters."""
        return self.get_x_ids(free=False)

    def get_x_nominal(self, free: bool = True, fixed: bool = True) -> list:
        """Generic function to get parameter nominal values.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.

        Returns
        -------
        The parameter nominal values.
        """
        v = [
            p.nominal_value if p.nominal_value is not None else nan
            for p in self.parameters
        ]

        return self._apply_mask(v, free=free, fixed=fixed)

    def get_x_nominal_dict(
        self, free: bool = True, fixed: bool = True
    ) -> dict[str, float]:
        """Get parameter nominal values as dict.

        :param free:
            Whether to return free parameters, i.e. parameters to estimate.
        :param fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.
        :returns:
            A dictionary mapping parameter IDs to their nominal values.
        """
        return dict(
            zip(
                self.get_x_ids(free=free, fixed=fixed),
                self.get_x_nominal(free=free, fixed=fixed),
                strict=True,
            )
        )

    @property
    def x_nominal(self) -> list:
        """Parameter table nominal values"""
        return self.get_x_nominal()

    @property
    def x_nominal_free(self) -> list:
        """Parameter table nominal values, for free parameters."""
        return self.get_x_nominal(fixed=False)

    @property
    def x_nominal_fixed(self) -> list:
        """Parameter table nominal values, for fixed parameters."""
        return self.get_x_nominal(free=False)

    def get_lb(self, free: bool = True, fixed: bool = True):
        """Generic function to get lower parameter bounds.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.

        Returns
        -------
        The lower parameter bounds.
        """
        v = [p.lb if p.lb is not None else nan for p in self.parameters]
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def lb(self) -> list:
        """Parameter table lower bounds."""
        return self.get_lb()

    def get_ub(self, free: bool = True, fixed: bool = True):
        """Generic function to get upper parameter bounds.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.

        Returns
        -------
        The upper parameter bounds.
        """
        v = [p.ub if p.ub is not None else nan for p in self.parameters]
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def ub(self) -> list:
        """Parameter table upper bounds"""
        return self.get_ub()

    @property
    def x_free_indices(self) -> list[int]:
        """Parameter table estimated parameter indices."""
        return [i for i, p in enumerate(self.parameters) if p.estimate]

    @property
    def x_fixed_indices(self) -> list[int]:
        """Parameter table non-estimated parameter indices."""
        return [i for i, p in enumerate(self.parameters) if not p.estimate]

    @property
    def has_map_objective(self) -> bool:
        """Whether this problem encodes a maximum a posteriori (MAP) objective.

        A PEtab problem is considered to have a MAP objective if there is a
        prior distribution specified for at least one estimated parameter.

        :returns: ``True`` if MAP objective, ``False`` otherwise.
        """
        return any(
            p.prior_distribution is not None
            for p in self.parameters
            if p.estimate
        )

    @property
    def has_ml_objective(self) -> bool:
        """Whether this problem encodes a maximum likelihood (ML) objective.

        A PEtab problem is considered to have an ML objective if there are no
        prior distributions specified for any estimated parameters.

        :returns: ``True`` if ML objective, ``False`` otherwise.
        """
        return not self.has_map_objective

    def get_priors(self) -> dict[str, Distribution]:
        """Get prior distributions.

        Note that this will default to uniform distributions over the
        parameter bounds for parameters without an explicit prior.

        :returns: The prior distributions for the estimated parameters in case
            the problem has a MAP objective, an empty dictionary otherwise.
        """
        if not self.has_map_objective:
            return {}

        return {
            p.id: p.prior_dist if p.prior_distribution else Uniform(p.lb, p.ub)
            for p in self.parameters
            if p.estimate
        }

    def get_startpoint_distributions(self) -> dict[str, Distribution]:
        """Get distributions for sampling startpoints.

        The distributions are the prior distributions for estimated parameters
        that have a prior distribution defined, and uniform distributions
        over the parameter bounds for estimated parameters without an explicit
        prior.

        :returns: Mapping of parameter IDs to distributions for sampling
            startpoints.
        """
        return {
            p.id: p.prior_dist if p.prior_distribution else Uniform(p.lb, p.ub)
            for p in self.parameters
            if p.estimate
        }

    def sample_parameter_startpoints(self, n_starts: int = 100, **kwargs):
        """Create 2D array with starting points for optimization"""
        priors = self.get_priors()
        return np.vstack([p.sample(n_starts) for p in priors.values()]).T

    def sample_parameter_startpoints_dict(
        self, n_starts: int = 100
    ) -> list[dict[str, float]]:
        """Create dictionaries with starting points for optimization

        :returns:
            A list of dictionaries with parameter IDs mapping to sampled
            parameter values.
        """
        return [
            dict(zip(self.x_free_ids, parameter_values, strict=True))
            for parameter_values in self.sample_parameter_startpoints(
                n_starts=n_starts
            )
        ]

    @property
    def n_estimated(self) -> int:
        """The number of estimated parameters."""
        return len(self.x_free_indices)

    @property
    def n_measurements(self) -> int:
        """Number of measurements."""
        return sum(len(mt.measurements) for mt in self.measurement_tables)

    @property
    def n_priors(self) -> int:
        """Number of priors."""
        return sum(p.prior_distribution is not None for p in self.parameters)

    def validate(
        self, validation_tasks: list[ValidationTask] = None
    ) -> ValidationResultList:
        """Validate the PEtab problem.

        Arguments:
            validation_tasks: List of validation tasks to run. If ``None``
             or empty, :attr:`Problem.validation_tasks` are used.
        Returns:
            A list of validation results.
        """
        from ..v2.lint import (
            ValidationIssue,
            ValidationIssueSeverity,
            ValidationResultList,
        )

        validation_results = ValidationResultList()

        if self.config and self.config.extensions:
            extensions = ",".join(self.config.extensions.keys())
            validation_results.append(
                ValidationIssue(
                    ValidationIssueSeverity.WARNING,
                    "Validation of PEtab extensions is not yet implemented, "
                    "but the given problem uses the following extensions: "
                    f"{extensions}",
                )
            )

        if len(self.models) > 1:
            # TODO https://github.com/PEtab-dev/libpetab-python/issues/392
            #  We might just want to split the problem into multiple
            #  problems, one for each model, and then validate each
            #  problem separately.
            validation_results.append(
                ValidationIssue(
                    ValidationIssueSeverity.WARNING,
                    "Problem contains multiple models. "
                    "Validation is not yet fully supported.",
                )
            )

        for task in validation_tasks or self.validation_tasks:
            try:
                cur_result = task.run(self)
            except Exception as e:
                cur_result = ValidationIssue(
                    ValidationIssueSeverity.CRITICAL,
                    f"Validation task {task} failed with exception: {e}\n"
                    f"{traceback.format_exc()}",
                )

            if cur_result:
                validation_results.append(cur_result)

                if cur_result.level == ValidationIssueSeverity.CRITICAL:
                    break

        return validation_results

    def assert_valid(self, **kwargs) -> None:
        """Assert that the PEtab problem is valid.

        :param kwargs: Additional arguments passed to :meth:`Problem.validate`.

        :raises AssertionError: If the PEtab problem is not valid.
        """
        from ..v2.lint import ValidationIssueSeverity

        validation_results = self.validate(**kwargs)
        errors = [
            r
            for r in validation_results
            if r.level >= ValidationIssueSeverity.ERROR
        ]
        if errors:
            raise AssertionError(
                "PEtab problem is not valid:\n"
                + "\n".join(e.message for e in errors)
            )

    def add_condition(
        self, id_: str, name: str = None, **kwargs: Number | str | sp.Expr
    ):
        """Add a simulation condition to the problem.

        If there are more than one condition tables, the condition
        is added to the last one.

        Arguments:
            id_: The condition id
            name: The condition name. If given, this will be added to the
                last mapping table. If no mapping table exists,
                a new mapping table will be created.
            kwargs: Entities to be added to the condition table in the form
                `target_id=target_value`.
        """
        if not kwargs:
            raise ValueError("Cannot add condition without any changes")

        changes = [
            Change(target_id=target_id, target_value=target_value)
            for target_id, target_value in kwargs.items()
        ]
        if not self.condition_tables:
            self.condition_tables.append(ConditionTable())
        self.condition_tables[-1].conditions.append(
            Condition(id=id_, changes=changes)
        )
        if name is not None:
            self.add_mapping(petab_id=id_, name=name)

    def add_observable(
        self,
        id_: str,
        formula: str,
        noise_formula: str | float | int = None,
        noise_distribution: str = None,
        observable_placeholders: list[str] = None,
        noise_placeholders: list[str] = None,
        name: str = None,
        **kwargs,
    ):
        """Add an observable to the problem.

        If there are more than one observable tables, the observable
        is added to the last one.

        Arguments:
            id_: The observable id
            formula: The observable formula
            noise_formula: The noise formula
            noise_distribution: The noise distribution
            observable_placeholders: Placeholders for the observable formula
            noise_placeholders: Placeholders for the noise formula
            name: The observable name
            kwargs: additional columns/values to add to the observable table

        """
        record = {
            C.OBSERVABLE_ID: id_,
            C.OBSERVABLE_FORMULA: formula,
        }
        if name is not None:
            record[C.OBSERVABLE_NAME] = name
        if noise_formula is not None:
            record[C.NOISE_FORMULA] = noise_formula
        if noise_distribution is not None:
            record[C.NOISE_DISTRIBUTION] = noise_distribution
        if observable_placeholders is not None:
            record[C.OBSERVABLE_PLACEHOLDERS] = observable_placeholders
        if noise_placeholders is not None:
            record[C.NOISE_PLACEHOLDERS] = noise_placeholders
        record.update(kwargs)

        if not self.observable_tables:
            self.observable_tables.append(ObservableTable())

        self.observable_tables[-1] += Observable(**record)

    def add_parameter(
        self,
        id_: str,
        estimate: bool | str = True,
        nominal_value: Number | None = None,
        lb: Number = None,
        ub: Number = None,
        prior_dist: str = None,
        prior_pars: str | Sequence = None,
        **kwargs,
    ):
        """Add a parameter to the problem.

        If there are more than one parameter tables, the parameter
        is added to the last one.

        Arguments:
            id_: The parameter id
            estimate: Whether the parameter is estimated
            nominal_value: The nominal value of the parameter
            lb: The lower bound of the parameter
            ub: The upper bound of the parameter
            prior_dist: The type of the prior distribution
            prior_pars: The parameters of the prior distribution
            kwargs: additional columns/values to add to the parameter table
        """
        record = {
            C.PARAMETER_ID: id_,
        }
        if estimate is not None:
            record[C.ESTIMATE] = estimate
        if nominal_value is not None:
            record[C.NOMINAL_VALUE] = nominal_value
        if lb is not None:
            record[C.LOWER_BOUND] = lb
        if ub is not None:
            record[C.UPPER_BOUND] = ub
        if prior_dist is not None:
            record[C.PRIOR_DISTRIBUTION] = prior_dist
        if prior_pars is not None:
            if isinstance(prior_pars, Sequence) and not isinstance(
                prior_pars, str
            ):
                prior_pars = C.PARAMETER_SEPARATOR.join(map(str, prior_pars))
            record[C.PRIOR_PARAMETERS] = prior_pars
        record.update(kwargs)

        if not self.parameter_tables:
            self.parameter_tables.append(ParameterTable())

        self.parameter_tables[-1] += Parameter(**record)

    def add_measurement(
        self,
        obs_id: str,
        *,
        time: float,
        measurement: float,
        experiment_id: str | None = None,
        observable_parameters: Sequence[str | float] | str | float = None,
        noise_parameters: Sequence[str | float] | str | float = None,
    ):
        """Add a measurement to the problem.

        If there are more than one measurement tables, the measurement
        is added to the last one.

        Arguments:
            obs_id: The observable ID
            experiment_id: The experiment ID
            time: The measurement time
            measurement: The measurement value
            observable_parameters: The observable parameters
            noise_parameters: The noise parameters
        """
        if observable_parameters is not None and not isinstance(
            observable_parameters, Sequence
        ):
            observable_parameters = [observable_parameters]
        if noise_parameters is not None and not isinstance(
            noise_parameters, Sequence
        ):
            noise_parameters = [noise_parameters]

        if not self.measurement_tables:
            self.measurement_tables.append(MeasurementTable())

        self.measurement_tables[-1].measurements.append(
            Measurement(
                observable_id=obs_id,
                experiment_id=experiment_id,
                time=time,
                measurement=measurement,
                observable_parameters=observable_parameters,
                noise_parameters=noise_parameters,
            )
        )

    def add_mapping(
        self, petab_id: str, model_id: str = None, name: str = None
    ):
        """Add a mapping table entry to the problem.

        If there are more than one mapping tables, the mapping
        is added to the last one.

        Arguments:
            petab_id: The new PEtab-compatible ID mapping to `model_id`
            model_id: The ID of some entity in the model
            name: A name (any string) for the entity referenced by `petab_id`.
        """
        if not self.mapping_tables:
            self.mapping_tables.append(MappingTable())
        self.mapping_tables[-1].mappings.append(
            Mapping(petab_id=petab_id, model_id=model_id, name=name)
        )

    def add_experiment(self, id_: str, *args):
        """Add an experiment to the problem.

        If there are more than one experiment tables, the experiment
        is added to the last one.

        :param id_: The experiment ID.
        :param args: Timepoints and associated conditions
            (single condition ID as string or multiple condition IDs as lists
            of strings).

        :example:
        >>> p = Problem()
        >>> p.add_experiment(
        ...     "experiment1",
        ...     1,
        ...     "condition1",
        ...     2,
        ...     ["condition2a", "condition2b"],
        ... )
        >>> p.experiments[0]  # doctest: +NORMALIZE_WHITESPACE
        Experiment(id='experiment1', periods=[\
ExperimentPeriod(time=1.0, condition_ids=['condition1']), \
ExperimentPeriod(time=2.0, condition_ids=['condition2a', 'condition2b'])])
        """
        if len(args) % 2 != 0:
            raise ValueError(
                "Arguments must be pairs of timepoints and condition IDs."
            )

        periods = [
            ExperimentPeriod(
                time=args[i],
                condition_ids=[cond]
                if isinstance((cond := args[i + 1]), str)
                else cond,
            )
            for i in range(0, len(args), 2)
        ]

        if not self.experiment_tables:
            self.experiment_tables.append(ExperimentTable())
        self.experiment_tables[-1].experiments.append(
            Experiment(id=id_, periods=periods)
        )

    def __iadd__(self, other):
        """Add Observable, Parameter, Measurement, Condition, or Experiment"""
        from .core import (
            Condition,
            Experiment,
            Measurement,
            Observable,
            Parameter,
        )

        if isinstance(other, Observable):
            if not self.observable_tables:
                self.observable_tables.append(ObservableTable())
            self.observable_tables[-1] += other
        elif isinstance(other, Parameter):
            if not self.parameter_tables:
                self.parameter_tables.append(ParameterTable())
            self.parameter_tables[-1] += other
        elif isinstance(other, Measurement):
            if not self.measurement_tables:
                self.measurement_tables.append(MeasurementTable())
            self.measurement_tables[-1] += other
        elif isinstance(other, Condition):
            if not self.condition_tables:
                self.condition_tables.append(ConditionTable())
            self.condition_tables[-1] += other
        elif isinstance(other, Experiment):
            if not self.experiment_tables:
                self.experiment_tables.append(ExperimentTable())
            self.experiment_tables[-1] += other
        else:
            raise ValueError(
                f"Cannot add object of type {type(other)} to Problem."
            )
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Convert this Problem to a dictionary.

        This function is intended for debugging purposes and should not be
        used for serialization. The output of this function may change
        without notice.

        The output includes all PEtab tables, but not the models.

        See `pydantic.BaseModel.model_dump <https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump>`__
        for details.

        :example:

        >>> from pprint import pprint
        >>> p = Problem()
        >>> p += Parameter(id="par", lb=0, ub=1)
        >>> pprint(p.model_dump())
        {'conditions': [],
         'config': {'condition_files': [],
                    'experiment_files': [],
                    'extensions': {},
                    'format_version': '2.0.0',
                    'id': None,
                    'mapping_files': [],
                    'measurement_files': [],
                    'model_files': {},
                    'observable_files': [],
                    'parameter_files': []},
         'experiments': [],
         'mappings': [],
         'measurements': [],
         'observables': [],
         'parameters': [{'estimate': 'true',
                         'id': 'par',
                         'lb': 0.0,
                         'nominal_value': None,
                         'prior_distribution': '',
                         'prior_parameters': '',
                         'ub': 1.0}]}
        """
        res = {
            "config": (self.config or ProblemConfig()).model_dump(
                **kwargs, by_alias=True
            ),
        }
        for field, table_list in (
            ("conditions", self.condition_tables),
            ("experiments", self.experiment_tables),
            ("observables", self.observable_tables),
            ("measurements", self.measurement_tables),
            ("parameters", self.parameter_tables),
            ("mappings", self.mapping_tables),
        ):
            res[field] = (
                list(
                    chain.from_iterable(
                        table.model_dump(**kwargs)["elements"]
                        for table in table_list
                    )
                )
                if table_list
                else []
            )
        return res

    def get_changes_for_period(self, period: ExperimentPeriod) -> list[Change]:
        """Get the changes for a given experiment period.

        :param period: The experiment period to get the changes for.
        :return: A list of changes for the given period.
        """
        return list(
            chain.from_iterable(
                self[condition].changes for condition in period.condition_ids
            )
        )

    def get_measurements_for_experiment(
        self, experiment: Experiment
    ) -> list[Measurement]:
        """Get the measurements for a given experiment.

        :param experiment: The experiment to get the measurements for.
        :return: A list of measurements for the given experiment.
        """
        return [
            measurement
            for measurement in self.measurements
            if measurement.experiment_id == experiment.id
        ]

    def get_output_parameters(
        self, observable: bool = True, noise: bool = True
    ) -> list[str]:
        """Get output parameters.

        Returns IDs of symbols used in observable and noise formulas that are
        not observables and that are not defined in the model.

        :param observable:
            Include parameters from observableFormulas
        :param noise:
            Include parameters from noiseFormulas
        :returns:
            List of output parameter IDs, including any placeholder parameters.
        """
        # collect free symbols from observable and noise formulas,
        # skipping observable IDs
        candidates = set()
        if observable:
            candidates |= {
                str_sym
                for o in self.observables
                if o.formula is not None
                for sym in o.formula.free_symbols
                if (str_sym := str(sym)) != o.id
            }
        if noise:
            candidates |= {
                str_sym
                for o in self.observables
                if o.noise_formula is not None
                for sym in o.noise_formula.free_symbols
                if (str_sym := str(sym)) != o.id
            }

        output_parameters = []

        # filter out symbols that are defined in the model or mapped to
        #  such symbols
        for candidate in sorted(candidates):
            if self.model and self.model.symbol_allowed_in_observable_formula(
                candidate
            ):
                continue

            # does it map to a model entity?
            for mapping in self.mappings:
                if (
                    mapping.petab_id == candidate
                    and mapping.model_id is not None
                ):
                    if (
                        self.model
                        and self.model.symbol_allowed_in_observable_formula(
                            mapping.model_id
                        )
                    ):
                        break
            else:
                # no mapping to a model entity, so it is an output parameter
                output_parameters.append(candidate)

        return output_parameters


class ModelFile(BaseModel):
    """A file in the PEtab problem configuration."""

    location: AnyUrl | Path
    language: str

    model_config = ConfigDict(
        validate_assignment=True,
    )


class ExtensionConfig(BaseModel):
    """The configuration of a PEtab extension."""

    version: str
    config: dict


class ProblemConfig(BaseModel):
    """The PEtab problem configuration."""

    #: The path to the PEtab problem configuration.
    filepath: AnyUrl | Path | None = Field(
        None,
        description="The path to the PEtab problem configuration.",
        exclude=True,
    )
    #: The base path to resolve relative paths.
    base_path: AnyUrl | Path | None = Field(
        None,
        description="The base path to resolve relative paths.",
        exclude=True,
    )
    #: The PEtab format version.
    format_version: str = "2.0.0"

    #: The problem ID.
    id: str | None = None

    #: The paths to the parameter tables.
    # Absolute or relative to `base_path`.
    parameter_files: list[AnyUrl | Path] = []
    #: The model IDs and files used by the problem (`id->ModelFile`).
    model_files: dict[str, ModelFile] | None = {}
    #: The paths to the measurement tables.
    # Absolute or relative to `base_path`.
    measurement_files: list[AnyUrl | Path] = []
    #: The paths to the condition tables.
    # Absolute or relative to `base_path`.
    condition_files: list[AnyUrl | Path] = []
    #: The paths to the experiment tables.
    # Absolute or relative to `base_path`.
    experiment_files: list[AnyUrl | Path] = []
    #: The paths to the observable tables.
    # Absolute or relative to `base_path`.
    observable_files: list[AnyUrl | Path] = []
    #: The paths to the mapping tables.
    # Absolute or relative to `base_path`.
    mapping_files: list[AnyUrl | Path] = []

    #: Extensions used by the problem.
    extensions: list[ExtensionConfig] | dict = {}

    model_config = ConfigDict(
        validate_assignment=True,
    )

    # convert parameter_file to list
    @field_validator(
        "parameter_files",
        mode="before",
    )
    def _convert_parameter_file(cls, v):
        """Convert parameter_file to a list."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(
            "parameter_files must be a string or a list of strings."
        )

    def to_yaml(self, filename: str | Path):
        """Write the configuration to a YAML file.

        :param filename: Destination file name. The parent directory will be
            created if necessary.
        """
        from ..v1.yaml import write_yaml

        data = self.model_dump(by_alias=True)
        # convert Paths to strings for YAML serialization
        for key in (
            "measurement_files",
            "condition_files",
            "experiment_files",
            "observable_files",
            "mapping_files",
            "parameter_files",
        ):
            data[key] = list(map(str, data[key]))

        for model_id in data.get("model_files", {}):
            data["model_files"][model_id][C.MODEL_LOCATION] = str(
                data["model_files"][model_id]["location"]
            )
        if data["id"] is None:
            # The schema requires a valid id or no id field at all.
            del data["id"]

        write_yaml(data, filename)

    @property
    def format_version_tuple(self) -> tuple[int, int, int, str]:
        """The format version as a tuple of major/minor/patch `int`s and a
        suffix."""
        return parse_version(self.format_version)

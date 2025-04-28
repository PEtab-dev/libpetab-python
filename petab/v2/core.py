"""Types around the PEtab object model."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import sympy as sp
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ..v1.distributions import *
from ..v1.lint import is_valid_identifier
from ..v1.math import petab_math_str, sympify_petab
from . import C, get_observable_df

__all__ = [
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
]


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
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _valid_petab_id(v: str) -> str:
    """Field validator for PEtab IDs."""
    if not v:
        raise ValueError("ID must not be empty.")
    if not is_valid_identifier(v):
        raise ValueError(f"Invalid ID: {v}")
    return v


class ParameterScale(str, Enum):
    """Parameter scales.

    Parameter scales as used in the PEtab parameters table.
    """

    LIN = C.LIN
    LOG = C.LOG
    LOG10 = C.LOG10


class NoiseDistribution(str, Enum):
    """Noise distribution types.

    Noise distributions as used in the PEtab observables table.
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

    Prior types as used in the PEtab parameters table.
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
    #: Log10-normal distribution.
    LOG10_NORMAL = C.LOG10_NORMAL
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
    PriorDistribution.LOG10_NORMAL: Normal,
    PriorDistribution.LOG_LAPLACE: Laplace,
    PriorDistribution.LOG_NORMAL: Normal,
    PriorDistribution.LOG_UNIFORM: Uniform,
    PriorDistribution.NORMAL: Normal,
    PriorDistribution.RAYLEIGH: Rayleigh,
    PriorDistribution.UNIFORM: Uniform,
}

assert not (_mismatch := set(PriorDistribution) ^ set(_prior_to_cls)), (
    "PriorDistribution enum does not match _prior_to_cls. "
    f"Mismatches: {_mismatch}"
)


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

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="allow"
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

    def _placeholders(
        self, type_: Literal["observable", "noise"]
    ) -> set[sp.Symbol]:
        formula = (
            self.formula
            if type_ == "observable"
            else self.noise_formula
            if type_ == "noise"
            else None
        )
        if formula is None or formula.is_number:
            return set()

        if not (free_syms := formula.free_symbols):
            return set()

        # TODO: add field validator to check for 1-based consecutive numbering
        t = f"{re.escape(type_)}Parameter"
        o = re.escape(self.id)
        pattern = re.compile(rf"(?:^|\W)({t}\d+_{o})(?=\W|$)")
        return {s for s in free_syms if pattern.match(str(s))}

    @property
    def observable_placeholders(self) -> set[sp.Symbol]:
        """Placeholder symbols for the observable formula."""
        return self._placeholders("observable")

    @property
    def noise_placeholders(self) -> set[sp.Symbol]:
        """Placeholder symbols for the noise formula."""
        return self._placeholders("noise")


class ObservableTable(BaseModel):
    """PEtab observable table."""

    #: List of observables.
    observables: list[Observable]

    def __getitem__(self, observable_id: str) -> Observable:
        """Get an observable by ID."""
        for observable in self.observables:
            if observable.id == observable_id:
                return observable
        raise KeyError(f"Observable ID {observable_id} not found")

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ObservableTable:
        """Create an ObservableTable from a DataFrame."""
        if df is None:
            return cls(observables=[])

        df = get_observable_df(df)
        observables = [
            Observable(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(observables=observables)

    def to_df(self) -> pd.DataFrame:
        """Convert the ObservableTable to a DataFrame."""
        records = self.model_dump(by_alias=True)["observables"]
        for record in records:
            obs = record[C.OBSERVABLE_FORMULA]
            noise = record[C.NOISE_FORMULA]
            record[C.OBSERVABLE_FORMULA] = petab_math_str(obs)
            record[C.NOISE_FORMULA] = petab_math_str(noise)
        return pd.DataFrame(records).set_index([C.OBSERVABLE_ID])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ObservableTable:
        """Create an ObservableTable from a TSV file."""
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        """Write the ObservableTable to a TSV file."""
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=True)

    def __add__(self, other: Observable) -> ObservableTable:
        """Add an observable to the table."""
        if not isinstance(other, Observable):
            raise TypeError("Can only add Observable to ObservableTable")
        return ObservableTable(observables=self.observables + [other])

    def __iadd__(self, other: Observable) -> ObservableTable:
        """Add an observable to the table in place."""
        if not isinstance(other, Observable):
            raise TypeError("Can only add Observable to ObservableTable")
        self.observables.append(other)
        return self


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
    to all rows of the PEtab conditions table with the same condition ID.

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
    model_config = ConfigDict(populate_by_name=True, extra="allow")

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


class ConditionTable(BaseModel):
    """PEtab conditions table."""

    #: List of conditions.
    conditions: list[Condition] = []

    def __getitem__(self, condition_id: str) -> Condition:
        """Get a condition by ID."""
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        raise KeyError(f"Condition ID {condition_id} not found")

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ConditionTable:
        """Create a ConditionTable from a DataFrame."""
        if df is None or df.empty:
            return cls(conditions=[])

        conditions = []
        for condition_id, sub_df in df.groupby(C.CONDITION_ID):
            changes = [Change(**row) for row in sub_df.to_dict("records")]
            conditions.append(Condition(id=condition_id, changes=changes))

        return cls(conditions=conditions)

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

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ConditionTable:
        """Create a ConditionTable from a TSV file."""
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        """Write the ConditionTable to a TSV file."""
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Condition) -> ConditionTable:
        """Add a condition to the table."""
        if not isinstance(other, Condition):
            raise TypeError("Can only add Condition to ConditionTable")
        return ConditionTable(conditions=self.conditions + [other])

    def __iadd__(self, other: Condition) -> ConditionTable:
        """Add a condition to the table in place."""
        if not isinstance(other, Condition):
            raise TypeError("Can only add Condition to ConditionTable")
        self.conditions.append(other)
        return self

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

    This corresponds to a row of the PEtab experiments table.
    """

    #: The start time of the period in time units as defined in the model.
    time: Annotated[float, AfterValidator(_is_finite_or_neg_inf)] = Field(
        alias=C.TIME
    )
    #: The IDs of the conditions to be applied at the start time.
    condition_ids: list[str] = Field(default_factory=list)

    #: :meta private:
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @field_validator("condition_ids", mode="before")
    @classmethod
    def _validate_ids(cls, condition_ids):
        if condition_ids is None:
            return []

        for condition_id in condition_ids:
            # condition_id may be empty
            if condition_id and not is_valid_identifier(condition_id):
                raise ValueError(f"Invalid ID: `{condition_id}'")
        return condition_ids


class Experiment(BaseModel):
    """An experiment or a timecourse defined by an ID and a set of different
    periods.

    Corresponds to a group of rows of the PEtab experiments table with the same
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
        arbitrary_types_allowed=True, populate_by_name=True, extra="allow"
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


class ExperimentTable(BaseModel):
    """PEtab experiments table."""

    #: List of experiments.
    experiments: list[Experiment]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ExperimentTable:
        """Create an ExperimentTable from a DataFrame."""
        if df is None:
            return cls(experiments=[])

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
                        time=timepoint, condition_ids=condition_ids
                    )
                )
            experiments.append(Experiment(id=experiment_id, periods=periods))

        return cls(experiments=experiments)

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

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ExperimentTable:
        """Create an ExperimentTable from a TSV file."""
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        """Write the ExperimentTable to a TSV file."""
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Experiment) -> ExperimentTable:
        """Add an experiment to the table."""
        if not isinstance(other, Experiment):
            raise TypeError("Can only add Experiment to ExperimentTable")
        return ExperimentTable(experiments=self.experiments + [other])

    def __iadd__(self, other: Experiment) -> ExperimentTable:
        """Add an experiment to the table in place."""
        if not isinstance(other, Experiment):
            raise TypeError("Can only add Experiment to ExperimentTable")
        self.experiments.append(other)
        return self

    def __getitem__(self, item):
        """Get an experiment by ID."""
        for experiment in self.experiments:
            if experiment.id == item:
                return experiment
        raise KeyError(f"Experiment ID {item} not found")


class Measurement(BaseModel):
    """A measurement.

    A measurement of an observable at a specific time point in a specific
    experiment.
    """

    #: The observable ID.
    observable_id: str = Field(alias=C.OBSERVABLE_ID)
    #: The experiment ID.
    experiment_id: str | None = Field(alias=C.EXPERIMENT_ID, default=None)
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
        arbitrary_types_allowed=True, populate_by_name=True, extra="allow"
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

    @field_validator("observable_id", "experiment_id")
    @classmethod
    def _validate_id(cls, v, info: ValidationInfo):
        if not v:
            if info.field_name == "experiment_id":
                return None
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
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


class MeasurementTable(BaseModel):
    """PEtab measurement table."""

    #: List of measurements.
    measurements: list[Measurement]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
    ) -> MeasurementTable:
        """Create a MeasurementTable from a DataFrame."""
        if df is None:
            return cls(measurements=[])

        measurements = [
            Measurement(
                **row.to_dict(),
            )
            for _, row in df.reset_index().iterrows()
        ]

        return cls(measurements=measurements)

    def to_df(self) -> pd.DataFrame:
        """Convert the MeasurementTable to a DataFrame."""
        records = self.model_dump(by_alias=True)["measurements"]
        for record in records:
            record[C.OBSERVABLE_PARAMETERS] = C.PARAMETER_SEPARATOR.join(
                map(str, record[C.OBSERVABLE_PARAMETERS])
            )
            record[C.NOISE_PARAMETERS] = C.PARAMETER_SEPARATOR.join(
                map(str, record[C.NOISE_PARAMETERS])
            )

        return pd.DataFrame(records)

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> MeasurementTable:
        """Create a MeasurementTable from a TSV file."""
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        """Write the MeasurementTable to a TSV file."""
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Measurement) -> MeasurementTable:
        """Add a measurement to the table."""
        if not isinstance(other, Measurement):
            raise TypeError("Can only add Measurement to MeasurementTable")
        return MeasurementTable(measurements=self.measurements + [other])

    def __iadd__(self, other: Measurement) -> MeasurementTable:
        """Add a measurement to the table in place."""
        if not isinstance(other, Measurement):
            raise TypeError("Can only add Measurement to MeasurementTable")
        self.measurements.append(other)
        return self


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
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class MappingTable(BaseModel):
    """PEtab mapping table."""

    #: List of mappings.
    mappings: list[Mapping]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> MappingTable:
        """Create a MappingTable from a DataFrame."""
        if df is None:
            return cls(mappings=[])

        mappings = [
            Mapping(**row.to_dict()) for _, row in df.reset_index().iterrows()
        ]

        return cls(mappings=mappings)

    def to_df(self) -> pd.DataFrame:
        """Convert the MappingTable to a DataFrame."""
        res = (
            pd.DataFrame(self.model_dump(by_alias=True)["mappings"])
            if self.mappings
            else pd.DataFrame(columns=C.MAPPING_DF_REQUIRED_COLS)
        )
        return res.set_index([C.PETAB_ENTITY_ID])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> MappingTable:
        """Create a MappingTable from a TSV file."""
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        """Write the MappingTable to a TSV file."""
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Mapping) -> MappingTable:
        """Add a mapping to the table."""
        if not isinstance(other, Mapping):
            raise TypeError("Can only add Mapping to MappingTable")
        return MappingTable(mappings=self.mappings + [other])

    def __iadd__(self, other: Mapping) -> MappingTable:
        """Add a mapping to the table in place."""
        if not isinstance(other, Mapping):
            raise TypeError("Can only add Mapping to MappingTable")
        self.mappings.append(other)
        return self

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
    id: str = Field(alias=C.PARAMETER_ID)
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
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v

    @field_validator("prior_parameters", mode="before")
    @classmethod
    def _validate_prior_parameters(cls, v):
        if isinstance(v, float) and np.isnan(v):
            return []

        if isinstance(v, str):
            v = v.split(C.PARAMETER_SEPARATOR)
        elif not isinstance(v, Sequence):
            v = [v]

        return [float(x) for x in v]

    @field_validator("estimate", mode="before")
    @classmethod
    def _validate_estimate_before(cls, v):
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
    def _serialize_prior_parameters(self, prior_parameters: list[str], _info):
        return C.PARAMETER_SEPARATOR.join(prior_parameters)

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

        # TODO: also if not estimated?
        if (
            self.estimate
            and self.lb is not None
            and self.ub is not None
            and self.lb >= self.ub
        ):
            raise ValueError("Lower bound must be less than upper bound.")

        # TODO priorType, priorParameters

        return self

    @property
    def prior_dist(self) -> Distribution:
        """Get the pior distribution of the parameter."""
        if self.estimate is False:
            raise ValueError(f"Parameter `{self.id}' is not estimated.")

        if self.prior_distribution is None:
            return Uniform(self.lb, self.ub)

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
        return cls(*self.prior_parameters, log=log, trunc=[self.lb, self.ub])


class ParameterTable(BaseModel):
    """PEtab parameter table."""

    #: List of parameters.
    parameters: list[Parameter]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ParameterTable:
        """Create a ParameterTable from a DataFrame."""
        if df is None:
            return cls(parameters=[])

        parameters = [
            Parameter(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(parameters=parameters)

    def to_df(self) -> pd.DataFrame:
        """Convert the ParameterTable to a DataFrame."""
        return pd.DataFrame(
            self.model_dump(by_alias=True)["parameters"]
        ).set_index([C.PARAMETER_ID])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ParameterTable:
        """Create a ParameterTable from a TSV file."""
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        """Write the ParameterTable to a TSV file."""
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Parameter) -> ParameterTable:
        """Add a parameter to the table."""
        if not isinstance(other, Parameter):
            raise TypeError("Can only add Parameter to ParameterTable")
        return ParameterTable(parameters=self.parameters + [other])

    def __iadd__(self, other: Parameter) -> ParameterTable:
        """Add a parameter to the table in place."""
        if not isinstance(other, Parameter):
            raise TypeError("Can only add Parameter to ParameterTable")
        self.parameters.append(other)
        return self

    def __getitem__(self, item) -> Parameter:
        """Get a parameter by ID."""
        for parameter in self.parameters:
            if parameter.id == item:
                return parameter
        raise KeyError(f"Parameter ID {item} not found")

    @property
    def n_estimated(self) -> int:
        """Number of estimated parameters."""
        return sum(p.estimate for p in self.parameters)

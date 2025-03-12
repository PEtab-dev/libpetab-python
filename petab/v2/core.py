"""Types around the PEtab object model."""
from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ..v1.lint import is_valid_identifier
from ..v1.math import sympify_petab
from . import C

__all__ = [
    "Observable",
    "ObservablesTable",
    "ObservableTransformation",
    "NoiseDistribution",
    "Change",
    "ChangeSet",
    "ConditionsTable",
    "OperationType",
    "ExperimentPeriod",
    "Experiment",
    "ExperimentsTable",
    "Measurement",
    "MeasurementTable",
    "Mapping",
    "MappingTable",
    "Parameter",
    "ParameterScale",
    "ParameterTable",
]


class ObservableTransformation(str, Enum):
    """Observable transformation types.

    Observable transformations as used in the PEtab observables table.
    """

    LIN = C.LIN
    LOG = C.LOG
    LOG10 = C.LOG10


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

    NORMAL = C.NORMAL
    LAPLACE = C.LAPLACE


class ObjectivePriorType(str, Enum):
    """Objective prior types.

    Objective prior types as used in the PEtab parameters table.
    """

    NORMAL = C.NORMAL
    LAPLACE = C.LAPLACE
    UNIFORM = C.UNIFORM
    LOG_NORMAL = C.LOG_NORMAL
    LOG_LAPLACE = C.LOG_LAPLACE
    PARAMETER_SCALE_NORMAL = C.PARAMETER_SCALE_NORMAL
    PARAMETER_SCALE_LAPLACE = C.PARAMETER_SCALE_LAPLACE
    PARAMETER_SCALE_UNIFORM = C.PARAMETER_SCALE_UNIFORM


assert set(C.PRIOR_TYPES) == {e.value for e in ObjectivePriorType}, (
    "ObjectivePriorType enum does not match C.PRIOR_TYPES: "
    f"{set(C.PRIOR_TYPES)} vs { {e.value for e in ObjectivePriorType} }"
)


class InitializationPriorType(str, Enum):
    """Initialization prior types.

    Initialization prior types as used in the PEtab parameters table.
    """

    NORMAL = C.NORMAL
    LAPLACE = C.LAPLACE
    UNIFORM = C.UNIFORM
    LOG_NORMAL = C.LOG_NORMAL
    LOG_LAPLACE = C.LOG_LAPLACE
    PARAMETER_SCALE_NORMAL = C.PARAMETER_SCALE_NORMAL
    PARAMETER_SCALE_LAPLACE = C.PARAMETER_SCALE_LAPLACE
    PARAMETER_SCALE_UNIFORM = C.PARAMETER_SCALE_UNIFORM


class Observable(BaseModel):
    """Observable definition."""

    id: str = Field(alias=C.OBSERVABLE_ID)
    name: str | None = Field(alias=C.OBSERVABLE_NAME, default=None)
    formula: sp.Basic | None = Field(alias=C.OBSERVABLE_FORMULA, default=None)
    transformation: ObservableTransformation = Field(
        alias=C.OBSERVABLE_TRANSFORMATION, default=ObservableTransformation.LIN
    )
    noise_formula: sp.Basic | None = Field(alias=C.NOISE_FORMULA, default=None)
    noise_distribution: NoiseDistribution = Field(
        alias=C.NOISE_DISTRIBUTION, default=NoiseDistribution.NORMAL
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v

    @field_validator(
        "name",
        "formula",
        "noise_formula",
        "noise_formula",
        "noise_distribution",
        "transformation",
        mode="before",
    )
    @classmethod
    def convert_nan_to_default(cls, v, info: ValidationInfo):
        if isinstance(v, float) and np.isnan(v):
            return cls.model_fields[info.field_name].default
        return v

    @field_validator("formula", "noise_formula", mode="before")
    @classmethod
    def sympify(cls, v):
        if v is None or isinstance(v, sp.Basic):
            return v
        if isinstance(v, float) and np.isnan(v):
            return None

        return sympify_petab(v)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class ObservablesTable(BaseModel):
    """PEtab observables table."""

    observables: list[Observable]

    def __getitem__(self, observable_id: str) -> Observable:
        """Get an observable by ID."""
        for observable in self.observables:
            if observable.id == observable_id:
                return observable
        raise KeyError(f"Observable ID {observable_id} not found")

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ObservablesTable:
        if df is None:
            return cls(observables=[])

        observables = [
            Observable(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(observables=observables)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["observables"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ObservablesTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Observable) -> ObservablesTable:
        """Add an observable to the table."""
        if not isinstance(other, Observable):
            raise TypeError("Can only add Observable to ObservablesTable")
        return ObservablesTable(observables=self.observables + [other])

    def __iadd__(self, other: Observable) -> ObservablesTable:
        """Add an observable to the table in place."""
        if not isinstance(other, Observable):
            raise TypeError("Can only add Observable to ObservablesTable")
        self.observables.append(other)
        return self


class OperationType(str, Enum):
    """Operation types for model changes in the PEtab conditions table."""

    # TODO update names
    SET_CURRENT_VALUE = "setCurrentValue"
    NO_CHANGE = "noChange"
    ...


class Change(BaseModel):
    """A change to the model or model state.

    A change to the model or model state, corresponding to an individual
    row of the PEtab conditions table.
    """

    target_id: str | None = Field(alias=C.TARGET_ID, default=None)
    operation_type: OperationType = Field(alias=C.OPERATION_TYPE)
    target_value: sp.Basic | None = Field(alias=C.TARGET_VALUE, default=None)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        use_enum_values = True

    @model_validator(mode="before")
    @classmethod
    def validate_id(cls, data: dict):
        if (
            data.get("operation_type", data.get(C.OPERATION_TYPE))
            != C.OT_NO_CHANGE
        ):
            target_id = data.get("target_id", data.get(C.TARGET_ID))

            if not is_valid_identifier(target_id):
                raise ValueError(f"Invalid ID: {target_id}")
        return data

    @field_validator("target_value", mode="before")
    @classmethod
    def sympify(cls, v):
        if v is None or isinstance(v, sp.Basic):
            return v
        if isinstance(v, float) and np.isnan(v):
            return None

        return sympify_petab(v)


class ChangeSet(BaseModel):
    """A set of changes to the model or model state.

    A set of simultaneously occurring changes to the model or model state,
    corresponding to a perturbation of the underlying system. This corresponds
    to all rows of the PEtab conditions table with the same condition ID.
    """

    id: str = Field(alias=C.CONDITION_ID)
    changes: list[Change]

    class Config:
        populate_by_name = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v

    def __add__(self, other: Change) -> ChangeSet:
        """Add a change to the set."""
        if not isinstance(other, Change):
            raise TypeError("Can only add Change to ChangeSet")
        return ChangeSet(id=self.id, changes=self.changes + [other])

    def __iadd__(self, other: Change) -> ChangeSet:
        """Add a change to the set in place."""
        if not isinstance(other, Change):
            raise TypeError("Can only add Change to ChangeSet")
        self.changes.append(other)
        return self


class ConditionsTable(BaseModel):
    """PEtab conditions table."""

    conditions: list[ChangeSet] = []

    def __getitem__(self, condition_id: str) -> ChangeSet:
        """Get a condition by ID."""
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        raise KeyError(f"Condition ID {condition_id} not found")

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ConditionsTable:
        if df is None:
            return cls(conditions=[])

        conditions = []
        for condition_id, sub_df in df.groupby(C.CONDITION_ID):
            changes = [Change(**row.to_dict()) for _, row in sub_df.iterrows()]
            conditions.append(ChangeSet(id=condition_id, changes=changes))

        return cls(conditions=conditions)

    def to_df(self) -> pd.DataFrame:
        records = [
            {C.CONDITION_ID: condition.id, **change.model_dump()}
            for condition in self.conditions
            for change in condition.changes
        ]
        return pd.DataFrame(records)

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ConditionsTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: ChangeSet) -> ConditionsTable:
        """Add a condition to the table."""
        if not isinstance(other, ChangeSet):
            raise TypeError("Can only add ChangeSet to ConditionsTable")
        return ConditionsTable(conditions=self.conditions + [other])

    def __iadd__(self, other: ChangeSet) -> ConditionsTable:
        """Add a condition to the table in place."""
        if not isinstance(other, ChangeSet):
            raise TypeError("Can only add ChangeSet to ConditionsTable")
        self.conditions.append(other)
        return self


class ExperimentPeriod(BaseModel):
    """A period of a timecourse defined by a start time and a set changes.

    This corresponds to a row of the PEtab experiments table.
    """

    start: float = Field(alias=C.TIME)
    condition_id: str = Field(alias=C.CONDITION_ID)

    class Config:
        populate_by_name = True

    @field_validator("condition_id")
    @classmethod
    def validate_id(cls, condition_id):
        if not condition_id:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(condition_id):
            raise ValueError(f"Invalid ID: {condition_id}")
        return condition_id


class Experiment(BaseModel):
    """An experiment or a timecourse defined by an ID and a set of different
    periods.

    Corresponds to a group of rows of the PEtab experiments table with the same
    experiment ID.
    """

    id: str = Field(alias=C.EXPERIMENT_ID)
    periods: list[ExperimentPeriod] = []

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v

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


class ExperimentsTable(BaseModel):
    """PEtab experiments table."""

    experiments: list[Experiment]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ExperimentsTable:
        if df is None:
            return cls(experiments=[])

        experiments = []
        for experiment_id, cur_exp_df in df.groupby(C.EXPERIMENT_ID):
            periods = [
                ExperimentPeriod(
                    start=row[C.TIME], condition_id=row[C.CONDITION_ID]
                )
                for _, row in cur_exp_df.iterrows()
            ]
            experiments.append(Experiment(id=experiment_id, periods=periods))

        return cls(experiments=experiments)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["experiments"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ExperimentsTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_df()
        df.to_csv(file_path, sep="\t", index=False)

    def __add__(self, other: Experiment) -> ExperimentsTable:
        """Add an experiment to the table."""
        if not isinstance(other, Experiment):
            raise TypeError("Can only add Experiment to ExperimentsTable")
        return ExperimentsTable(experiments=self.experiments + [other])

    def __iadd__(self, other: Experiment) -> ExperimentsTable:
        """Add an experiment to the table in place."""
        if not isinstance(other, Experiment):
            raise TypeError("Can only add Experiment to ExperimentsTable")
        self.experiments.append(other)
        return self


class Measurement(BaseModel):
    """A measurement.

    A measurement of an observable at a specific time point in a specific
    experiment.
    """

    observable_id: str = Field(alias=C.OBSERVABLE_ID)
    experiment_id: str | None = Field(alias=C.EXPERIMENT_ID, default=None)
    time: float = Field(alias=C.TIME)
    measurement: float = Field(alias=C.MEASUREMENT)
    observable_parameters: list[sp.Basic] = Field(
        alias=C.OBSERVABLE_PARAMETERS, default_factory=list
    )
    noise_parameters: list[sp.Basic] = Field(
        alias=C.NOISE_PARAMETERS, default_factory=list
    )

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

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
    def validate_id(cls, v, info: ValidationInfo):
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
    def sympify_list(cls, v):
        if isinstance(v, float) and np.isnan(v):
            return []
        if isinstance(v, str):
            v = v.split(C.PARAMETER_SEPARATOR)
        else:
            v = [v]
        return [sympify_petab(x) for x in v]


class MeasurementTable(BaseModel):
    """PEtab measurement table."""

    measurements: list[Measurement]

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
    ) -> MeasurementTable:
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
        return pd.DataFrame(self.model_dump()["measurements"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> MeasurementTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
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

    petab_id: str = Field(alias=C.PETAB_ENTITY_ID)
    model_id: str = Field(alias=C.MODEL_ENTITY_ID)

    class Config:
        populate_by_name = True

    @field_validator(
        "petab_id",
    )
    @classmethod
    def validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v


class MappingTable(BaseModel):
    """PEtab mapping table."""

    mappings: list[Mapping]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> MappingTable:
        if df is None:
            return cls(mappings=[])

        mappings = [
            Mapping(**row.to_dict()) for _, row in df.reset_index().iterrows()
        ]

        return cls(mappings=mappings)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["mappings"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> MappingTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
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


class Parameter(BaseModel):
    """Parameter definition."""

    id: str = Field(alias=C.PARAMETER_ID)
    lb: float | None = Field(alias=C.LOWER_BOUND, default=None)
    ub: float | None = Field(alias=C.UPPER_BOUND, default=None)
    nominal_value: float | None = Field(alias=C.NOMINAL_VALUE, default=None)
    scale: ParameterScale = Field(
        alias=C.PARAMETER_SCALE, default=ParameterScale.LIN
    )
    estimate: bool = Field(alias=C.ESTIMATE, default=True)
    # TODO priors

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        use_enum_values = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v

    @field_validator("lb", "ub", "nominal_value")
    @classmethod
    def convert_nan_to_none(cls, v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class ParameterTable(BaseModel):
    """PEtab parameter table."""

    parameters: list[Parameter]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> ParameterTable:
        if df is None:
            return cls(parameters=[])

        parameters = [
            Parameter(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(parameters=parameters)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["parameters"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ParameterTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_df(df)

    def to_tsv(self, file_path: str | Path) -> None:
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

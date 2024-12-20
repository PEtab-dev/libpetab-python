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
    def convert_nan_to_none(cls, v, info: ValidationInfo):
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
    def from_dataframe(cls, df: pd.DataFrame) -> ObservablesTable:
        if df is None:
            return cls(observables=[])

        observables = [
            Observable(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(observables=observables)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["observables"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ObservablesTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_dataframe(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep="\t", index=False)


class OperationType(str, Enum):
    """Operation types for model changes in the PEtab conditions table."""

    # TODO update names
    SET_CURRENT_VALUE = "setCurrentValue"
    SET_RATE = "setRate"
    SET_ASSIGNMENT = "setAssignment"
    ADD_TO_RATE = "addToRate"
    ADD_TO_ASSIGNMENT = "addToAssignment"
    NO_CHANGE = "noChange"
    CONSTANT = "constant"
    INITIAL = "initial"
    ...


class Change(BaseModel):
    """A change to the model or model state.

    A change to the model or model state, corresponding to an individual
    row of the PEtab conditions table.
    """

    target_id: str = Field(alias=C.TARGET_ID)
    operation_type: OperationType = Field(alias=C.VALUE_TYPE)
    target_value: sp.Basic = Field(alias=C.TARGET_VALUE)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        use_enum_values = True

    @field_validator("target_id")
    @classmethod
    def validate_id(cls, v):
        if not v:
            raise ValueError("ID must not be empty.")
        if not is_valid_identifier(v):
            raise ValueError(f"Invalid ID: {v}")
        return v

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


class ConditionsTable(BaseModel):
    """PEtab conditions table."""

    conditions: list[ChangeSet]

    def __getitem__(self, condition_id: str) -> ChangeSet:
        """Get a condition by ID."""
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        raise KeyError(f"Condition ID {condition_id} not found")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ConditionsTable:
        if df is None:
            return cls(conditions=[])

        conditions = []
        for condition_id, sub_df in df.groupby(C.CONDITION_ID):
            changes = [Change(**row.to_dict()) for _, row in sub_df.iterrows()]
            conditions.append(ChangeSet(id=condition_id, changes=changes))

        return cls(conditions=conditions)

    def to_dataframe(self) -> pd.DataFrame:
        records = [
            {C.CONDITION_ID: condition.id, **change.model_dump()}
            for condition in self.conditions
            for change in condition.changes
        ]
        return pd.DataFrame(records)

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ConditionsTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_dataframe(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep="\t", index=False)


class ExperimentPeriod(BaseModel):
    """A period of a timecourse defined by a start time and a set changes.

    This corresponds to a row of the PEtab experiments table.
    """

    start: float = Field(alias=C.TIME)
    condition_ids: list[str] = Field(alias=C.CONDITION_ID)

    class Config:
        populate_by_name = True

    @field_validator("condition_ids")
    @classmethod
    def validate_id(cls, v):
        for condition_id in v:
            if not condition_id:
                raise ValueError("ID must not be empty.")
            if not is_valid_identifier(condition_id):
                raise ValueError(f"Invalid ID: {condition_id}")
        return v


class Experiment(BaseModel):
    """An experiment or a timecourse defined by an ID and a set of different
    periods.

    Corresponds to a group of rows of the PEtab experiments table with the same
    experiment ID.
    """

    id: str = Field(alias=C.EXPERIMENT_ID)
    periods: list[ExperimentPeriod]

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


class ExperimentsTable(BaseModel):
    """PEtab experiments table."""

    experiments: list[Experiment]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ExperimentsTable:
        if df is None:
            return cls(experiments=[])

        experiments = []
        for experiment_id, cur_exp_df in df.groupby(C.EXPERIMENT_ID):
            periods = []
            for time, cur_period_df in cur_exp_df.groupby(C.TIME):
                period_conditions = list(cur_period_df[C.CONDITION_ID])
                periods.append(
                    ExperimentPeriod(
                        start=time, condition_ids=period_conditions
                    )
                )
            experiments.append(Experiment(id=experiment_id, periods=periods))

        return cls(experiments=experiments)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["experiments"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ExperimentsTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_dataframe(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep="\t", index=False)


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
    def from_dataframe(
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

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["measurements"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> MeasurementTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_dataframe(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep="\t", index=False)


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
    def from_dataframe(cls, df: pd.DataFrame) -> MappingTable:
        if df is None:
            return cls(mappings=[])

        mappings = [
            Mapping(**row.to_dict()) for _, row in df.reset_index().iterrows()
        ]

        return cls(mappings=mappings)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["mappings"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> MappingTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_dataframe(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep="\t", index=False)


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
    def from_dataframe(cls, df: pd.DataFrame) -> ParameterTable:
        if df is None:
            return cls(parameters=[])

        parameters = [
            Parameter(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

        return cls(parameters=parameters)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.model_dump()["parameters"])

    @classmethod
    def from_tsv(cls, file_path: str | Path) -> ParameterTable:
        df = pd.read_csv(file_path, sep="\t")
        return cls.from_dataframe(df)

    def to_tsv(self, file_path: str | Path) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep="\t", index=False)

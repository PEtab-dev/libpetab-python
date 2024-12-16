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


class ObservableTransformation(str, Enum):
    """Observable transformation types.

    Observable transformations as used in the PEtab observables table.
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

    A set of simultaneously occuring changes to the model or model state,
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
    conditions: list[ChangeSet]

    class Config:
        populate_by_name = True


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


class ExperimentsTable(BaseModel):
    """PEtab experiments table."""

    experiments: list[Experiment]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> ExperimentsTable:
        if df is None:
            return cls(experiments=[])

        experiments = [
            Experiment(**row.to_dict())
            for _, row in df.reset_index().iterrows()
        ]

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

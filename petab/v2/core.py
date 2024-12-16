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
    LIN = C.LIN
    LOG = C.LOG
    LOG10 = C.LOG10


class NoiseDistribution(str, Enum):
    NORMAL = C.NORMAL
    LAPLACE = C.LAPLACE


class Observable(BaseModel):
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

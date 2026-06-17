from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import sympy as sp
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

import petab.v2.C as C

from ...v1.math.sympify import sympify_petab
from ..base import BaseTable

__all__ = [
    "Hybridization",
    "HybridizationTable",
    "NeuralNetConfig",
    "SciMLConfig",
]


class Hybridization(BaseModel):
    """Assigns PEtab SciML NN inputs and outputs."""

    #: The target ID.
    target_id: str = Field(alias=C.TARGET_ID)
    #: The target value.
    target_value: sp.Basic = Field(alias=C.TARGET_VALUE)

    #: :meta private:
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
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


class HybridizationTable(BaseTable[Hybridization]):
    """PEtab SciML hybridization table."""

    @property
    def hybridizations(self) -> list[Hybridization]:
        """List of hybridizations."""
        return self.elements

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> Self:
        """Create a HybridizationTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        hybridizations = [
            Hybridization(
                **row.to_dict(),
            )
            for _, row in df.iterrows()
        ]

        return cls(hybridizations, **kwargs)

    def to_df(self) -> pd.DataFrame:
        """Convert the HybridizationTable to a DataFrame."""
        records = self.model_dump(by_alias=True)["elements"]

        return pd.DataFrame(records)

    def __getitem__(self, target_id: str) -> Hybridization:
        """Get a hybridization by target ID."""
        for hybridization in self.hybridizations:
            if hybridization.target_id == target_id:
                return hybridization
        raise KeyError(f"Target ID {target_id} not found")

    def get(self, target_id, default=None):
        """Get a hybridization by target ID or return a default value."""
        try:
            return self[target_id]
        except KeyError:
            return default


class NeuralNetConfig(BaseModel):
    """A neural net in the PEtab SciML problem configuration."""

    location: AnyUrl | Path
    pre_initialization: bool
    format: str

    model_config = ConfigDict(
        validate_assignment=True,
    )


class SciMLConfig(BaseModel):
    """The extended configuration of a PEtab SciML problem."""

    #: The PEtab SciML format version.
    version: str = "0.1.0"
    #: The paths to the array data files.
    # Absolute or relative to `base_path`.
    array_files: list[AnyUrl | Path] = []
    #: The paths to the hybridization tables.
    # Absolute or relative to `base_path`.
    hybridization_files: list[AnyUrl | Path] = []
    #: The neural network IDs and info.
    neural_networks: dict[str, NeuralNetConfig] | None = {}

    model_config = ConfigDict(
        validate_assignment=True,
    )

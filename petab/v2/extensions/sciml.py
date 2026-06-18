"""PEtab SciML extension — classes and runtime state for hybrid ODE/ML
problems."""

from __future__ import annotations

from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator

from petab._utils import _generate_path
from petab.v1.math import sympify_petab

try:
    from petab_sciml import (
        ArrayData,
        ArrayDataStandard,
        NNModel,
        NNModelStandard,
    )
except ModuleNotFoundError:
    pass

from .. import C


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


class HybridizationTable:
    """PEtab SciML hybridization table."""

    def __init__(self, hybridizations: list[Hybridization] = None, **kwargs):
        self.hybridizations: list[Hybridization] = hybridizations or []
        self.rel_path: AnyUrl | Path | None = kwargs.get("rel_path")
        self.base_path: AnyUrl | Path | None = kwargs.get("base_path")

    @property
    def elements(self) -> list[Hybridization]:
        return self.hybridizations

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs) -> HybridizationTable:
        """Create a HybridizationTable from a DataFrame."""
        if df is None:
            return cls(**kwargs)

        hybridizations = [
            Hybridization(**row.to_dict()) for _, row in df.iterrows()
        ]
        return cls(hybridizations, **kwargs)

    @classmethod
    def from_tsv(
        cls,
        file_path: str | Path,
        base_path: str | Path | None = None,
    ) -> HybridizationTable:
        """Create a HybridizationTable from a TSV file."""
        df = pd.read_csv(_generate_path(file_path, base_path), sep="\t")
        return cls.from_df(df, rel_path=file_path, base_path=base_path)

    def to_df(self) -> pd.DataFrame:
        """Convert the HybridizationTable to a DataFrame."""
        records = [h.model_dump(by_alias=True) for h in self.hybridizations]
        return pd.DataFrame(records)

    def to_tsv(self, file_path: str | Path = None) -> None:
        """Write the table to a TSV file."""
        df = self.to_df()
        df.to_csv(
            file_path or _generate_path(self.rel_path, self.base_path),
            sep="\t",
            index=False,
        )

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
    array_files: list[AnyUrl | Path] = []
    #: The paths to the hybridization tables.
    hybridization_files: list[AnyUrl | Path] = []
    #: The neural network IDs and info.
    neural_networks: dict[str, NeuralNetConfig] | None = {}

    model_config = ConfigDict(
        validate_assignment=True,
    )


class SciMLExt:
    """SciML extension runtime state.

    Accessible as ``Problem.extensions.sciml``.
    """

    def __init__(
        self,
        neural_networks: list = None,
        hybridization_tables: list[HybridizationTable] = None,
        array_data_files: list = None,
    ):
        self.neural_networks: list = neural_networks or []
        self.hybridization_tables: list[HybridizationTable] = (
            hybridization_tables or [HybridizationTable()]
        )
        self.array_data_files: list = array_data_files or []

    @property
    def hybridizations(self) -> list[Hybridization]:
        """Flat list of all hybridizations across all hybridization tables."""
        return list(
            chain.from_iterable(
                ht.hybridizations for ht in self.hybridization_tables
            )
        )

    @property
    def hybridization_df(self) -> pd.DataFrame | None:
        """Combined hybridization tables as a single DataFrame."""
        hybs = self.hybridizations
        return HybridizationTable(hybs).to_df() if hybs else None

    @hybridization_df.setter
    def hybridization_df(self, value: pd.DataFrame):
        self.hybridization_tables = [HybridizationTable.from_df(value)]

    def add_hybridization(self, target_id: str, target_value: str):
        """Add a hybridization entry.

        If there is more than one hybridization table the entry is added to
        the last one.

        Arguments:
            target_id: The ID of the target entity in the PEtab problem
                or neural network model
            target_value: The value that is assigned to the target id.
        """
        if not self.hybridization_tables:
            self.hybridization_tables.append(HybridizationTable())
        self.hybridization_tables[-1].hybridizations.append(
            Hybridization(target_id=target_id, target_value=target_value)
        )

    def add_neural_network_from_dict(self, model_id: str, nn_dict: dict):
        """Add a neural network from a dictionary."""
        nn_model = NNModel.model_validate(nn_dict)
        nn_model.nn_model_id = model_id
        self.neural_networks.append(nn_model)

    def add_neural_network_from_yaml(
        self,
        model_id: str,
        file_path: str | Path,
        base_path: str | Path | None = None,
    ):
        """Add a neural network from a YAML file."""
        self.neural_networks.append(
            NNModelStandard.load_data(
                _generate_path(file_path=file_path, base_path=base_path),
                nn_model_id=model_id,
            )
        )

    def add_array_data_from_dict(self, array_data: dict):
        """Add array data from a dictionary."""
        self.array_data_files.append(ArrayData.model_validate(array_data))

    def add_array_data_from_hdf5(
        self,
        file_path: str | Path,
        base_path: str | Path | None = None,
    ):
        """Add array data from an HDF5 file."""
        self.array_data_files.append(
            ArrayDataStandard.load_data(_generate_path(file_path, base_path))
        )

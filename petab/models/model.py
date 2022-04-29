"""PEtab model abstraction"""
from __future__ import annotations

import abc
from typing import Any, Iterable, Tuple


class Model:
    """Base class for wrappers for any PEtab-supported model type"""

    def __init__(self):
        ...
    # TODO more coherent method names / arguments
    # TODO doc
    # TODO remove unused

    @staticmethod
    @abc.abstractmethod
    def from_file(filepath_or_buffer: Any) -> Model:
        ...

    @abc.abstractmethod
    def get_parameter_ids(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def get_parameter_value(self, id_: str) -> float:
        ...

    @abc.abstractmethod
    def get_parameter_ids_with_values(self) -> Iterable[Tuple[str, float]]:
        ...

    @abc.abstractmethod
    def has_species_with_id(self, entity_id: str) -> bool:
        ...

    @abc.abstractmethod
    def has_compartment_with_id(self, entity_id: str) -> bool:
        ...

    @abc.abstractmethod
    def has_entity_with_id(self, entity_id) -> bool:
        ...

    @abc.abstractmethod
    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def get_valid_ids_for_condition_table(self) -> Iterable[str]:
        ...

    @abc.abstractmethod
    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        ...

    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate model

        :returns: `True` if errors occurred, `False` otherwise.
        """
        ...


def model_factory(filepath_or_buffer: Any, model_language: str) -> Model:
    """Create a PEtab model instance from the given model"""
    if model_language == "sbml":
        from .sbml_model import SbmlModel
        return SbmlModel.from_file(filepath_or_buffer)

    raise ValueError(f"Unsupported model format: {model_language}")

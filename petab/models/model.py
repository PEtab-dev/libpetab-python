"""PEtab model abstraction"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Iterable, Tuple


class Model(abc.ABC):
    """Base class for wrappers for any PEtab-supported model type"""

    def __init__(self):
        ...

    @staticmethod
    @abc.abstractmethod
    def from_file(filepath_or_buffer: Any, model_id: str) -> Model:
        """Load the model from the given path/URL

        :param filepath_or_buffer: URL or path of the model
        :param model_id: Model ID
        :returns: A ``Model`` instance holding the given model
        """
        ...

    @abc.abstractmethod
    def to_file(self, filename: [str, Path]):
        """Save the model to the given file

        :param filename: Destination filename
        """
        ...

    @classmethod
    @property
    @abc.abstractmethod
    def type_id(cls):
        ...

    @property
    @abc.abstractmethod
    def model_id(self):
        ...

    @abc.abstractmethod
    def get_parameter_value(self, id_: str) -> float:
        """Get a parameter value

        :param id_: ID of the parameter whose value is to be returned
        :raises ValueError: If no parameter with the given ID exists
        :returns: The value of the given parameter as specified in the model
        """
        ...

    @abc.abstractmethod
    def get_free_parameter_ids_with_values(
            self
    ) -> Iterable[Tuple[str, float]]:
        """Get free model parameters along with their values

        Returns:
            Iterator over tuples of (parameter_id, parameter_value)
        """
        ...

    @abc.abstractmethod
    def get_parameter_ids(self) -> Iterable[str]:
        """Get all parameter IDs from this model

        :returns: Iterator over model parameter IDs
        """
        ...

    @abc.abstractmethod
    def has_entity_with_id(self, entity_id) -> bool:
        """Check if there is a model entity with the given ID

        :param entity_id: ID to check for
        :returns: ``True``, if there is an entity with the given ID,
        ``False`` otherwise
        """
        ...

    @abc.abstractmethod
    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        """Get IDs of all parameters that are allowed to occur in the PEtab
        parameters table

        :returns: Iterator over parameter IDs
        """
        ...

    @abc.abstractmethod
    def get_valid_ids_for_condition_table(self) -> Iterable[str]:
        """Get IDs of all model entities that are allowed to occur as columns
        in the PEtab conditions table.

        :returns: Iterator over model entity IDs
        """
        ...

    @abc.abstractmethod
    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        """Check if the given ID is allowed to be used in observable and noise
        formulas

        :returns: ``True``, if allowed, ``False`` otherwise
        """

        ...

    @abc.abstractmethod
    def is_valid(self) -> bool:
        """Validate this model

        :returns: `True` if the model is valid, `False` if there are errors in
        this model
        """
        ...

    @abc.abstractmethod
    def is_state_variable(self, id_: str) -> bool:
        """Check whether the given ID corresponds to a model state variable"""
        ...


def model_factory(
        filepath_or_buffer: Any,
        model_language: str,
        model_id: str = None
) -> Model:
    """Create a PEtab model instance from the given model

    :param filepath_or_buffer: Path/URL of the model
    :param model_language: PEtab model language ID for the given model
    :param model_id: PEtab model ID for the given model
    :returns: A :py:class:`Model` instance representing the given model
    """
    from . import MODEL_TYPE_SBML, MODEL_TYPE_PYSB, known_model_types

    if model_language == MODEL_TYPE_SBML:
        from .sbml_model import SbmlModel
        return SbmlModel.from_file(filepath_or_buffer, model_id=model_id)

    if model_language == MODEL_TYPE_PYSB:
        from .pysb_model import PySBModel
        return PySBModel.from_file(filepath_or_buffer, model_id=model_id)

    if model_language in known_model_types:
        raise NotImplementedError(
            f"Unsupported model format: {model_language}")

    raise ValueError(f"Unknown model format: {model_language}")

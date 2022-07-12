"""Functions for handling PySB models"""

import itertools
import sys
from pathlib import Path
from typing import Iterable, Tuple, Union

import pysb

from . import MODEL_TYPE_PYSB
from .model import Model


def _pysb_model_from_path(pysb_model_file: Union[str, Path]) -> pysb.Model:
    """Load a pysb model module and return the :class:`pysb.Model` instance

    :param pysb_model_file: Full or relative path to the PySB model module
    :return: The pysb Model instance
    """
    pysb_model_file = Path(pysb_model_file)
    pysb_model_module_name = pysb_model_file.with_suffix('').name

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        pysb_model_module_name, pysb_model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[pysb_model_module_name] = module
    spec.loader.exec_module(module)

    # find a pysb.Model instance in the module
    # 1) check if module.model exists and is a pysb.Model
    model = getattr(module, 'model', None)
    if model:
        return model

    # 2) check if there is any other pysb.Model instance
    for x in dir(module):
        attr = getattr(module, x)
        if isinstance(attr, pysb.Model):
            return attr

    raise ValueError(f"Could not find any pysb.Model in {pysb_model_file}.")


class PySBModel(Model):
    """PEtab wrapper for PySB models"""
    type_id = MODEL_TYPE_PYSB

    def __init__(
            self,
            model: pysb.Model,
            model_id: str
    ):
        super().__init__()

        self.model = model
        self._model_id = model_id

    @staticmethod
    def from_file(filepath_or_buffer, model_id: str):
        return PySBModel(
            model=_pysb_model_from_path(filepath_or_buffer),
            model_id=model_id
        )

    def to_file(self, filename: [str, Path]):
        from pysb.export import export
        model_source = export(self.model, 'pysb_flat')
        with open(filename, 'w') as f:
            f.write(model_source)

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    def get_parameter_ids(self) -> Iterable[str]:
        return (p.name for p in self.model.parameters)

    def get_parameter_value(self, id_: str) -> float:
        try:
            return self.model.parameters[id_].name
        except KeyError as e:
            raise ValueError(f"Parameter {id_} does not exist.") from e

    def get_free_parameter_ids_with_values(
            self
    ) -> Iterable[Tuple[str, float]]:
        return ((p.name, p.value) for p in self.model.parameters)

    def has_entity_with_id(self, entity_id) -> bool:
        try:
            _ = self.model.components[entity_id]
            return True
        except KeyError:
            return False

    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        # all parameters are allowed in the parameter table
        return self.get_parameter_ids()

    def get_valid_ids_for_condition_table(self) -> Iterable[str]:
        # TODO what else is allowed?
        # TODO compartments (size vs initial size)
        return self.get_parameter_ids()

    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        return id_ in (
            x.name for x in itertools.chain(
                self.model.parameters,
                self.model.observables,
                self.model.expressions,
            )
        )

    def is_valid(self) -> bool:
        # PySB models are always valid
        return True

    def is_state_variable(self, id_: str) -> bool:
        # TODO can we handle that without network generation?
        return False

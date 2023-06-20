"""Functions for handling SBML models"""

import itertools
from pathlib import Path
from typing import Iterable, Optional, Tuple

import libsbml

from . import MODEL_TYPE_SBML
from .model import Model
from ..sbml import (get_sbml_model, is_sbml_consistent, load_sbml_from_string,
                    write_sbml)


class SbmlModel(Model):
    """PEtab wrapper for SBML models"""

    type_id = MODEL_TYPE_SBML

    def __init__(
            self,
            sbml_model: libsbml.Model = None,
            sbml_reader: libsbml.SBMLReader = None,
            sbml_document: libsbml.SBMLDocument = None,
            model_id: str = None
    ):
        super().__init__()

        self.sbml_reader: Optional[libsbml.SBMLReader] = sbml_reader
        self.sbml_document: Optional[libsbml.SBMLDocument] = sbml_document
        self.sbml_model: Optional[libsbml.Model] = sbml_model

        self._model_id = model_id or sbml_model.getIdAttribute()

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()

        # libsbml stuff cannot be serialized directly
        if self.sbml_model:
            sbml_document = self.sbml_model.getSBMLDocument()
            sbml_writer = libsbml.SBMLWriter()
            state['sbml_string'] = sbml_writer.writeSBMLToString(sbml_document)

        exclude = ['sbml_reader', 'sbml_document', 'sbml_model']
        for key in exclude:
            state.pop(key)

        return state

    def __setstate__(self, state):
        """Set state after unpickling"""
        # load SBML model from pickled string
        sbml_string = state.pop('sbml_string', None)
        if sbml_string:
            self.sbml_reader, self.sbml_document, self.sbml_model = \
                load_sbml_from_string(sbml_string)

        self.__dict__.update(state)

    @staticmethod
    def from_file(filepath_or_buffer, model_id: str = None):
        sbml_reader, sbml_document, sbml_model = get_sbml_model(
            filepath_or_buffer)
        return SbmlModel(
            sbml_model=sbml_model,
            sbml_reader=sbml_reader,
            sbml_document=sbml_document,
            model_id=model_id,
        )

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    def to_file(self, filename: [str, Path]):
        write_sbml(self.sbml_document or self.sbml_model.getSBMLDocument(),
                   filename)

    def get_parameter_value(self, id_: str) -> float:
        parameter = self.sbml_model.getParameter(id_)
        if not parameter:
            raise ValueError(f"Parameter {id_} does not exist.")
        return parameter.getValue()

    def get_free_parameter_ids_with_values(
            self
    ) -> Iterable[Tuple[str, float]]:
        rule_targets = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (
            (p.getId(), p.getValue())
            for p in self.sbml_model.getListOfParameters()
            if p.getId() not in rule_targets
        )

    def get_parameter_ids(self) -> Iterable[str]:
        rule_targets = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (
            p.getId() for p in self.sbml_model.getListOfParameters()
            if p.getId() not in rule_targets
        )

    def get_parameter_ids_with_values(self) -> Iterable[Tuple[str, float]]:
        rule_targets = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (
            (p.getId(), p.getValue())
            for p in self.sbml_model.getListOfParameters()
            if p.getId() not in rule_targets
        )

    def has_entity_with_id(self, entity_id) -> bool:
        return self.sbml_model.getElementBySId(entity_id) is not None

    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        # All parameters except rule-targets
        disallowed_set = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (p.getId() for p in self.sbml_model.getListOfParameters()
                if p.getId() not in disallowed_set)

    def get_valid_ids_for_condition_table(self) -> Iterable[str]:
        return (
            x.getId() for x in itertools.chain(
                self.sbml_model.getListOfParameters(),
                self.sbml_model.getListOfSpecies(),
                self.sbml_model.getListOfCompartments()
            )
        )

    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        return self.sbml_model.getElementBySId(id_) or id_ == 'time'

    def is_valid(self) -> bool:
        return is_sbml_consistent(self.sbml_model.getSBMLDocument())

    def is_state_variable(self, id_: str) -> bool:
        return (self.sbml_model.getSpecies(id_) is not None
                or self.sbml_model.getCompartment(id_) is not None
                or self.sbml_model.getRuleByVariable(id_) is not None)

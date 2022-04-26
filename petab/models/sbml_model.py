"""Functions for handling SBML models"""

from .model import Model
from ..sbml import get_sbml_model
import libsbml
from typing import Optional, Iterable, Tuple


class SbmlModel(Model):
    """PEtab wrapper for SBML models"""
    def __init__(
            self,
            sbml_model: libsbml.Model = None,
            sbml_reader: libsbml.SBMLReader = None,
            sbml_document: libsbml.SBMLDocument = None,
    ):
        super().__init__()

        self.sbml_reader: Optional[libsbml.SBMLReader] = sbml_reader
        self.sbml_document: Optional[libsbml.SBMLDocument] = sbml_document
        self.sbml_model: Optional[libsbml.Model] = sbml_model

    @staticmethod
    def from_file(filepath_or_buffer):
        sbml_reader, sbml_document, sbml_model = get_sbml_model(
            filepath_or_buffer)
        return SbmlModel(
            sbml_model=sbml_model,
            sbml_reader=sbml_reader,
            sbml_document=sbml_document,
        )

    def get_parameter_ids(self) -> Iterable[str]:
        return (
            p.getId() for p in self.sbml_model.getListOfParameters()
            if self.sbml_model.getAssignmentRuleByVariable(p.getId()) is None
        )

    def get_parameter_ids_with_values(self) -> Iterable[Tuple[str, float]]:
        return (
            (p.getId(), p.getValue())
            for p in self.sbml_model.getListOfParameters()
            if self.sbml_model.getAssignmentRuleByVariable(p.getId()) is None
        )

    def has_species_with_id(self, entity_id: str) -> bool:
        return self.sbml_model.getSpecies(entity_id) is not None

    def has_compartment_with_id(self, entity_id: str) -> bool:
        return self.sbml_model.getCompartment(entity_id) is not None

    def has_entity_with_id(self, entity_id) -> bool:
        return self.sbml_model.getElementBySId(entity_id) is not None

    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        # TODO
        pass

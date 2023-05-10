"""Functions for handling PySB models"""

import itertools
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
            return self.model.parameters[id_].value
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
        return itertools.chain(self.get_parameter_ids(),
                               self.get_compartment_ids())

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
        # If there is a component with that name, it's not a state variable
        # (there are no dynamically-sized compartments)
        if self.model.components.get(id_, None):
            return False

        # Try parsing the ID
        try:
            result = parse_species_name(id_)
        except ValueError:
            return False
        else:
            # check if the ID is plausible
            for monomer, compartment, site_config in result:
                pysb_monomer: pysb.Monomer = self.model.monomers.get(monomer)
                if pysb_monomer is None:
                    return False
                if compartment:
                    pysb_compartment = self.model.compartments.get(compartment)
                    if pysb_compartment is None:
                        return False
                for site, state in site_config.items():
                    if site not in pysb_monomer.sites:
                        return False
                    if state not in pysb_monomer.site_states[site]:
                        return False
                if set(pysb_monomer.sites) - set(site_config.keys()):
                    # There are undefined sites
                    return False
            return True

    def get_compartment_ids(self) -> Iterable[str]:
        return (compartment.name for compartment in self.model.compartments)


def parse_species_name(
        name: str
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """Parse a PySB species name

    :param name: Species name to parse
    :returns: List of species, representing complex constituents, each as
        a tuple of the monomer name, the compartment name, and a dict of sites
        mapping to site states.
    :raises ValueError: In case this is not a valid ID
    """
    if '=MultiState(' in name:
        raise NotImplementedError("MultiState is not yet supported.")

    complex_constituent_pattern = re.compile(
        r'^(?P<monomer>\w+)\((?P<site_config>.*)\)'
        r'( \*\* (?P<compartment>.*))?$'
    )
    result = []
    complex_constituents = name.split(" % ")

    for complex_constituent in complex_constituents:
        match = complex_constituent_pattern.match(complex_constituent)
        if not match:
            raise ValueError(f"Invalid species name: '{name}' "
                             f"('{complex_constituent}')")
        monomer = match.groupdict()['monomer']
        site_config_str = match.groupdict()['site_config']
        compartment = match.groupdict()['compartment']

        site_config = {}
        for site_str in site_config_str.split(", "):
            if not site_str:
                continue
            site, config = site_str.split("=")
            if config == 'None':
                config = None
            elif config.startswith("'"):
                if not config.endswith("'"):
                    raise ValueError(f"Invalid species name: '{name}' "
                                     f"('{config}')")
                # strip quotes
                config = config[1:-1]
            else:
                config = int(config)
            site_config[site] = config
        result.append((monomer, compartment, site_config),)

    return result


def pattern_from_string(string: str, model: pysb.Model) -> pysb.ComplexPattern:
    """Convert a pattern string to a Pattern instance"""
    parts = parse_species_name(string)
    patterns = []
    for part in parts:
        patterns.append(
            pysb.MonomerPattern(
                monomer=model.monomers.get(part[0]),
                compartment=model.compartments.get(part[1], None),
                site_conditions=part[2]
            ))

    return pysb.ComplexPattern(patterns, compartment=None)

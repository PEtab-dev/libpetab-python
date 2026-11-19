"""Functions for handling PySB models"""

from __future__ import annotations

import ast
import itertools
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pysb

from ..._utils import _generate_path
from .. import is_valid_identifier
from . import MODEL_TYPE_PYSB
from .model import Model

__all__ = ["PySBModel", "parse_species_name", "pattern_from_string"]


def _pysb_model_from_path(pysb_model_file: str | Path) -> pysb.Model:
    """Load a pysb model module and return the :class:`pysb.Model` instance

    :param pysb_model_file: Full or relative path to the PySB model module
    :return: The pysb Model instance
    """
    pysb_model_file = Path(pysb_model_file)
    pysb_model_module_name = pysb_model_file.with_suffix("").name

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        pysb_model_module_name, pysb_model_file
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pysb_model_module_name] = module
    spec.loader.exec_module(module)

    # find a pysb.Model instance in the module
    # 1) check if module.model exists and is a pysb.Model
    model = getattr(module, "model", None)
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
        model_id: str | None = None,
        rel_path: Path | str | None = None,
        base_path: str | Path | None = None,
    ):
        super().__init__()

        self.rel_path = rel_path
        self.base_path = base_path

        self.model = model
        self._model_id = model_id or self.model.name

        if not is_valid_identifier(self._model_id):
            raise ValueError(
                f"Model ID '{self._model_id}' is not a valid identifier. "
                "Either provide a valid identifier or change the model name "
                "to a valid PEtab model identifier."
            )

    @staticmethod
    def from_file(
        filepath_or_buffer,
        model_id: str | None = None,
        base_path: str | Path | None = None,
    ) -> PySBModel:
        return PySBModel(
            model=_pysb_model_from_path(
                _generate_path(filepath_or_buffer, base_path)
            ),
            model_id=model_id,
            rel_path=filepath_or_buffer,
            base_path=base_path,
        )

    def to_file(self, filename: str | Path | None = None) -> None:
        model_source = self.to_str()

        with open(
            filename or _generate_path(self.rel_path, self.base_path), "w"
        ) as f:
            f.write(model_source)

    def to_str(self) -> str:
        """Get the PySB model Python code as a string."""
        from pysb.export import export

        return export(self.model, "pysb_flat")

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
        self,
    ) -> Iterable[tuple[str, float]]:
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
        return itertools.chain(
            self.get_parameter_ids(), self.get_compartment_ids()
        )

    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        return id_ in (
            x.name
            for x in itertools.chain(
                self.model.parameters,
                self.model.observables,
                self.model.expressions,
            )
        )

    def is_valid(self) -> bool:
        # PySB models are always valid
        return True

    def is_state_variable(self, id_: str) -> bool:
        """Check whether the given ID is a plausible PySB species name.

        The ID has to be a species pattern (e.g., ``A(b=1, s='u') % B(a=1)``)
        with all sites of all monomers specified and all bonds paired.
        For models with compartments, the pattern must specify either a
        compartment for the whole complex or one for every monomer.
        This does not check whether the species is actually generated by
        the model's rules, or whether all monomers of a complex are
        connected.
        """
        # If there is a component with that name, it's not a state variable
        # (there are no dynamically-sized compartments)
        if self.model.components.get(id_, None):
            return False

        try:
            pattern = pattern_from_string(id_, self.model)
        except ValueError:
            return False

        if not pattern.is_concrete():
            return False

        for mp in pattern.monomer_patterns:
            # is_site_concrete() doesn't check this for duplicate sites
            for site, count in Counter(mp.monomer.sites).items():
                if site not in mp.site_conditions:
                    return False
                site_value = mp.site_conditions[site]
                if (
                    isinstance(site_value, pysb.MultiState)
                    and len(site_value) != count
                ):
                    return False

        # each bond has to connect exactly two sites
        bond_counts = Counter(
            bond
            for mp in pattern.monomer_patterns
            for site_value in mp.site_conditions.values()
            for bond in _iter_bonds(site_value)
        )
        return all(count == 2 for count in bond_counts.values())

    def get_compartment_ids(self) -> Iterable[str]:
        return (compartment.name for compartment in self.model.compartments)


def parse_species_name(
    name: str,
) -> list[tuple[str, str | None, dict[str, Any]]]:
    """Parse a PySB species name

    Site values in the form of ``(state, bond)`` tuples and
    :class:`pysb.MultiState` are not supported.
    See :func:`pattern_from_string` for a more complete parser.

    :param name: Species name to parse
    :returns: List of species, representing complex constituents, each as
        a tuple of the monomer name, the compartment name, and a dict of sites
        mapping to site states.
    :raises ValueError: In case this is not a valid ID
    """
    if "=MultiState(" in name:
        raise NotImplementedError("MultiState is not yet supported.")

    complex_constituent_pattern = re.compile(
        r"^(?P<monomer>\w+)\((?P<site_config>.*)\)"
        r"( \*\* (?P<compartment>.*))?$"
    )
    result = []
    complex_constituents = name.split(" % ")

    for complex_constituent in complex_constituents:
        match = complex_constituent_pattern.match(complex_constituent)
        if not match:
            raise ValueError(
                f"Invalid species name: '{name}' ('{complex_constituent}')"
            )
        monomer = match.groupdict()["monomer"]
        site_config_str = match.groupdict()["site_config"]
        compartment = match.groupdict()["compartment"]

        site_config = {}
        for site_str in site_config_str.split(", "):
            if not site_str:
                continue
            site, config = site_str.split("=")
            if config == "None":
                config = None
            elif config.startswith("'"):
                if not config.endswith("'"):
                    raise ValueError(
                        f"Invalid species name: '{name}' ('{config}')"
                    )
                # strip quotes
                config = config[1:-1]
            else:
                config = int(config)
            site_config[site] = config
        result.append(
            (monomer, compartment, site_config),
        )

    return result


def pattern_from_string(string: str, model: pysb.Model) -> pysb.ComplexPattern:
    """Convert a pattern string to a pattern of the given model.

    The string has to be in the format of ``str(pysb.ComplexPattern)``,
    e.g., ``A(b=1, s=('u', 2)) ** c % B(a=1, c=2)``.
    Site values may be ``None``, bond numbers, lists of bond numbers,
    state strings, ``(state, bond)`` tuples, or ``MultiState(...)``.
    Wildcards such as ``ANY`` or ``WILD`` are not supported.

    :param string: The pattern string.
    :param model: The model that defines the monomers and compartments.
    :returns: The pattern.
    :raises ValueError: If the string is not a valid pattern for this model.
    """

    class InvalidPattern(ValueError):
        pass

    def invalid(node: ast.AST | None = None) -> InvalidPattern:
        detail = f" ('{ast.unparse(node)}')" if node is not None else ""
        return InvalidPattern(f"Invalid species name: '{string}'{detail}")

    def convert_site_value(node: ast.expr):
        """Convert the AST of a site value to the PySB site condition."""
        if isinstance(node, ast.Constant) and (
            node.value is None or type(node.value) in (int, str)
        ):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(convert_site_value(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [convert_site_value(e) for e in node.elts]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "MultiState"
            and not node.keywords
        ):
            return pysb.MultiState(*map(convert_site_value, node.args))
        raise invalid(node)

    def convert(node: ast.expr):
        """Convert the AST of a (complex/monomer) pattern to a PySB pattern."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            monomer = model.monomers.get(node.func.id)
            if monomer is None or node.args:
                raise invalid(node)
            sites = [kw.arg for kw in node.keywords]
            if (
                None in sites
                or len(set(sites)) != len(sites)
                or set(sites) - set(monomer.sites)
            ):
                raise invalid(node)
            site_conditions = {
                kw.arg: convert_site_value(kw.value) for kw in node.keywords
            }
            # not `monomer(**site_conditions)`, which fails for a site
            #  named `conditions`
            return pysb.MonomerPattern(monomer, site_conditions, None)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return convert(node.left) % convert(node.right)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and isinstance(node.right, ast.Name)
            and (compartment := model.compartments.get(node.right.id))
        ):
            return convert(node.left) ** compartment
        raise invalid(node)

    try:
        tree = ast.parse(string, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as e:
        # RecursionError / MemoryError for too deeply nested expressions
        raise invalid() from e

    try:
        return pysb.as_complex_pattern(convert(tree.body))
    except InvalidPattern:
        raise
    except Exception as e:
        # PySB raises ValueError, KeyError, or bare Exception
        #  for invalid patterns
        raise invalid() from e


def _iter_bonds(site_value) -> Iterator[int]:
    """Iterate over the bond numbers in a site value."""
    if isinstance(site_value, int):
        yield site_value
    elif isinstance(site_value, tuple | list | pysb.MultiState):
        for x in site_value:
            yield from _iter_bonds(x)

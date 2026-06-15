"""Functions for handling BNGL (BioNetGen Language) models.

``petab`` ships ``sbml`` and ``pysb`` model loaders; this module adds
``bngl`` so that a ``language: bngl`` PEtab problem can be loaded and
validated at the model level (see PEtab-dev/PEtab#436).

:class:`BnglModel` is a :class:`petab.v1.models.model.Model` backed by a
small, dependency-free BNGL *block reader* (:func:`parse_bngl`). PEtab
validation only ever introspects a model -- it enumerates the model's named
entities -- so the reader does not run BNG2.pl or generate a reaction
network. The single exception is :meth:`BnglModel.is_valid`, which shells
out to ``BNG2.pl --check`` (a parse/semantic check, *without* network
generation) when a BNG2.pl is locatable, and degrades gracefully to
``True`` when no BNG backend is available -- mirroring how the SBML loader
always validates because ``libsbml`` is always present.

The BNGL entity sets that back the introspection methods were established
against BioNetGen's ``Perl2/`` source (the reference implementation), not
inferred from the PySB analogy:

* expression symbols are exactly the BNG ``ParamList`` -- parameters,
  observables, and global functions; compartments are *not* expression
  symbols (they never enter the ``ParamList``);
* the full model-entity namespace additionally includes molecule types,
  compartments, and seed species.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..._utils import _generate_path
from .. import is_valid_identifier
from . import MODEL_TYPE_BNGL
from .model import Model

__all__ = ["BnglModel", "BnglEntities", "parse_bngl"]

#: The three keywords that open an observable declaration line.
_OBS_KEYWORDS = frozenset({"Molecules", "Species", "Counter"})


@dataclass(frozen=True)
class BnglEntities:
    """The named entities of a BNGL model that the PEtab layer reads.

    :ivar text: The verbatim BNGL model text.
    :ivar parameters: Maps a parameter name to its raw right-hand side -- a
        number (``"5"``, ``"6.02e23"``) or an expression (``"2*base"``),
        kept verbatim; numeric coercion is the caller's job.
    :ivar observable_names: Bare observable names.
    :ivar function_names: Bare global-function names (without ``()``).
    :ivar molecule_type_names: Bare molecule-type names.
    :ivar seed_species: Concrete seed-species pattern strings (verbatim).
    :ivar compartment_names: Bare compartment names.
    """

    text: str
    parameters: dict[str, str]
    observable_names: frozenset[str]
    function_names: frozenset[str]
    molecule_type_names: frozenset[str]
    seed_species: frozenset[str]
    compartment_names: frozenset[str]


def parse_bngl(text: str) -> BnglEntities:
    """Parse BNGL ``text`` into a :class:`BnglEntities`.

    A stdlib ``begin``/``end <block>`` scanner -- no BNG2.pl, no network
    generation. Sufficient for PEtab validation, which only introspects the
    model's declared entities.

    :param text: The BNGL model text.
    :returns: The model's named entities.
    """
    parameters: dict[str, str] = {}
    for line in _block_lines(text, "parameters"):
        name_value = _parameter_name_value(line)
        if name_value is not None:
            parameters[name_value[0]] = name_value[1]
    return BnglEntities(
        text=text,
        parameters=parameters,
        observable_names=_names(text, "observables", _observable_name),
        function_names=_names(text, "functions", _function_name),
        molecule_type_names=_names(
            text, "molecule types", _molecule_type_name
        ),
        seed_species=_names(text, "seed species", _seed_species_pattern),
        compartment_names=_names(text, "compartments", _compartment_name),
    )


def _names(text: str, block_name: str, extractor) -> frozenset[str]:
    """The non-empty names ``extractor`` yields over a block's lines."""
    return frozenset(
        name
        for name in (
            extractor(line) for line in _block_lines(text, block_name)
        )
        if name
    )


def _block_lines(text: str, block_name: str) -> list[str]:
    """The comment-stripped, non-blank lines inside ``begin``/``end``."""
    begin = re.compile(rf"^begin\s+{block_name}\b", re.IGNORECASE)
    end = re.compile(rf"^end\s+{block_name}\b", re.IGNORECASE)
    lines = []
    in_block = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if begin.match(line):
            in_block = True
        elif end.match(line):
            in_block = False
        elif in_block and line:
            lines.append(line)
    return lines


def _parameter_name_value(line: str) -> tuple[str, str] | None:
    """``(name, rhs)`` for a ``Name (WS | "=") MathExpression`` line."""
    match = re.match(r"^(\w+)\s*=\s*(.+)$", line) or re.match(
        r"^(\w+)\s+(.+)$", line
    )
    return (match.group(1), match.group(2).strip()) if match else None


def _observable_name(line: str) -> str | None:
    """The name in a ``<keyword> <name> <pattern>`` observable line."""
    tokens = line.split()
    if len(tokens) >= 2 and tokens[0] in _OBS_KEYWORDS:
        return tokens[1]
    return None


def _function_name(line: str) -> str | None:
    """The name in a ``<name>() = ...`` global-function line."""
    match = re.match(r"(\w+)\s*\(", line) or re.match(r"(\w+)\s*=", line)
    return match.group(1) if match else None


def _molecule_type_name(line: str) -> str | None:
    """The name in a ``<name>(...)`` molecule-type line."""
    match = re.match(r"(\w+)", line)
    return match.group(1) if match else None


def _seed_species_pattern(line: str) -> str | None:
    """The species pattern in a ``<pattern> <value>`` seed-species line."""
    tokens = line.split()
    return tokens[0] if tokens else None


def _compartment_name(line: str) -> str | None:
    """The name in a ``<name> <dims> <size> [outside]`` line."""
    tokens = line.split()
    return tokens[0] if tokens else None


class BnglModel(Model):
    """PEtab wrapper for BNGL models (introspection only; no simulation)."""

    type_id = MODEL_TYPE_BNGL

    def __init__(
        self,
        model: BnglEntities,
        model_id: str = None,
        rel_path: Path | str | None = None,
        base_path: str | Path | None = None,
    ):
        super().__init__()

        self.rel_path = rel_path
        self.base_path = base_path

        self.model = model
        self._model_id = model_id

        if not is_valid_identifier(self._model_id):
            raise ValueError(
                f"Model ID '{self._model_id}' is not a valid identifier. "
                "Either provide a valid identifier or rename the model file "
                "to a valid PEtab model identifier."
            )

    @staticmethod
    def from_file(
        filepath_or_buffer, model_id: str = None, base_path: str | Path = None
    ) -> BnglModel:
        path = Path(_generate_path(filepath_or_buffer, base_path))
        text = path.read_text(encoding="utf-8", errors="replace")
        return BnglModel(
            model=parse_bngl(text),
            model_id=model_id or path.stem,
            rel_path=filepath_or_buffer,
            base_path=base_path,
        )

    def to_file(self, filename: str | Path | None = None) -> None:
        target = filename or _generate_path(self.rel_path, self.base_path)
        with open(target, "w") as f:
            f.write(self.model.text)

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    # -- parameters ---------------------------------------------------------

    def get_parameter_ids(self) -> Iterable[str]:
        return list(self.model.parameters)

    def get_parameter_value(self, id_: str) -> float:
        try:
            rhs = self.model.parameters[id_]
        except KeyError as e:
            raise ValueError(f"Parameter {id_} does not exist.") from e
        try:
            return float(rhs)
        except ValueError as e:
            raise NotImplementedError(
                f"Parameter '{id_}' has an expression value '{rhs}'. "
                "Evaluating a BNGL parameter expression requires BNG2.pl / "
                "network generation, which is out of scope for the "
                "introspection-only BnglModel."
            ) from e

    def get_free_parameter_ids_with_values(
        self,
    ) -> Iterable[tuple[str, float]]:
        out = []
        for name, rhs in self.model.parameters.items():
            try:
                out.append((name, float(rhs)))
            except ValueError:
                # An expression-valued parameter has no introspection-grade
                # value; skip it rather than evaluate the expression.
                continue
        return out

    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        # All parameters are allowed in the parameter table.
        return list(self.model.parameters)

    # -- model-entity namespaces (verified against BioNetGen) ---------------

    def has_entity_with_id(self, entity_id) -> bool:
        # The full declared-identifier namespace.
        return (
            entity_id in self.model.parameters
            or entity_id in self.model.observable_names
            or entity_id in self.model.function_names
            or entity_id in self.model.molecule_type_names
            or entity_id in self.model.compartment_names
            or entity_id in self.model.seed_species
        )

    def get_valid_ids_for_condition_table(self) -> Iterable[str]:
        return list(self.model.parameters) + list(self.model.compartment_names)

    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        # The BNG ParamList: parameters, observables, global functions only.
        return (
            id_ in self.model.parameters
            or id_ in self.model.observable_names
            or id_ in self.model.function_names
        )

    def is_state_variable(self, id_: str) -> bool:
        # At introspection grade only the concrete seed species are known;
        # the full species set is a network-generation product.
        return id_ in self.model.seed_species

    # -- validity -----------------------------------------------------------

    def is_valid(self) -> bool:
        # Real BNG2.pl --check (parse/semantic validation, no network
        # generation) when locatable, else True -- never a false failure
        # where no BNG backend is available.
        bng2 = _locate_bng2()
        if bng2 is None or self.rel_path is None:
            return True
        path = Path(_generate_path(self.rel_path, self.base_path))
        if not path.is_file():
            # No local model file to check (e.g. a buffer-loaded model).
            return True
        # Resolve to an absolute path so BNG2.pl finds the model regardless
        # of the working directory it runs in (its output stays next to the
        # model via ``cwd``).
        path = path.resolve()
        try:
            result = subprocess.run(  # noqa: S603
                [bng2, "--check", str(path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(path.parent),
            )
        except (OSError, subprocess.SubprocessError):
            # A tooling hiccup must not masquerade as an invalid model.
            return True
        return result.returncode == 0


def _locate_bng2() -> str | None:
    """A path to ``BNG2.pl`` via ``BNGPATH`` or ``PATH``, else ``None``."""
    bngpath = os.environ.get("BNGPATH")
    if bngpath:
        candidate = Path(bngpath) / "BNG2.pl"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("BNG2.pl")

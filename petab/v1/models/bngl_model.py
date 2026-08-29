"""BNGL (BioNetGen Language) model support for PEtab.

Adds a ``bngl`` model type, so a PEtab problem declaring ``language: bngl``
can be loaded and validated.

:class:`BnglModel` is backed by :func:`parse_bngl`, a small, dependency-free
BNGL reader. It only reads a model's declared entities -- parameters,
observables, functions, molecule types, compartments, seed species -- which
is all PEtab validation needs; it never runs BNG2.pl or generates a reaction
network. The one exception is :meth:`BnglModel.is_valid`: if a ``BNG2.pl``
is found (``BNGPATH`` or ``PATH``), it runs ``BNG2.pl --check`` (a
parse/semantic check, no network generation); otherwise the model is
assumed valid.

Three things worth knowing if a model doesn't parse the way you expect:

* Symbols usable in an observable formula are parameters, observables, and
  functions -- *not* compartments.
* A parameter whose value is an expression over other parameters
  (``kon  koff/(Kd*NA*V)``) is evaluated, since a parameters block is
  arithmetic over other parameters and needs no reaction network. The
  arithmetic follows BNGL rather than Python, so ``^`` is a power, it
  groups from the left, and unary minus binds tighter than it does in
  Python. See :func:`evaluate_bngl_parameters`.
* The reader accepts line continuations (a trailing ``\\``), ``begin
  species`` as an alias for ``begin seed species``, line labels (both the
  numeric ``1 L0 1`` and named ``CD14: ...`` forms), and a leading ``$``
  (fixed-concentration) marker on a seed species.

Examples
--------

For example BNGL PEtab v2 problems, see the `PyBioNetFit
<https://github.com/lanl/PyBNF>`_ `tutorials
<https://github.com/lanl/PyBNF/tree/main/examples/tutorial/>`_.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..._utils import _generate_path
from .. import is_valid_identifier
from . import MODEL_TYPE_BNGL
from .model import Model

__all__ = ["BnglEntities", "BnglModel", "parse_bngl"]

#: The three keywords that open an observable declaration line.
_OBS_KEYWORDS = frozenset({"Molecules", "Species", "Counter"})

#: Short spellings BNG2.pl accepts for a block's canonical (long) name; either
#: spelling opens and closes the same block. The grammar doc
#: (``BNG_vscode_extension/docs/bngl-grammar.md``) also lists ``molecules`` and
#: ``rules``, but BNG2.pl 2.9.3 -- the reference this reader targets -- rejects
#: both ("Could not process block type"), so honoring them would accept models
#: BNG2.pl refuses. Only ``species`` (for ``seed species``, also BNG2.pl's own
#: canonical output spelling) is real.
_BLOCK_ALIASES = {
    "seed species": ("species",),
}


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


def _logical_lines(text: str) -> list[str]:
    """The comment-stripped *logical* lines of ``text`` -- physical lines with
    BNGL line continuations joined.

    Mirrors BNG2.pl's ``readFile`` (``Perl2/BNGModel.pm``): strip the ``#``
    comment first, then while the line ends with ``\\`` (as its last
    non-whitespace character) drop that ``\\`` and append the next
    comment-stripped physical line *directly* -- no separating space, so a
    token split across the break (``1e\\`` + ``3`` -> ``1e3``) rejoins.
    Without this, a continued parameter / function / observable is truncated at
    the ``\\`` (e.g. a ``k = \\`` line would read as the value ``"\\"``).
    """
    raw_lines = text.splitlines()
    out = []
    i, n = 0, len(raw_lines)
    while i < n:
        line = raw_lines[i].split("#", 1)[0]
        i += 1
        while re.search(r"\\\s*$", line):
            line = re.sub(r"\\\s*$", "", line)
            if i >= n:
                break  # a dangling continuation at EOF
            line += raw_lines[i].split("#", 1)[0]
            i += 1
        out.append(line.strip())
    return out


def _block_lines(text: str, block_name: str) -> list[str]:
    """The comment-stripped, non-blank lines inside ``begin``/``end``.

    ``block_name`` is the canonical (long) spelling; a BNG2.pl-accepted alias
    for it (only ``species`` for ``seed species``; see :data:`_BLOCK_ALIASES`)
    opens and closes the same block. Lines are logical lines -- continuations
    joined (see :func:`_logical_lines`).
    """
    names = "|".join(
        re.escape(name)
        for name in (block_name, *_BLOCK_ALIASES.get(block_name, ()))
    )
    begin = re.compile(rf"^begin\s+(?:{names})\b", re.IGNORECASE)
    end = re.compile(rf"^end\s+(?:{names})\b", re.IGNORECASE)
    lines = []
    in_block = False
    for line in _logical_lines(text):
        if begin.match(line):
            in_block = True
        elif end.match(line):
            in_block = False
        elif in_block and line:
            lines.append(line)
    return lines


def _strip_line_label(line: str) -> str:
    """Drop a leading BNGL line label so the entity, not the label, is read.

    ``LineLabel = {Digit}, WS | Name, ":", [WS]`` (grammar) -- either a numeric
    index (the legacy ``.net``-style ``1 L0 1`` form) or a named label
    (``CD14: CD14(...)``). A valid BNGL identifier starts with a letter, so a
    leading digit-run is always an index; a compartment prefix is ``@Name:``
    (with the ``@``), so a bare ``Name:`` at line start is unambiguously a
    label.
    """
    match = re.match(r"^\d+\s+(.*)$", line) or re.match(
        r"^[A-Za-z]\w*:\s+(.*)$", line
    )
    return match.group(1) if match else line


def _parameter_name_value(line: str) -> tuple[str, str] | None:
    """``(name, rhs)`` for a ``[LineLabel] Name (WS|"=") MathExpr`` line."""
    line = _strip_line_label(line)
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
    """The species pattern in a ``[LineLabel] ["$"] <pattern> <value>`` line.

    A leading line label (numeric index ``1 A() 100`` or named
    ``CD14: CD14(...)``; see :func:`_strip_line_label`) is dropped first so the
    label is not mistaken for the species. A leading ``$`` (the fixed/clamped-
    concentration marker, ``SeedSpeciesDefn = ["$"], Species, WS,
    MathExpression``) is a modifier, not part of the species identity, so it
    too is stripped: ``$counter() 10``
    enumerates the state variable ``counter()``.
    """
    line = _strip_line_label(line)
    if line.startswith("$"):
        line = line[1:].lstrip()
    tokens = line.split()
    return tokens[0] if tokens else None


def _compartment_name(line: str) -> str | None:
    """The name in a ``<name> <dims> <size> [outside]`` line."""
    tokens = line.split()
    return tokens[0] if tokens else None


# -- parameter expression evaluation -----------------------------------------
#
# A ``parameters`` block may give a parameter an expression over other
# parameters (``kon  koff/(Kd*NA*V)``) rather than a literal, which is
# ordinary BNGL style rather than an edge case: across 303 models drawn from
# the BioNetGen model collections, 1934 of 9323 parameter declarations
# (20.8%) are expression-valued. Resolving them needs no BNG2.pl and no
# network generation, because a parameters block is arithmetic over other
# parameters.
#
# The sublanguage is BNGL's, not Python's, and the two disagree in ways that
# are silent rather than loud. Every rule below was checked against BNG2.pl
# 2.9.3 by running the expression through
# ``writeNET({evaluate_expressions=>1})``, the only export path that emits
# numbers instead of echoing the source text. The function table and the
# precedence order are BioNetGen's ``Perl2/Expression.pm`` (``%functions``,
# ``%NARGS``, and the operator list in ``arrayToExpression``):
#
# * ``^`` raises to a power, where Python's is bitwise exclusive-or.
# * ``^`` is *left* associative, so ``2^3^2`` is 64 rather than 512.
# * Unary minus binds *tighter* than ``^``, so ``-2^2`` is ``(-2)^2`` == 4
#   rather than ``-(2^2)`` == -4. This holds for literals, parameters,
#   parenthesised groups and function calls alike (``-exp(0)^2`` == 1).
# * The natural logarithm is ``ln``. A bare ``log`` is rejected, as BNG2.pl
#   rejects it, so a typo stays an error instead of becoming a plausible
#   wrong number.
# * ``rint`` is ``floor(x + 0.5)``, rounding a half upward, where Python's
#   ``round`` sends a half to the nearest even number.
# * ``_pi`` and ``_e`` are zero-argument functions, written ``_pi()``.
# * Comparison and logical operators yield 1.0/0.0, and ``if(cond, a, b)``
#   selects on ``cond != 0``. BNG2.pl evaluates all three arguments before
#   selecting, so ``if(1, 5, 1/0)`` is an error there and here.
#
# Expressions are tokenized and parsed rather than handed to ``eval``, which
# would import Python's precedence and operator meanings along with the
# obvious injection problem.


class BnglExpressionError(ValueError):
    """A parameter expression could not be parsed or evaluated."""


class CircularParameterError(BnglExpressionError):
    """A parameter's definition depends on itself, directly or not."""


def _bngl_if(condition: float, then_: float, else_: float) -> float:
    """BNGL's ``if``, which selects on ``condition != 0``."""
    return then_ if condition != 0 else else_


#: The built-in functions BNG2.pl accepts, mirroring ``%functions`` in
#: ``Expression.pm``. ``log`` is absent because BNGL has no bare ``log``.
#: ``floor`` and ``ceil`` are absent because ``Expression.pm`` keeps them
#: commented out as unsupported, so BNG2.pl rejects them. ``TFUN`` is absent
#: deliberately: it reads a data file while a simulation runs, so it is not a
#: parameters-block constant.
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "_pi": lambda: math.pi,
    "_e": lambda: math.e,
    "exp": math.exp,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "abs": abs,
    "rint": lambda x: float(math.floor(x + 0.5)),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "if": _bngl_if,
    "min": min,
    "max": max,
    "sum": lambda *a: math.fsum(a),
    "avg": lambda *a: math.fsum(a) / len(a),
}

#: Names BNG2.pl refuses to accept as a parameter name ("Cannot use built-in
#: function name '_pi' as a parameter").
RESERVED_PARAMETER_NAMES = frozenset(_FUNCTIONS)

# Longest-first, so ``**``, ``>=`` and ``&&`` are not split into single
# characters. ``~=`` is BNG2.pl's alias for ``!=``.
_TOKEN_RE = re.compile(
    r"""
    (?P<number>\d+\.\d*(?:[eE][+-]?\d+)?
              |\.\d+(?:[eE][+-]?\d+)?
              |\d+(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\*\*|&&|\|\||<=|>=|==|!=|~=|[-+*/^(),<>])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)

_COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "~=": lambda a, b: a != b,
}


def _tokenize(text: str) -> list[tuple[str, str]]:
    """``(kind, value)`` tokens for a BNGL expression."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise BnglExpressionError(
                f"Unexpected character {text[pos]!r} at position {pos} "
                f"in {text!r}"
            )
        pos = match.end()
        kind = match.lastgroup
        if kind == "space":
            continue
        value = match.group()
        # BNGL writes exponentiation as ^, and BNG2.pl also accepts **.
        tokens.append(("op", "^") if value == "**" else (kind, value))
    return tokens


class _Parser:
    """Recursive-descent parser for the arithmetic sublanguage.

    Precedence, loosest to tightest, is the order of the operator list in
    ``arrayToExpression``, which folds each level left to right::

        && ||  <  < > <= >= == != ~=  <  + -  <  * /  <  unary - +  <  ^

    Unary minus sitting below ``^`` is what makes ``-2^2`` come out as 4,
    and the left fold is what makes ``2^3^2`` come out as 64.
    """

    def __init__(
        self,
        tokens: list[tuple[str, str]],
        text: str,
        lookup: Callable[[str], float],
    ):
        self._tokens = tokens
        self._text = text
        self._lookup = lookup
        self._pos = 0

    def parse(self) -> float:
        """The value of the whole expression."""
        value = self._parse_logical()
        if self._pos != len(self._tokens):
            raise BnglExpressionError(
                f"Unexpected trailing input in {self._text!r} at token "
                f"{self._tokens[self._pos][1]!r}"
            )
        return value

    def _peek(self) -> tuple[str, str] | None:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _accept(self, value: str) -> bool:
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] == value:
            self._pos += 1
            return True
        return False

    def _accept_any(self, values: Iterable[str]) -> str | None:
        token = self._peek()
        if token is not None and token[0] == "op" and token[1] in values:
            self._pos += 1
            return token[1]
        return None

    def _expect(self, value: str) -> None:
        if not self._accept(value):
            found = self._peek()
            seen = repr(found[1]) if found else "end of expression"
            raise BnglExpressionError(
                f"Expected {value!r} in {self._text!r}, found {seen}"
            )

    def _parse_logical(self) -> float:
        value = self._parse_comparison()
        while True:
            op = self._accept_any(("&&", "||"))
            if op is None:
                return value
            rhs = self._parse_comparison()
            # BNG2.pl normalises these to 1/0 rather than returning an
            # operand the way bare Perl would, so ``0||5`` is 1.0.
            if op == "&&":
                value = float(value != 0 and rhs != 0)
            else:
                value = float(value != 0 or rhs != 0)

    def _parse_comparison(self) -> float:
        value = self._parse_sum()
        while True:
            op = self._accept_any(_COMPARISONS)
            if op is None:
                return value
            value = float(_COMPARISONS[op](value, self._parse_sum()))

    def _parse_sum(self) -> float:
        value = self._parse_product()
        while True:
            if self._accept("+"):
                value += self._parse_product()
            elif self._accept("-"):
                value -= self._parse_product()
            else:
                return value

    def _parse_product(self) -> float:
        value = self._parse_power()
        while True:
            if self._accept("*"):
                value *= self._parse_power()
            elif self._accept("/"):
                divisor = self._parse_power()
                if divisor == 0:
                    raise BnglExpressionError(
                        f"Division by zero in {self._text!r}"
                    )
                # True division throughout: BNGL has no integer division.
                value = float(value) / float(divisor)
            else:
                return value

    def _parse_power(self) -> float:
        # Left associative, and a signed operand belongs to the base rather
        # than to the whole power: BNG2.pl gives -2^2 == 4, 2^3^2 == 64.
        value = self._parse_unary()
        while self._accept("^"):
            exponent = self._parse_unary()
            try:
                value = float(value**exponent)
            except (ArithmeticError, TypeError, ValueError) as e:
                # 0^-1, an overflow, or a negative base raised to a
                # fractional power, which Python answers with a complex.
                raise BnglExpressionError(
                    f"Cannot raise {value!r} to the power {exponent!r} "
                    f"in {self._text!r}"
                ) from e
        return value

    def _parse_unary(self) -> float:
        if self._accept("-"):
            return -self._parse_unary()
        if self._accept("+"):
            return self._parse_unary()
        return self._parse_atom()

    def _parse_atom(self) -> float:
        token = self._peek()
        if token is None:
            raise BnglExpressionError(
                f"Expression ended unexpectedly: {self._text!r}"
            )
        kind, value = token

        if kind == "number":
            self._pos += 1
            return float(value)

        if kind == "op" and value == "(":
            self._pos += 1
            inner = self._parse_logical()
            self._expect(")")
            return inner

        if kind == "name":
            self._pos += 1
            if self._accept("("):
                # ``_pi()`` and ``_e()`` take no arguments.
                args = []
                if self._peek() != ("op", ")"):
                    args.append(self._parse_logical())
                    while self._accept(","):
                        args.append(self._parse_logical())
                self._expect(")")
                return self._call(value, args)
            return self._lookup(value)

        raise BnglExpressionError(
            f"Unexpected token {value!r} in {self._text!r}"
        )

    def _call(self, name: str, args: list[float]) -> float:
        try:
            func = _FUNCTIONS[name]
        except KeyError:
            raise BnglExpressionError(
                f"Unknown function {name!r} in {self._text!r}"
            ) from None
        try:
            return float(func(*args))
        except TypeError as e:
            raise BnglExpressionError(
                f"Wrong number of arguments to {name!r} in {self._text!r}"
            ) from e
        except ArithmeticError as e:
            raise BnglExpressionError(
                f"{name}() could not be evaluated in {self._text!r}: {e}"
            ) from e
        except ValueError as e:
            raise BnglExpressionError(
                f"{name}() is undefined for its argument in "
                f"{self._text!r}: {e}"
            ) from e


def evaluate_bngl_expression(text: str, symbols: dict[str, float]) -> float:
    """Evaluate one BNGL expression against already-resolved ``symbols``.

    :param text: The expression, for example ``koff/(Kd*NA*V)``.
    :param symbols: Values for the names the expression refers to.
    :returns: The value of the expression.
    :raises BnglExpressionError: If it cannot be parsed or evaluated, or
        refers to a name ``symbols`` does not define.
    """

    def lookup(name: str) -> float:
        try:
            return symbols[name]
        except KeyError:
            raise BnglExpressionError(
                f"Unknown parameter {name!r} in {text!r}"
            ) from None

    return _Parser(_tokenize(text), text, lookup).parse()


def _parameter_resolver(
    parameters: dict[str, str],
) -> tuple[Callable[[str], float], dict[str, float]]:
    """A memoizing ``lookup(name)`` over a parameters block, and its cache."""
    resolved: dict[str, float] = {}
    resolving: list[str] = []

    def lookup(name: str) -> float:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            start = resolving.index(name)
            cycle = " -> ".join([*resolving[start:], name])
            raise CircularParameterError(
                f"Parameter {name!r} is defined in terms of itself: {cycle}"
            )
        if name not in parameters:
            raise BnglExpressionError(f"Unknown parameter {name!r}")
        if name in RESERVED_PARAMETER_NAMES:
            raise BnglExpressionError(
                f"{name!r} is a BNGL built-in function name and cannot be "
                f"used as a parameter name"
            )
        resolving.append(name)
        try:
            value = _Parser(
                _tokenize(parameters[name]), parameters[name], lookup
            ).parse()
        finally:
            resolving.pop()
        resolved[name] = value
        return value

    return lookup, resolved


def evaluate_bngl_parameters(
    parameters: dict[str, str],
) -> dict[str, float]:
    """Resolve a BNGL parameters block to numbers.

    Values are resolved lazily in dependency order, so a parameter may be
    defined before the ones it depends on. BNG2.pl is stricter here, since
    it drops a forward-referencing parameter, but accepting the
    order-independent form loses no model BNG2.pl would have accepted.

    :param parameters: Parameter name to raw right-hand side, literal or
        expression, as :func:`parse_bngl` collects it.
    :returns: Parameter name to value.
    :raises CircularParameterError: On a definition that depends on itself.
    :raises BnglExpressionError: On anything unparseable, or a reference to
        a name the block does not define. Use
        :func:`evaluate_bngl_parameters_partial` when one bad definition
        should not cost the caller the whole block.
    """
    lookup, resolved = _parameter_resolver(parameters)
    for name in parameters:
        lookup(name)
    return resolved


def evaluate_bngl_parameters_partial(
    parameters: dict[str, str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve what can be resolved in a parameters block, and report the rest.

    A block is a single namespace, so one unusable definition should cost
    the caller that parameter and whatever depends on it, rather than the
    entire block.

    :param parameters: Parameter name to raw right-hand side.
    :returns: ``(values, errors)``, where ``values`` maps each parameter
        that could be computed to its value and ``errors`` maps each one
        that could not to the reason. Every parameter appears in exactly
        one of the two.
    """
    lookup, resolved = _parameter_resolver(parameters)
    errors: dict[str, str] = {}
    for name in parameters:
        if name in resolved:
            continue
        try:
            lookup(name)
        except BnglExpressionError as e:
            errors[name] = str(e)
    return resolved, errors


class BnglModel(Model):
    """PEtab wrapper for BNGL models."""

    type_id = MODEL_TYPE_BNGL

    def __init__(
        self,
        model: BnglEntities,
        model_id: str | None = None,
        rel_path: Path | str | None = None,
        base_path: str | Path | None = None,
    ):
        super().__init__()

        self.rel_path = rel_path
        self.base_path = base_path

        self.model = model
        self._model_id = model_id
        self._resolved_parameters: (
            tuple[dict[str, float], dict[str, str]] | None
        ) = None

        if not is_valid_identifier(self._model_id):
            raise ValueError(
                f"Model ID '{self._model_id}' is not a valid identifier. "
                "Either provide a valid identifier or rename the model file "
                "to a valid PEtab model identifier."
            )

    @staticmethod
    def from_file(
        filepath_or_buffer,
        model_id: str | None = None,
        base_path: str | Path | None = None,
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
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.model.text)

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    def get_parameter_ids(self) -> Iterable[str]:
        return list(self.model.parameters)

    def _parameter_values(self) -> tuple[dict[str, float], dict[str, str]]:
        """``(values, errors)`` for the parameters block, computed once.

        A parameters block is arithmetic over other parameters, so this
        needs no BNG2.pl and no network generation. Resolution is partial:
        one unusable definition costs that parameter and whatever depends
        on it, rather than the whole block.
        """
        if self._resolved_parameters is None:
            self._resolved_parameters = evaluate_bngl_parameters_partial(
                dict(self.model.parameters)
            )
        return self._resolved_parameters

    def get_parameter_value(self, id_: str) -> float:
        if id_ not in self.model.parameters:
            raise ValueError(f"Parameter {id_} does not exist.")
        values, errors = self._parameter_values()
        if id_ in values:
            return values[id_]
        raise ValueError(
            f"Parameter '{id_}' has an expression value "
            f"'{self.model.parameters[id_]}' that could not be evaluated: "
            f"{errors[id_]}"
        )

    def get_free_parameter_ids_with_values(
        self,
    ) -> Iterable[tuple[str, float]]:
        # An expression-valued parameter used to be skipped here, which
        # lost it from the PEtab problem with nothing said. They are
        # resolved now, and anything still unusable is named in a warning
        # rather than disappearing, without taking the block with it.
        values, errors = self._parameter_values()
        if errors:
            detail = "; ".join(
                f"{name} ({errors[name]})" for name in sorted(errors)
            )
            warnings.warn(
                f"Model {self._model_id!r}: {len(errors)} of "
                f"{len(self.model.parameters)} parameters could not be "
                f"evaluated and are omitted: {detail}",
                stacklevel=2,
            )
        return [
            (name, values[name])
            for name in self.model.parameters
            if name in values
        ]

    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        # All parameters are allowed in the parameter table.
        return list(self.model.parameters)

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
                check=False,
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

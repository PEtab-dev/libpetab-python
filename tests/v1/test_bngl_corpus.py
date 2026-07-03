"""Corpus regression gate: the BNGL reader agrees with BNG2.pl on real models.

Asserts :func:`petab.v1.models.bngl_model.parse_bngl` enumerates the same model
entities BNG2.pl does, over a curated set of public community BNGL models under
``bngl_corpus/``. BNG2.pl's answers are cached in ``bngl_corpus/golden.json``:
the entity name sets BNG2.pl emits from ``writeModel`` (its canonical parse, no
network generation). So the test itself needs **no BNG2.pl** -- it compares the
reader's output against the frozen oracle. Seed species are compared by
molecule composition, which absorbs BNG2.pl's pattern canonicalization (``t``
vs ``t()``, component reordering, ``@compartment`` prefix vs suffix) while
still catching a genuinely missing or invented species.

Regenerate the golden after adding a fixture or changing the reader (needs a
BNG2.pl on ``BNGPATH``/``PATH``) and review the diff::

    python tests/v1/test_bngl_corpus.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from petab.v1.models.bngl_model import parse_bngl

_CORPUS = Path(__file__).parent / "bngl_corpus"
_GOLDEN = _CORPUS / "golden.json"
_MODELS = sorted(_CORPUS.glob("*.bngl"))


def _molecule_name(part):
    """The molecule name in one ``.``-separated piece of a species pattern,
    ignoring an ``@compartment:`` prefix, a ``$`` clamp, and any components.

    BNG2.pl writes the compartment prefix before the clamp (``@C::$ADP()``), so
    the prefix is stripped first, then the ``$``.
    """
    part = part.strip()
    prefix = re.match(r"@\w+::?", part)
    if prefix:
        part = part[prefix.end() :]
    part = part.lstrip("$")
    name = re.match(r"(\w+)", part)
    return name.group(1) if name else None


def seed_composition(patterns):
    """Reordering-robust seed-species signature: each species as its sorted
    molecule-name list, the whole sorted. JSON-safe (lists, not tuples)."""
    sig = []
    for pattern in patterns:
        mols = sorted(
            n for n in (_molecule_name(p) for p in pattern.split(".")) if n
        )
        sig.append(mols)
    return sorted(sig)


def _reader_entities(entities):
    """The entity name sets the golden records / the reader is checked against.
    BNG2.pl-generated ``_``-names (a valid BNGL name starts with a letter) are
    excluded from both sides."""

    def named(names):
        return sorted(n for n in names if not n.startswith("_"))

    return {
        "parameters": named(entities.parameters),
        "observables": sorted(entities.observable_names),
        "functions": named(entities.function_names),
        "molecule_types": sorted(entities.molecule_type_names),
        "compartments": sorted(entities.compartment_names),
        "seed_composition": seed_composition(entities.seed_species),
    }


@pytest.mark.parametrize("model", _MODELS, ids=lambda p: p.stem)
def test_reader_matches_bng2_golden(model):
    golden = json.loads(_GOLDEN.read_text())
    expected = golden[model.name]
    actual = _reader_entities(parse_bngl(model.read_text()))
    assert actual == expected, (
        f"{model.name}: reader disagrees with the BNG2.pl golden -- "
        f"regenerate with `python {__file__}` if the change is intended"
    )


# --------------------------------------------------------------------------
# Golden regeneration -- dev-only, needs BNG2.pl. Runs `writeModel` on each
# model's definition blocks (actions stripped, so no network generation),
# re-parses the canonical BNGL BNG2.pl emits, and records its entity name sets.
# --------------------------------------------------------------------------

_MODEL_BLOCKS = frozenset(
    {
        "parameters",
        "molecule types",
        "molecules",
        "seed species",
        "species",
        "observables",
        "functions",
        "compartments",
        "reaction rules",
        "rules",
        "energy patterns",
        "population types",
        "population maps",
    }
)


def _strip_to_model(text):
    """The model-definition blocks plus a single ``writeModel`` action; drops
    actions (even nested in ``begin model``) and bare top-level directives."""
    out, stack = [], []
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        begin = re.match(r"begin\s+(.+)", s, re.I)
        end = re.match(r"end\s+(.+)", s, re.I)
        if begin:
            stack.append(begin.group(1).strip().lower())
            if stack[-1] != "actions":
                out.append(line)
        elif end:
            top = stack.pop() if stack else None
            if top != "actions":
                out.append(line)
        elif "actions" not in stack and [b for b in stack if b != "model"]:
            out.append(line)
    return "\n".join(out) + '\nwriteModel({prefix=>"canon"})\n'


def _canonical_bngl(model_text, bng2):
    work = tempfile.mkdtemp(prefix="bnggold_")
    try:
        (Path(work) / "in.bngl").write_text(_strip_to_model(model_text))
        result = subprocess.run(  # noqa: S603
            [bng2, "in.bngl"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=120,
        )
        canon = Path(work) / "canon.bngl"
        if result.returncode != 0 or not canon.is_file():
            raise RuntimeError(
                f"BNG2.pl rejected the model:\n{result.stdout}{result.stderr}"
            )
        return canon.read_text(encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _regenerate():
    from petab.v1.models.bngl_model import _locate_bng2

    bng2 = _locate_bng2()
    if bng2 is None:
        sys.exit("No BNG2.pl found (set BNGPATH or put it on PATH).")
    golden = {}
    for model in _MODELS:
        canon = _canonical_bngl(model.read_text(), bng2)
        golden[model.name] = _reader_entities(parse_bngl(canon))
    _GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"wrote {_GOLDEN} ({len(golden)} models)")


if __name__ == "__main__":
    _regenerate()

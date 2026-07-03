"""Tests for petab.v1.models.bngl_model"""

from pathlib import Path

import pytest

from petab.v1.models.bngl_model import BnglEntities, BnglModel, parse_bngl

#: A self-contained BNGL model exercising every entity kind read by the
#: PEtab layer: parameters, a molecule type, a seed species, an observable,
#: and a global function (the parabola measurement model).
EXAMPLE_BNGL = Path(__file__).parent / "bngl" / "parabola.bngl"


@pytest.fixture
def model() -> BnglModel:
    return BnglModel.from_file(EXAMPLE_BNGL)


def test_parse_bngl():
    entities = parse_bngl(EXAMPLE_BNGL.read_text())
    assert isinstance(entities, BnglEntities)
    assert entities.parameters == {
        "v1": "5",
        "v2": "5",
        "v3": "5",
        "k_cond": "1",
    }
    assert entities.observable_names == {"x"}
    assert entities.function_names == {"y"}
    assert entities.molecule_type_names == {"counter"}
    assert entities.seed_species == {"counter()"}
    assert entities.compartment_names == frozenset()


def test_parameter_ids_and_values(model):
    assert set(model.get_parameter_ids()) == {"v1", "v2", "v3", "k_cond"}
    assert model.get_parameter_value("v1") == 5.0
    assert dict(model.get_free_parameter_ids_with_values()) == {
        "v1": 5.0,
        "v2": 5.0,
        "v3": 5.0,
        "k_cond": 1.0,
    }


def test_get_parameter_value_unknown_raises(model):
    with pytest.raises(ValueError):
        model.get_parameter_value("nope")


def test_expression_valued_parameter_is_not_evaluated():
    # A numeric RHS coerces to float; an expression RHS is confined to
    # NotImplementedError rather than evaluated (that needs BNG2.pl).
    entities = parse_bngl(
        "begin parameters\n base 2\n k_on 2*base\nend parameters\n"
    )
    model = BnglModel(entities, model_id="m")
    assert model.get_parameter_value("base") == 2.0
    with pytest.raises(NotImplementedError):
        model.get_parameter_value("k_on")
    # The expression-valued parameter is still an enumerated entity, but it
    # contributes no introspection-grade value.
    assert dict(model.get_free_parameter_ids_with_values()) == {"base": 2.0}


# -- grammar hardening: block aliases + seed-species "$" clamp ---------------
# Kept in sync with PyBNF's sibling reader (pybnf/petab/_bngl.py, ADR-0026);
# these cases are the anchor that keeps the two block scanners from drifting.


def test_seed_species_dollar_clamp_is_stripped():
    # SeedSpeciesDefn = ["$"], Species, WS, MathExpression -- the "$" fixes the
    # concentration but is not part of the species identity, so the enumerated
    # state variable is the bare pattern (attached "$A()" and spaced "$ A()").
    entities = parse_bngl(
        "begin seed species\n $A() 100\n $ B() 0\n C() 5\nend seed species\n"
    )
    assert entities.seed_species == frozenset({"A()", "B()", "C()"})
    assert "$A()" not in entities.seed_species  # the marker never leaks


def test_rejected_block_aliases_are_not_honored():
    # The grammar doc lists `molecules`/`rules` as aliases, but BNG2.pl 2.9.3
    # REJECTS both ("Could not process block type"); only `species` is real. To
    # match the reference implementation the reader must NOT treat
    # `begin molecules`/`begin rules` as their canonical blocks.
    entities = parse_bngl("begin molecules\n A()\n B(x)\nend molecules\n")
    assert entities.molecule_type_names == frozenset()


def test_seed_species_block_alias():
    # `begin species` is BNG's short alias for `begin seed species` -- and the
    # "$" clamp is stripped under the alias spelling too.
    entities = parse_bngl("begin species\n $A() 100\n B() 0\nend species\n")
    assert entities.seed_species == frozenset({"A()", "B()"})


def test_line_continuation_is_joined():
    # A trailing "\" continues the logical line (BNG2.pl readFile). Without
    # joining, a continued parameter reads as the value "\"; the join
    # concatenates directly -- no space -- so a token split across the break
    # ("1e\"+"3" -> "1e3") rejoins, matching BNG2.pl.
    entities = parse_bngl(
        "begin parameters\n"
        "  minusb = \\\n(p4-1)/(p4*(1+p2))\n"  # continued expression value
        "  r 1e\\\n3\n"  # token split -> 1e3, no space
        "  a = 1+\\\n2+\\\n3\n"  # chained continuation
        "end parameters\n"
    )
    assert entities.parameters["minusb"] == "(p4-1)/(p4*(1+p2))"
    assert entities.parameters["r"] == "1e3"
    assert entities.parameters["a"] == "1+2+3"


def test_backslash_in_comment_is_not_a_continuation():
    # BNG2.pl strips the comment before testing for a trailing "\", so a "\"
    # inside a comment must not swallow the next line.
    entities = parse_bngl(
        "begin parameters\n k 1 # note \\\n j 2\nend parameters\n"
    )
    assert entities.parameters == {"k": "1", "j": "2"}


def test_indexed_declarations():
    # Legacy .net-style leading index (LineLabel = {Digit}, WS): the index must
    # not be read as the name.
    entities = parse_bngl(
        "begin parameters\n 1 L0 1\n 2 R0 2\nend parameters\n"
        "begin seed species\n 1 A() 100\n 2 B() 50\nend seed species\n"
    )
    assert entities.parameters == {"L0": "1", "R0": "2"}
    assert entities.seed_species == frozenset({"A()", "B()"})


def test_labeled_seed_species():
    # Named line label (LineLabel = Name, ":"): "CD14: CD14(...)" -- the label,
    # which here even equals the molecule name, must not be read as the
    # species. The label is stripped before the "$" clamp.
    entities = parse_bngl(
        "begin seed species\n"
        " CD14: CD14(TLR4,MD2) v1\n"
        " clamp: $MD2(x~0) v2\n"
        "end seed species\n"
    )
    assert entities.seed_species == frozenset({"CD14(TLR4,MD2)", "MD2(x~0)"})


def test_line_label_does_not_over_strip():
    # A normal `name value` param and an `@compartment:` species must be left
    # alone -- a compartment prefix carries "@", so it is not a bare label.
    entities = parse_bngl(
        "begin parameters\n NA = 6.02e23\n k1 1.0\nend parameters\n"
        "begin seed species\n @PM:Rec() 100\nend seed species\n"
    )
    assert entities.parameters == {"NA": "6.02e23", "k1": "1.0"}
    assert entities.seed_species == frozenset({"@PM:Rec()"})


def test_alias_does_not_shadow_the_canonical_block():
    # The `species` alias must not swallow the block whose name it is a
    # substring of: `seed species` and `molecule types` stay distinct.
    entities = parse_bngl(
        "begin molecule types\n Counter()\nend molecule types\n"
        "begin seed species\n $Counter() 1\nend seed species\n"
    )
    assert entities.molecule_type_names == frozenset({"Counter"})
    assert entities.seed_species == frozenset({"Counter()"})


def test_state_variable_ignores_the_clamp():
    # A clamped seed species is still a state variable under its bare id
    # (is_state_variable drives CheckModel's species cross-checks).
    model = BnglModel(
        parse_bngl("begin seed species\n $A() 100\nend seed species\n"),
        model_id="m",
    )
    assert model.is_state_variable("A()")
    assert not model.is_state_variable("$A()")


def test_has_entity_spans_full_declared_namespace(model):
    # parameter, observable, global function, molecule type, seed species.
    for entity in ("v1", "x", "y", "counter", "counter()"):
        assert model.has_entity_with_id(entity)
    # Prefixed PEtab IDs and unknowns are not model entities.
    for non_entity in ("obs_x", "func_y", "nope"):
        assert not model.has_entity_with_id(non_entity)


def test_symbol_allowed_is_the_paramlist_only(model):
    # parameters u observables u global functions (the BNG ParamList).
    for symbol in ("x", "y", "v1"):
        assert model.symbol_allowed_in_observable_formula(symbol)
    # A molecule type / seed species is an entity but not a formula symbol.
    for non_symbol in ("counter", "counter()", "nope"):
        assert not model.symbol_allowed_in_observable_formula(non_symbol)


def test_is_state_variable_is_seed_species_only(model):
    assert model.is_state_variable("counter()")
    assert not model.is_state_variable("v1")
    assert not model.is_state_variable("x")


def test_valid_ids_for_condition_table_is_params_and_compartments():
    entities = parse_bngl(
        "begin parameters\n k 1\nend parameters\n"
        "begin compartments\n EC 3 1.0\nend compartments\n"
    )
    model = BnglModel(entities, model_id="m")
    assert set(model.get_valid_ids_for_condition_table()) == {"k", "EC"}
    assert model.has_entity_with_id("EC")
    # A compartment is an entity but not an observable-formula symbol.
    assert not model.symbol_allowed_in_observable_formula("EC")


def test_invalid_model_id_raises():
    with pytest.raises(ValueError, match="not a valid identifier"):
        BnglModel(parse_bngl(""), model_id="1nope")


def test_repr(model):
    assert repr(model) == "<BnglModel 'parabola'>"


def test_is_valid(model):
    # A valid model validates: a real ``BNG2.pl --check`` where a BNG backend
    # is locatable, or a graceful ``True`` fallback where it is not.
    assert model.is_valid() is True
    # A buffer-loaded model has no file to check and falls back to True.
    assert BnglModel(parse_bngl(""), model_id="m").is_valid() is True


def test_is_valid_detects_broken_model_when_bng_available(tmp_path):
    from petab.v1.models.bngl_model import _locate_bng2

    if _locate_bng2() is None:
        pytest.skip("BNG2.pl not available; is_valid falls back to True")
    # An undefined parameter in the function body -> BNG2.pl --check fails.
    broken = tmp_path / "broken.bngl"
    broken.write_text(
        "begin model\n"
        "  begin parameters\n    k 1\n  end parameters\n"
        "  begin functions\n    f()=k*undefined_symbol\n  end functions\n"
        "end model\n"
    )
    assert BnglModel.from_file(broken).is_valid() is False


def test_model_factory_routes_bngl():
    from petab.v1.models.model import model_factory

    model = model_factory(EXAMPLE_BNGL, "bngl")
    assert isinstance(model, BnglModel)
    assert model.type_id == "bngl"


def test_full_petab_validation_is_clean(tmp_path):
    """A ``language: bngl`` problem loads via ``Problem.from_yaml`` and passes
    every default validation task -- including the model-cross checks that
    read the BnglModel (CheckModel, CheckObservablesDoNotShadowModelEntities,
    CheckAllParametersPresentInParameterTable, CheckValidConditionTargets,
    CheckInitialChangeSymbols).
    """
    petab_v2 = pytest.importorskip("petab.v2")
    from petab.v2 import Problem
    from petab.v2.lint import (
        ValidationIssueSeverity,
        default_validation_tasks,
    )

    # The BNGL model is the only genuinely BNGL-specific artifact; the PEtab
    # tables are built with the v2 API so their format is guaranteed correct.
    (tmp_path / "parabola.bngl").write_text(EXAMPLE_BNGL.read_text())

    problem = Problem()
    for param in ("v1", "v2", "v3"):
        problem.add_parameter(param, estimate=True, lb=0, ub=10)
    # observableFormula is the bare model name (x is an observable, y a global
    # function); both are vouched for by the model, so neither is demanded as
    # an output parameter in the parameter table.
    problem.add_observable(
        "obs_x", "x", noise_formula="0.1", noise_distribution="normal"
    )
    problem.add_observable(
        "func_y", "y", noise_formula="0.1", noise_distribution="normal"
    )
    # A condition targets a model parameter that is *not* estimated -- a valid
    # condition target (parameters u compartments) that exercises
    # CheckValidConditionTargets against the BnglModel.
    problem.add_condition("cond1", k_cond=2)
    problem.add_experiment("exp1", 0, "cond1")
    for t in (0.0, 1.0, 2.0):
        problem.add_measurement(
            "obs_x", time=t, measurement=1.0, experiment_id="exp1"
        )
        problem.add_measurement(
            "func_y", time=t, measurement=2.0, experiment_id="exp1"
        )

    petab_v2.write_parameter_df(
        problem.parameter_df, tmp_path / "parameters.tsv"
    )
    petab_v2.write_observable_df(
        problem.observable_df, tmp_path / "observables.tsv"
    )
    petab_v2.write_condition_df(
        problem.condition_df, tmp_path / "conditions.tsv"
    )
    petab_v2.write_experiment_df(
        problem.experiment_df, tmp_path / "experiments.tsv"
    )
    petab_v2.write_measurement_df(
        problem.measurement_df, tmp_path / "measurements.tsv"
    )

    (tmp_path / "problem.yaml").write_text(
        "format_version: 2.0.0\n"
        "parameter_files: [parameters.tsv]\n"
        "model_files:\n"
        "  parabola:\n"
        "    location: parabola.bngl\n"
        "    language: bngl\n"
        "condition_files: [conditions.tsv]\n"
        "experiment_files: [experiments.tsv]\n"
        "observable_files: [observables.tsv]\n"
        "measurement_files: [measurements.tsv]\n"
    )

    loaded = Problem.from_yaml(str(tmp_path / "problem.yaml"))
    assert isinstance(loaded.model, BnglModel)

    errors = [
        (type(task).__name__, issue.message)
        for task in default_validation_tasks
        if (issue := task.run(loaded)) is not None
        and issue.level == ValidationIssueSeverity.ERROR
    ]
    assert errors == []

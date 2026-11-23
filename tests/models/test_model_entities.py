"""Tests for the model entity classification (`petab._models._entities`)."""

import libsbml
import pytest
import sympy as sp

from petab._models._entities import (
    Dynamics,
    EntityCategory,
    EntityInfo,
    Initializer,
    InitializerKind,
    SpeciesQuantity,
    condition_target_issue,
    has_literal_default,
    is_constant_parameter,
    parameter_table_issue,
)
from petab.v2.models.sbml_model import SbmlModel

K = EntityCategory
T = Dynamics
LIT = Initializer.literal
UNSPECIFIED = Initializer.unspecified()


def kw(**kwargs):
    """Keyword arguments as a dict."""
    return kwargs


def formula(raw, *dependencies):
    return Initializer.formula(raw=raw, dependencies=dependencies)


def assert_consistent(model):
    """`_iter_entities` and `_get_entity` agree."""
    entities = list(model._iter_entities())
    assert len({e.id for e in entities}) == len(entities)
    for info in entities:
        assert model._get_entity(info.id) == info
    return {e.id: e for e in entities}


# --- _entities.py ---


def test_initializer():
    calls = []

    def factory():
        calls.append(1)
        return sp.Symbol("k") * 2

    init = Initializer.formula(
        raw="2*k", dependencies=["k"], expression=factory
    )
    assert init.kind == InitializerKind.FORMULA
    assert init.dependencies == {"k"}
    assert not calls
    assert init.expression == 2 * sp.Symbol("k")
    assert init.expression == 2 * sp.Symbol("k")
    assert calls == [1]
    # the expression factory is not part of equality
    assert init == Initializer.formula(raw="2*k", dependencies=["k"])
    assert Initializer.formula(raw="2*k").expression is None

    assert LIT(2) == Initializer(kind=InitializerKind.LITERAL, value=2.0)
    assert LIT(2).expression is None
    assert UNSPECIFIED.kind == InitializerKind.UNSPECIFIED


def _info(**kwargs):
    defaults = kw(id="x", category=K.PARAMETER, dynamics=T.CONSTANT)
    return EntityInfo(**(defaults | kwargs))


@pytest.mark.parametrize(
    "info, v1, v2",
    [
        (_info(), None, None),
        (_info(event_assigned=True), None, None),
        (_info(dynamics=T.DIFFERENTIAL), None, None),
        (_info(category=K.SPECIES, dynamics=T.DIFFERENTIAL), None, None),
        (
            _info(category=K.SPECIES, initializer=formula("k", "k")),
            None,
            None,
        ),
        (_info(category=K.COMPARTMENT), None, None),
        (_info(dynamics=T.ALGEBRAIC), "warning", "error"),
        (_info(category=K.OTHER_QUANTITY), "error", "error"),
        (_info(category=K.NON_QUANTITY, dynamics=None), "error", "error"),
        (_info(initializer=formula("2*k", "k")), "warning", None),
        (
            _info(category=K.SPECIES, declared=False),
            "warning",
            "warning",
        ),
        (_info(notes=("ambiguous",)), "warning", "warning"),
    ],
)
def test_condition_target_issue(info, v1, v2):
    for version, expected in ((1, v1), (2, v2)):
        issue = condition_target_issue(info, petab_version=version)
        assert (issue and issue.level) == expected
        if issue:
            assert "`x`" in issue.message


@pytest.mark.parametrize(
    "info, v1, v2",
    [
        (_info(), None, None),
        (_info(event_assigned=True), "warning", "error"),
        (_info(dynamics=T.DIFFERENTIAL), "error", "error"),
        (_info(dynamics=T.ALGEBRAIC), "error", "error"),
        (_info(category=K.SPECIES), "error", "error"),
        (_info(category=K.COMPARTMENT), "error", "error"),
        (_info(category=K.OTHER_QUANTITY), "error", "error"),
        (_info(initializer=formula("2*k", "k")), "warning", "warning"),
        (_info(notes=("ambiguous",)), "warning", "warning"),
    ],
)
def test_parameter_table_issue(info, v1, v2):
    for version, expected in ((1, v1), (2, v2)):
        issue = parameter_table_issue(info, petab_version=version)
        assert (issue and issue.level) == expected


def test_issue_functions_require_version():
    with pytest.raises(TypeError):
        condition_target_issue(_info())
    with pytest.raises(ValueError):
        parameter_table_issue(_info(), petab_version=3)


def test_predicates():
    assert is_constant_parameter(_info())
    assert is_constant_parameter(_info(event_assigned=True))
    assert not is_constant_parameter(_info(dynamics=T.DIFFERENTIAL))
    assert not is_constant_parameter(_info(category=K.SPECIES))
    assert has_literal_default(_info(initializer=LIT(1)))
    assert not has_literal_default(_info())
    assert not has_literal_default(_info(initializer=formula("k", "k")))


# --- SBML ---


@pytest.fixture
def sbml_model() -> SbmlModel:
    model = SbmlModel.from_antimony("""
        function f(x)
          2 * x
        end

        model m
        compartment c = 2
        compartment c_unset
        species S in c = 1
        species B in c
        $B = 3
        species B_ev in c
        $B_ev = 3
        species B_rate in c
        $B_rate = 3
        species S_ar in c
        species S_const in c
        const S_const = 5
        species S_free in c
        S_free = 4
        substanceOnly species S_amount in c = 6
        species S_ia in c
        S_ia = 2 * k

        J0: S -> ; k * S

        k = 0.1
        k_ev = 1
        k_ar := 2 * k
        S_ar := 2 * k
        k_rate' = 1
        k_rate = 0
        B_rate' = 1
        k_ia = 2 * k
        k_ia_num = 3^2
        k_ia_func = f(2)

        E1: at time > 1: k_ev = 2, B_ev = 1
        end
    """)
    parameter = model.sbml_model.createParameter()
    parameter.setId("k_unset")
    parameter.setConstant(True)
    return model


@pytest.mark.parametrize(
    "id_, expected",
    [
        (
            "k",
            kw(
                category=K.PARAMETER, dynamics=T.CONSTANT, initializer=LIT(0.1)
            ),
        ),
        (
            "k_ev",
            kw(
                category=K.PARAMETER,
                dynamics=T.CONSTANT,
                event_assigned=True,
                initializer=LIT(1),
            ),
        ),
        ("k_ar", kw(category=K.PARAMETER, dynamics=T.ALGEBRAIC)),
        (
            "k_rate",
            kw(
                category=K.PARAMETER,
                dynamics=T.DIFFERENTIAL,
                initializer=LIT(0),
            ),
        ),
        (
            "k_ia",
            kw(
                category=K.PARAMETER,
                dynamics=T.CONSTANT,
                initializer=formula("2 * k", "k"),
            ),
        ),
        (
            "k_ia_num",
            kw(category=K.PARAMETER, dynamics=T.CONSTANT, initializer=LIT(9)),
        ),
        (
            "k_ia_func",
            kw(
                category=K.PARAMETER,
                dynamics=T.CONSTANT,
                initializer=formula("f(2)", "f"),
            ),
        ),
        ("k_unset", kw(category=K.PARAMETER, dynamics=T.CONSTANT)),
        (
            "S",
            kw(
                category=K.SPECIES,
                dynamics=T.DIFFERENTIAL,
                initializer=LIT(1),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "B",
            kw(
                category=K.SPECIES,
                dynamics=T.CONSTANT,
                initializer=LIT(3),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "B_ev",
            kw(
                category=K.SPECIES,
                dynamics=T.CONSTANT,
                event_assigned=True,
                initializer=LIT(3),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "B_rate",
            kw(
                category=K.SPECIES,
                dynamics=T.DIFFERENTIAL,
                initializer=LIT(3),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "S_ar",
            kw(
                category=K.SPECIES,
                dynamics=T.ALGEBRAIC,
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "S_const",
            kw(
                category=K.SPECIES,
                dynamics=T.CONSTANT,
                initializer=LIT(5),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            # in no reaction, no rule
            "S_free",
            kw(
                category=K.SPECIES,
                dynamics=T.CONSTANT,
                initializer=LIT(4),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "S_amount",
            kw(
                category=K.SPECIES,
                dynamics=T.CONSTANT,
                initializer=LIT(6),
                species_quantity=SpeciesQuantity.AMOUNT,
            ),
        ),
        (
            "S_ia",
            kw(
                category=K.SPECIES,
                dynamics=T.CONSTANT,
                initializer=formula("2 * k", "k"),
                species_quantity=SpeciesQuantity.CONCENTRATION,
            ),
        ),
        (
            "c",
            kw(
                category=K.COMPARTMENT, dynamics=T.CONSTANT, initializer=LIT(2)
            ),
        ),
        ("c_unset", kw(category=K.COMPARTMENT, dynamics=T.CONSTANT)),
        ("J0", kw(category=K.OTHER_QUANTITY, dynamics=T.ALGEBRAIC)),
        (
            "E1",
            kw(category=K.NON_QUANTITY, dynamics=None, usable_in_math=False),
        ),
        (
            "f",
            kw(category=K.NON_QUANTITY, dynamics=None, usable_in_math=False),
        ),
    ],
)
def test_sbml_entities(sbml_model, id_, expected):
    element = sbml_model.sbml_model.getElementBySId(id_)
    expected = EntityInfo(
        id=id_, native_type=f"sbml:{type(element).__name__}", **expected
    )
    assert sbml_model._get_entity(id_) == expected


def test_sbml_consistency(sbml_model):
    entities = assert_consistent(sbml_model)
    assert {"k", "S", "c", "J0", "E1", "f"} <= entities.keys()
    assert sbml_model._get_entity("time") is None
    assert sbml_model._get_entity("nonexistent") is None


def test_sbml_filter(sbml_model):
    ids = {
        e.id
        for e in sbml_model._iter_entities(
            category=K.PARAMETER, dynamics=T.CONSTANT
        )
    }
    assert ids == {"k", "k_ev", "k_ia", "k_ia_num", "k_ia_func", "k_unset"}


def test_sbml_initializer_expression(sbml_model):
    expr = sbml_model._get_entity("k_ia").initializer.expression
    assert {str(s) for s in expr.free_symbols} == {"k"}
    assert float(expr.subs(dict.fromkeys(expr.free_symbols, 0.5))) == 1


def test_sbml_species_other_form():
    """Initial amount for a concentration-based species."""
    model = SbmlModel.from_antimony("""
        compartment c = 2
        species S in c
        S = 1
    """)
    species = model.sbml_model.getSpecies("S")
    species.unsetInitialConcentration()
    species.setInitialAmount(4)
    assert model._get_entity("S").initializer == formula("4.0 / c", "c")
    expr = model._get_entity("S").initializer.expression
    assert float(expr.subs(dict.fromkeys(expr.free_symbols, 2))) == 2


def test_sbml_species_other_form_reserved_compartment_id():
    """A compartment ID that is a constant in L3 formulas is not one here."""
    model = SbmlModel.from_antimony("""
        compartment c = 2
        species S in c
        S = 1
    """)
    model.sbml_model.getCompartment("c").setId("pi")
    species = model.sbml_model.getSpecies("S")
    species.setCompartment("pi")
    species.unsetInitialConcentration()
    species.setInitialAmount(4)
    expr = model._get_entity("S").initializer.expression
    assert {str(s) for s in expr.free_symbols} == {"pi"}


def test_sbml_live_mutation(sbml_model):
    sbml_model.sbml_model.getParameter("k").setValue(7)
    assert sbml_model._get_entity("k").initializer == LIT(7)
    rule = sbml_model.sbml_model.createAssignmentRule()
    rule.setVariable("k")
    rule.setMath(libsbml.parseL3Formula("1"))
    assert sbml_model._get_entity("k").dynamics == T.ALGEBRAIC
    assert sbml_model._get_entity("k").initializer == UNSPECIFIED


def _sbml_with_algebraic_rule(formula: str) -> SbmlModel:
    model = SbmlModel.from_antimony("""
        compartment c = 1
        species S in c = 1
        x = 1
        y = 1
        J0: S -> ; S
    """)
    for id_ in ("x", "y"):
        model.sbml_model.getParameter(id_).setConstant(False)
    rule = model.sbml_model.createAlgebraicRule()
    rule.setMath(libsbml.parseL3Formula(formula))
    return model


def test_sbml_algebraic_rule_single_candidate():
    # `S` is a non-boundary reactant, so `x` is the only candidate
    model = _sbml_with_algebraic_rule("x - S")
    assert model._get_entity("x").dynamics == T.ALGEBRAIC
    assert model._get_entity("x").initializer == UNSPECIFIED
    assert model._get_entity("x").notes == ()
    assert model._get_entity("S").dynamics == T.DIFFERENTIAL
    assert model._get_entity("y").dynamics == T.CONSTANT
    assert_consistent(model)


def test_sbml_algebraic_rule_several_candidates():
    model = _sbml_with_algebraic_rule("x + y - 1")
    for id_ in ("x", "y"):
        info = model._get_entity(id_)
        assert info.dynamics == T.CONSTANT
        assert len(info.notes) == 1
        assert "x, y" in info.notes[0]
        assert condition_target_issue(info, petab_version=2).level == (
            "warning"
        )
    assert_consistent(model)


def test_sbml_algebraic_rules_resolved_in_turn():
    # `0 = y - 2` determines `y`, leaving `x` as the only candidate of
    #  `0 = x + y - 1`
    model = _sbml_with_algebraic_rule("x + y - 1")
    rule = model.sbml_model.createAlgebraicRule()
    rule.setMath(libsbml.parseL3Formula("y - 2"))
    for id_ in ("x", "y"):
        info = model._get_entity(id_)
        assert info.dynamics == T.ALGEBRAIC
        assert info.notes == ()
    assert_consistent(model)


def test_sbml_ids_of_rules():
    """IDs of rules and initial assignments are not entities.

    libsbml does not resolve them by ID.
    """
    model = SbmlModel.from_antimony("""
        a = 1
        b := 2 * a
    """)
    model.sbml_model.getRuleByVariable("b").setIdAttribute("rule1")
    assert model._get_entity("rule1") is None
    assert "rule1" not in assert_consistent(model)


def test_sbml_ids_of_other_elements():
    """IDs of elements that are not enumerated are not entities."""
    model = SbmlModel.from_antimony("""
        S -> ; k * S
        k = 1
        S = 1
    """)
    reaction = model.sbml_model.getReaction(0)
    reaction.getKineticLaw().setIdAttribute("kl")
    assert model.sbml_model.getElementBySId("kl") is not None
    assert model._get_entity("kl") is None
    assert "kl" not in assert_consistent(model)


def test_sbml_species_reference_and_l2_local_parameter():
    document = libsbml.SBMLDocument(2, 4)
    model = document.createModel()
    model.setId("m")
    c = model.createCompartment()
    c.setId("c")
    c.setSize(1)
    s = model.createSpecies()
    s.setId("S")
    s.setCompartment("c")
    s.setInitialConcentration(1)
    reaction = model.createReaction()
    reaction.setId("R")
    reactant = reaction.createReactant()
    reactant.setSpecies("S")
    reactant.setId("sr")
    reactant.setStoichiometry(2)
    kinetic_law = reaction.createKineticLaw()
    local = kinetic_law.createParameter()
    local.setId("k_local")
    local.setValue(1)
    kinetic_law.setMath(libsbml.parseL3Formula("k_local * S"))
    # no stoichiometry: 1 in SBML L2
    product = reaction.createProduct()
    product.setSpecies("S")
    product.setId("sp")
    # stoichiometryMath
    reactant2 = reaction.createReactant()
    reactant2.setSpecies("S")
    reactant2.setId("sr_math")
    reactant2.createStoichiometryMath().setMath(
        libsbml.parseL3Formula("2 * c")
    )

    sbml_model = SbmlModel(sbml_model=model)
    assert not product.isSetStoichiometry()
    assert sbml_model._get_entity("sp").initializer == LIT(1)
    assert sbml_model._get_entity("sr_math").dynamics == T.ALGEBRAIC
    assert sbml_model._get_entity("sr") == EntityInfo(
        id="sr",
        category=K.OTHER_QUANTITY,
        dynamics=T.CONSTANT,
        initializer=LIT(2),
        # no mathematical meaning in SBML L2
        usable_in_math=False,
        native_type="sbml:SpeciesReference",
    )
    assert sbml_model._get_entity("k_local") is None
    entities = assert_consistent(sbml_model)
    assert "k_local" not in entities

"""Test related to ``petab.v2.lint``."""

from copy import deepcopy

import pysb
import pytest
import sympy as sp

from petab.v2 import Problem
from petab.v2.C import LOG_NORMAL, NORMAL, TIME_SYMBOL, UNIFORM
from petab.v2.lint import *
from petab.v2.lint import (
    get_required_parameters_for_parameter_table,
    get_valid_parameters_for_parameter_table,
)
from petab.v2.models.pysb_model import PySBModel
from petab.v2.models.sbml_model import SbmlModel


def test_check_experiments():
    """Test ``CheckExperimentTable``."""
    problem = Problem()

    check = CheckExperimentTable()
    assert check.run(problem) is None

    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_experiment("e2", "-inf", "c1", 1, "c2")
    assert check.run(problem) is None

    tmp_problem = deepcopy(problem)
    tmp_problem["e1"].periods[0].time = tmp_problem["e1"].periods[1].time
    assert check.run(tmp_problem) is not None


def test_check_incompatible_targets():
    """Multiple conditions with overlapping targets cannot be applied
    at the same time."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("p1 = 1; p2 = 2")
    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_condition("c1", p1="1")
    problem.add_condition("c2", p1="2", p2="2")
    check = CheckValidConditionTargets()
    assert check.run(problem) is None

    problem["e1"].periods[0].condition_ids.append("c2")
    assert (error := check.run(problem)) is not None
    assert "overlapping targets {'p1'}" in error.message


def test_invalid_model_id_in_measurements():
    """Test that measurements with an invalid model ID are caught."""
    problem = Problem()
    problem.models.append(SbmlModel.from_antimony("p1 = 1", model_id="model1"))
    problem.add_observable("obs1", "A", 1)
    problem.add_measurement("obs1", experiment_id="e1", time=0, measurement=1)

    check = CheckMeasurementModelId()

    # Single model -> model ID is optional
    assert (error := check.run(problem)) is None, error

    # Two models -> model ID must be set
    problem.models.append(SbmlModel.from_antimony("p2 = 2", model_id="model2"))
    assert (error := check.run(problem)) is not None
    assert "multiple models" in error.message

    # Set model ID to a non-existing model ID
    problem.measurements[0].model_id = "invalid_model_id"
    assert (error := check.run(problem)) is not None
    assert "does not match" in error.message

    # Use a valid model ID
    problem.measurements[0].model_id = "model1"
    assert (error := check.run(problem)) is None, error


def test_undefined_experiment_id_in_measurements():
    """Test that measurements with an undefined experiment ID are caught."""
    problem = Problem()
    problem.add_experiment("e1", 0, "c1")
    problem.add_observable("obs1", "A", 1)
    problem.add_measurement("obs1", experiment_id="e1", time=0, measurement=1)

    check = CheckUndefinedExperiments()

    # Valid experiment ID
    assert (error := check.run(problem)) is None, error

    # Invalid experiment ID
    problem.measurements[0].experiment_id = "invalid_experiment_id"
    assert (error := check.run(problem)) is not None
    assert "not defined" in error.message


def test_validate_initial_change_symbols():
    """Test validation of symbols in target value expressions for changes
    applied at the start of an experiment."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("p1 = 1; p2 = 2")
    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_condition("c1", p1="p2 + time")
    problem.add_condition("c2", p1="p2", p2="p1")
    problem.add_parameter("p1", nominal_value=1, estimate=False)
    problem.add_parameter("p2", nominal_value=2, estimate=False)

    check = CheckInitialChangeSymbols()
    assert check.run(problem) is None

    # removing `p1` from the parameter table is okay, as `c2` is never
    #  used at the start of an experiment
    problem.parameter_tables[0].parameters.remove(problem["p1"])
    assert check.run(problem) is None

    # removing `p2` is not okay, as it is used at the start of an experiment
    problem.parameter_tables[0].parameters.remove(problem["p2"])
    assert (error := check.run(problem)) is not None
    assert "contains additional symbols: {'p2'}" in error.message


def test_check_mapping_table(clean_pysb_exports):
    """Test checks related to the mapping table."""
    problem = Problem()

    # PySB model with a compartment and a monomer, and mapping of model entity
    # to a valid PEtab id
    pysb_model = pysb.Model("test_model")
    pysb.Monomer("A_")
    pysb.Initial(A_() ** pysb.Compartment("C"), pysb.Parameter("a0", 1))
    problem.model = PySBModel(model=pysb_model, model_id="test_model")
    problem.add_mapping("A", "A_() ** C")

    check = CheckMappingTable()
    assert check.run(problem) is None

    check = CheckAllParametersPresentInParameterTable()
    assert check.run(problem) is None

    # add a petab id without model id but with name for annotation
    problem.add_mapping(petab_id="p2", model_id=None, name="Parameter 2")
    problem.add_parameter("p2", estimate=True, nominal_value=1, lb=0, ub=10)

    check = CheckMappingTable()
    assert check.run(problem) is None

    # Valid: mapping table is used purely for annotation
    # petabEntityId appears elsewhere in the problem
    problem.add_mapping(petab_id="c1", model_id=None, name="Condition 1")
    problem.add_condition("c1", p2="1")

    check = CheckMappingTable()
    assert check.run(problem) is None

    # Invalid: petabEntityId is referenced in the model
    clean_pysb_exports()
    pysb_model_invalid = pysb.Model("test_model_invalid")
    pysb.Monomer("A_")
    pysb.Initial(A_() ** pysb.Compartment("C"), pysb.Parameter("A", 1))
    problem.model = PySBModel(
        model=pysb_model_invalid, model_id="test_model_invalid"
    )
    assert check.run(problem) is None
    assert (error := CheckGloballyUniqueIds().run(problem)) is not None
    assert "`A` (mapping table petabEntityId, model entity)" in error.message


def test_pysb_species_with_sites_as_condition_target(
    clean_pysb_exports,
):
    """PySB species with binding sites can be used as condition targets."""
    pysb_model = pysb.Model("test_model", _export=False)
    pysb_model.add_component(
        pysb.Monomer("A", ["b", "s"], {"s": ["u", "p"]}, _export=False)
    )
    pysb_model.add_component(pysb.Monomer("B", ["a"], _export=False))
    problem = Problem()
    problem.model = PySBModel(model=pysb_model, model_id="test_model")
    problem.add_mapping("A_u", "A(b=None, s='u')")
    problem.add_mapping("AB", "A(b=None, s=('p', 1)) % B(a=1)")
    problem.add_condition("c1", A_u=1, AB=2)

    assert problem.model.is_state_variable("A(b=None, s='u')")
    assert problem.model.is_state_variable("A(b=None, s=('p', 1)) % B(a=1)")

    check = CheckValidParameterInConditionOrParameterTable()
    assert check.run(problem) is None
    assert not [
        issue
        for issue in problem.validate()
        if issue.level >= ValidationIssueSeverity.ERROR
    ]


def test_time_in_condition_target_value_not_a_parameter():
    """`time` in condition target values is not a parameter."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("species A = 1")
    problem.add_experiment("e1", 0, "c1")
    problem.add_condition("c1", A="time + 3")

    assert TIME_SYMBOL not in get_required_parameters_for_parameter_table(
        problem
    )
    assert TIME_SYMBOL not in get_valid_parameters_for_parameter_table(problem)
    check = CheckAllParametersPresentInParameterTable()
    assert (error := check.run(problem)) is None, error


def test_time_in_observable_formula_not_a_parameter(clean_pysb_exports):
    """`time` in observable formulas is not a parameter, for any model type."""
    problem = Problem()
    pysb_model = pysb.Model("time_model")
    pysb.Parameter("k", 1)
    problem.model = PySBModel(model=pysb_model, model_id="time_model")
    problem.add_observable("obs1", "k * time", "time")

    assert TIME_SYMBOL not in problem.get_output_parameters()
    check = CheckAllParametersPresentInParameterTable()
    assert (error := check.run(problem)) is None, error


def test_check_unused_placeholders():
    """Test detection of declared placeholders that occur in no formula."""
    problem = Problem()
    problem.model = SbmlModel.from_antimony("xa = 1; xa' = -0.1*xa;")
    check = CheckUnusedPlaceholders()

    # observable placeholder used only in the noise formula is fine
    problem.add_observable(
        "obs_a",
        formula="xa + offset",
        noise_formula="sigma * scale",
        observable_placeholders=["offset", "scale"],
        noise_placeholders=["sigma"],
    )
    assert check.run(problem) is None

    problem.add_observable(
        "obs_b",
        formula="xa + offset",
        noise_formula="1",
        observable_placeholders=["unused_obs", "offset"],
        noise_placeholders=["unused_noise"],
    )
    # noise placeholder occurring only in the observable formula is unused
    problem.add_observable(
        "obs_c",
        formula="xa * sigma",
        noise_formula="1",
        noise_placeholders=["sigma"],
    )
    issue = check.run(problem)
    assert issue.level == ValidationIssueSeverity.WARNING
    assert "obs_a" not in issue.message
    assert (
        "`obs_b`: observable placeholders `unused_obs` are not used"
        in issue.message
    )
    assert (
        "`obs_b`: noise placeholders `unused_noise` are not used"
        in issue.message
    )
    assert "`obs_c`: noise placeholders `sigma` are not used" in issue.message
    assert "offset" not in issue.message
    assert "Consider removing them." in issue.message


@pytest.mark.parametrize(
    "prior_parameters, lb, ub, hint",
    [
        ([0, float("inf")], 0, float("inf"), "finite limits"),
        ([0, 1], 2, 3, "do not define a non-empty interval"),
    ],
)
def test_check_prior_distribution_invalid_uniform(
    prior_parameters, lb, ub, hint
):
    """Invalid uniform priors are reported with an informative hint."""
    problem = Problem()
    problem.add_parameter(
        "p1",
        estimate=True,
        lb=lb,
        ub=ub,
        prior_dist="uniform",
        prior_pars=prior_parameters,
    )
    assert (error := CheckPriorDistribution().run(problem)) is not None
    assert "`p1` are invalid" in error.message
    assert hint in error.message


def test_check_prior_distribution_infinite_bounds():
    """Estimated parameters with infinite bounds need a prior if the problem
    has a MAP objective."""
    problem = Problem()
    problem.add_parameter("p1", estimate=True, lb=0, ub=float("inf"))
    problem.add_parameter("p2", estimate=True, lb=0, ub=1)
    # non-estimated parameters are irrelevant
    problem.add_parameter("p3", estimate=False, nominal_value=1)

    check = CheckPriorDistribution()
    # ML problem: infinite bounds are fine
    assert check.run(problem) is None

    problem["p2"].prior_distribution = NORMAL
    problem["p2"].prior_parameters = [0.5, 1]
    assert (error := check.run(problem)) is not None
    assert error.level == ValidationIssueSeverity.ERROR
    assert "['p1']" in error.message

    problem["p1"].prior_distribution = LOG_NORMAL
    problem["p1"].prior_parameters = [0, 1]
    assert check.run(problem) is None

    # improper explicit prior
    problem["p1"].prior_distribution = UNIFORM
    problem["p1"].prior_parameters = [0, float("inf")]
    assert (error := check.run(problem)) is not None
    assert "`p1` are invalid" in error.message


def _unique_ids_problem() -> Problem:
    problem = Problem()
    problem.model = SbmlModel.from_antimony(
        "species x = 1; k = 1", model_id="model1"
    )
    problem.add_observable(
        "o1", "a * x", noise_formula="1", observable_placeholders=["a"]
    )
    return problem


@pytest.mark.parametrize(
    "modify, expected",
    [
        (
            lambda p: p.add_observable(
                "o2", "a", noise_formula="1", observable_placeholders=["a"]
            ),
            (
                "`a` (observable placeholder of `o1`, "
                "observable placeholder of `o2`)"
            ),
        ),
        (
            lambda p: p.add_observable(
                "o2",
                "b",
                noise_formula="b",
                observable_placeholders=["b"],
                noise_placeholders=["b"],
            ),
            "`b` (observable placeholder of `o2`, noise placeholder of `o2`)",
        ),
        (
            lambda p: p.add_observable(
                "o2", "x", noise_formula="1", observable_placeholders=["x"]
            ),
            "`x` (observable placeholder of `o2`, model entity)",
        ),
        (
            lambda p: p.add_observable(
                "o2", "o1", noise_formula="1", observable_placeholders=["o1"]
            ),
            "`o1` (observable ID, observable placeholder of `o2`)",
        ),
        (
            lambda p: p.add_condition("a", k=2),
            "`a` (condition ID, observable placeholder of `o1`)",
        ),
        (
            lambda p: p.add_experiment("a", 0, "c1"),
            "`a` (experiment ID, observable placeholder of `o1`)",
        ),
        (
            lambda p: p.add_parameter("a", estimate=False, nominal_value=1),
            "`a` (observable placeholder of `o1`, parameter ID)",
        ),
        (
            lambda p: p.add_mapping("a", "net.out[0]"),
            (
                "`a` (observable placeholder of `o1`, "
                "mapping table petabEntityId)"
            ),
        ),
        (
            lambda p: (
                p.add_parameter("a", estimate=False, nominal_value=1),
                p.add_mapping("a", None, name="annotation"),
            ),
            "`a` (observable placeholder of `o1`, parameter ID)",
        ),
        (
            lambda p: p.add_observable("x", "x", noise_formula="1"),
            "`x` (observable ID, model entity)",
        ),
        (
            lambda p: (
                p.add_condition("c1", k=2),
                p.add_experiment("c1", 0, "c1"),
            ),
            "`c1` (condition ID, experiment ID)",
        ),
        (
            lambda p: p.add_condition("model1", k=2),
            "`model1` (condition ID, model ID)",
        ),
    ],
)
def test_check_globally_unique_ids(modify, expected):
    """Test ``CheckGloballyUniqueIds``."""
    problem = _unique_ids_problem()
    check = CheckGloballyUniqueIds()
    assert check.run(problem) is None

    modify(problem)
    assert (error := check.run(problem)) is not None
    assert expected in error.message
    assert error.message.count("`") == expected.count("`")


def test_check_globally_unique_ids_unmodeled_model_entity():
    """Mapping rows without modelEntityId are for entities not in any model."""
    problem = _unique_ids_problem()
    problem.add_mapping("k", None, name="k")
    assert (error := CheckGloballyUniqueIds().run(problem)) is not None
    assert error.message.endswith(
        "`k` (mapping table entry without modelEntityId, model entity)"
    )


def test_check_globally_unique_ids_unmodeled_parameter():
    """A mapping row without modelEntityId for a model entity parameter."""
    problem = _unique_ids_problem()
    problem.add_parameter("k", estimate=True, lb=0, ub=10, nominal_value=1)
    assert CheckGloballyUniqueIds().run(problem) is None

    problem.add_mapping("k", None, name="k")
    assert (error := CheckGloballyUniqueIds().run(problem)) is not None
    assert error.message.endswith(
        "`k` (parameter ID, mapping table entry without modelEntityId, "
        "model entity)"
    )


def test_check_globally_unique_ids_references():
    """References to model entities or mapped IDs are not definitions."""
    problem = _unique_ids_problem()
    # model parameter in the parameter table and as condition target
    problem.add_parameter("k", estimate=True, lb=0, ub=10, nominal_value=1)
    problem.add_condition("c1", k=2)
    problem.add_experiment("e1", 0, "c1")
    # identity mapping
    problem.add_mapping("x", "x")
    # mapped entity in the parameter table
    problem.add_mapping("w", "net.parameters[0]")
    problem.add_parameter("w", estimate=True, lb=0, ub=10, nominal_value=1)
    # mapping row without modelEntityId
    problem.add_mapping("o1", None, name="Observable 1")

    assert CheckGloballyUniqueIds().run(problem) is None


def _multi_model_problem(m2: str = "species y = 1; o1 = 1") -> Problem:
    problem = Problem()
    problem.models = [
        SbmlModel.from_antimony("species x = 1", model_id="m1"),
        SbmlModel.from_antimony(m2, model_id="m2"),
    ]
    problem.add_observable("o1", "x", noise_formula="1")
    problem.add_observable(
        "o2", "a * y", noise_formula="1", observable_placeholders=["a"]
    )
    problem.add_experiment("e1", 0, "c1")
    problem.add_condition("c1", x=2)
    problem.add_measurement(
        "o1", experiment_id="e1", time=1, measurement=1, model_id="m1"
    )
    problem.add_measurement(
        "o2",
        experiment_id="e1",
        time=1,
        measurement=1,
        model_id="m2",
        observable_parameters=[2],
    )
    return problem


def _unique_id_issues(problem: Problem) -> list[ValidationIssue]:
    return [
        issue
        for issue in problem.validate()
        if issue.task == "CheckGloballyUniqueIds"
    ]


def test_check_globally_unique_ids_multi_model():
    """Model entities only clash with IDs used for the same model."""
    # `o1` is an entity of m2, but only measured with m1
    assert _unique_id_issues(_multi_model_problem()) == []

    problem = _multi_model_problem("species y = 1; a = 1")
    assert len(issues := _unique_id_issues(problem)) == 1
    assert issues[0].message == (
        "Identifiers must be unique within the PEtab problem: "
        "`a` (observable placeholder of `o2`, entity of model `m2`)"
    )

    # entity of both models, used by both: reported once
    problem = _multi_model_problem("species y = 1; c1 = 1")
    problem.models[0] = SbmlModel.from_antimony(
        "species x = 1; c1 = 1", model_id="m1"
    )
    assert len(issues := _unique_id_issues(problem)) == 1
    assert issues[0].message.endswith(
        "`c1` (condition ID, entity of model `m1`, entity of model `m2`)"
    )

    # unused entries don't clash with model entities
    problem = _multi_model_problem("species y = 1; z = 1")
    problem.add_condition("z", y=2)
    assert _unique_id_issues(problem) == []
    # ...but with other table entries
    problem.add_observable("z", "y", noise_formula="1")
    assert len(issues := _unique_id_issues(problem)) == 1
    assert issues[0].message.endswith("`z` (condition ID, observable ID)")


@pytest.mark.parametrize(
    "m2, modify, expected",
    [
        # parameter table entries are shared by all models
        (
            "species y = 1; k = 1",
            lambda p: (
                p.add_parameter(
                    "k", estimate=True, lb=0, ub=10, nominal_value=1
                ),
                p.add_observable(
                    "o3",
                    "k * x",
                    noise_formula="1",
                    observable_placeholders=["k"],
                ),
                p.add_measurement(
                    "o3",
                    experiment_id="e1",
                    time=1,
                    measurement=1,
                    model_id="m1",
                    observable_parameters=[2],
                ),
            ),
            (
                "`k` (observable placeholder of `o3`, parameter ID, "
                "entity of model `m2`)"
            ),
        ),
        # model IDs are shared by all models
        (
            "species y = 1; m1 = 1",
            lambda p: None,
            "`m1` (model ID, entity of model `m2`)",
        ),
    ],
)
def test_check_globally_unique_ids_multi_model_shared(m2, modify, expected):
    """IDs shared by all models are compared with all models."""
    assert _unique_id_issues(_multi_model_problem("species y = 1")) == []

    problem = _multi_model_problem(m2)
    modify(problem)
    issues = CheckGloballyUniqueIds().run(problem)
    assert issues is not None
    assert issues.message.endswith(expected)


def test_check_globally_unique_ids_multi_model_tables():
    """Tables are compared across models."""
    problem = _multi_model_problem()
    # experiment used with both models, observable only with m2
    problem.add_observable("e1", "y", noise_formula="1")
    problem.add_measurement(
        "e1", experiment_id="e1", time=1, measurement=1, model_id="m2"
    )
    assert len(issues := _unique_id_issues(problem)) == 1
    assert issues[0].message.endswith("`e1` (experiment ID, observable ID)")


@pytest.mark.parametrize(
    "modify, expected",
    [
        (
            lambda p: p.add_observable("o2", "a + x", noise_formula="1"),
            "`a` of observable `o1` (used in observable `o2`)",
        ),
        (
            lambda p: p.add_observable("o2", "x", noise_formula="a"),
            "`a` of observable `o1` (used in observable `o2`)",
        ),
        (
            lambda p: p.add_condition("c1", a=2),
            "`a` of observable `o1` (used in condition `c1`)",
        ),
        (
            lambda p: p.add_condition("c1", k="a + 1"),
            "`a` of observable `o1` (used in condition `c1`)",
        ),
        (
            lambda p: p.add_measurement(
                "o1", time=1, measurement=1, observable_parameters=["a"]
            ),
            "`a` of observable `o1` (used in measurement table)",
        ),
        (
            lambda p: p.add_observable(
                "o2",
                "b * x",
                noise_formula="b",
                noise_placeholders=["b"],
            ),
            "`b` of observable `o2` (used in observable formula of `o2`)",
        ),
    ],
)
def test_check_placeholders_only_used_in_own_observable(modify, expected):
    """Test ``CheckPlaceholdersOnlyUsedInOwnObservable``."""
    problem = _unique_ids_problem()
    # observable placeholders may be used in the noise formula
    problem["o1"].noise_formula = sp.Symbol("a")
    problem.add_measurement(
        "o1", time=1, measurement=1, observable_parameters=[2]
    )
    check = CheckPlaceholdersOnlyUsedInOwnObservable()
    assert check.run(problem) is None

    modify(problem)
    assert (error := check.run(problem)) is not None
    assert error.message == (
        "Placeholders must only be used in their own observable: " + expected
    )


def test_check_placeholders_only_used_in_own_observable_duplicates():
    """Duplicate placeholders are left to ``CheckGloballyUniqueIds``."""
    problem = _unique_ids_problem()
    problem.add_observable(
        "o2", "a * x", noise_formula="1", observable_placeholders=["a"]
    )
    assert CheckPlaceholdersOnlyUsedInOwnObservable().run(problem) is None


def test_check_observables_do_not_shadow_model_entities_deprecated():
    with pytest.deprecated_call():
        CheckObservablesDoNotShadowModelEntities()

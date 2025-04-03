"""Test related to ``petab.v2.lint``."""

from copy import deepcopy

from petab.v2 import Problem
from petab.v2.lint import *
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

"""Test related to ``petab.v2.lint``."""

from copy import deepcopy

from petab.v2 import Problem
from petab.v2.C import *
from petab.v2.lint import *


def test_check_experiments():
    """Test ``CheckExperimentTable``."""
    problem = Problem()
    problem.add_experiment("e1", 0, "c1", 1, "c2")
    problem.add_experiment("e2", "-inf", "c1", 1, "c2")
    assert problem.experiment_df.shape == (4, 3)

    check = CheckExperimentTable()
    assert check.run(problem) is None

    assert check.run(Problem()) is None

    tmp_problem = deepcopy(problem)
    tmp_problem.experiment_df.loc[0, TIME] = "invalid"
    assert check.run(tmp_problem) is not None

    tmp_problem = deepcopy(problem)
    tmp_problem.experiment_df.loc[0, TIME] = "inf"
    assert check.run(tmp_problem) is not None

    tmp_problem = deepcopy(problem)
    tmp_problem.experiment_df.drop(columns=[TIME], inplace=True)
    assert check.run(tmp_problem) is not None

from math import inf

from petab.v2 import Change, Condition, Experiment, ExperimentPeriod, Problem
from petab.v2.converters import ExperimentsToEventsConverter
from petab.v2.models.sbml_model import SbmlModel


def test_experiments_to_events_converter():
    """Test the ExperimentsToEventsConverter."""
    ant_model = """
    species X = 0
    X' = 1
    """
    problem = Problem()
    problem.model = SbmlModel.from_antimony(ant_model)
    problem.add_condition("c1", X=1)
    problem.add_condition("c2", X=2)
    problem.add_experiment("e1", -inf, "c1", 10, "c2")

    converter = ExperimentsToEventsConverter(problem)
    converted = converter.convert()
    assert converted.validate().has_errors() is False

    assert isinstance(converted.model, SbmlModel)
    sbml_model = converted.model.sbml_model

    assert sbml_model.getNumEvents() == 2
    assert converted.conditions == [
        Condition(
            id="_petab_preequilibration_on",
            changes=[
                Change(
                    target_id="_petab_preequilibration_indicator",
                    target_value=1,
                )
            ],
        ),
        Condition(
            id="_petab_preequilibration_off",
            changes=[
                Change(
                    target_id="_petab_preequilibration_indicator",
                    target_value=0,
                )
            ],
        ),
        Condition(
            id="_petab_experiment_condition_e1",
            changes=[
                Change(
                    target_id="_petab_experiment_indicator_e1", target_value=1
                )
            ],
        ),
    ]
    assert converted.experiments == [
        Experiment(
            id="e1",
            periods=[
                ExperimentPeriod(
                    time=-inf,
                    condition_ids=[
                        "_petab_experiment_condition_e1",
                        "_petab_preequilibration_on",
                    ],
                ),
                ExperimentPeriod(
                    time=10.0,
                    condition_ids=[
                        "_petab_experiment_condition_e1",
                        "_petab_preequilibration_off",
                    ],
                ),
            ],
        ),
    ]

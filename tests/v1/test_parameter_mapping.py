import copy
import os
from math import nan

import numpy as np
import pandas as pd

import petab
from petab.C import *
from petab.models.sbml_model import SbmlModel
from petab.v1.parameter_mapping import _apply_parameter_table

# import fixtures
pytest_plugins = [
    "tests.v1.test_petab",
]


class TestGetSimulationToOptimizationParameterMapping:
    @staticmethod
    def test_no_condition_specific(condition_df_2_conditions):
        # Trivial case - no condition-specific parameters

        condition_df = condition_df_2_conditions
        experiment_df = petab.get_experiment_df(
            pd.DataFrame(
                data={
                    EXPERIMENT_ID: ["experiment1", "experiment2"],
                    EXPERIMENT: [
                        "0:condition1",
                        "0:condition2",
                    ],
                }
            )
        )

        measurement_df = pd.DataFrame(
            data={
                OBSERVABLE_ID: ["obs1", "obs2"],
                EXPERIMENT_ID: ["experiment1", "experiment2"],
                PREEQUILIBRATION_CONDITION_ID: ["", ""],
                OBSERVABLE_PARAMETERS: ["", ""],
                NOISE_PARAMETERS: ["", ""],
            }
        )

        import simplesbml

        ss_model = simplesbml.SbmlModel()
        ss_model.addParameter("dynamicParameter1", 1.0)
        ss_model.addParameter("dynamicParameter2", 2.0)
        ss_model.addParameter("dynamicParameter3", 3.0)

        # add species, which will have initial concentration in condition table
        #  but which should not show up in mapping
        ss_model.addSpecies("[someSpecies]", 1.0)
        condition_df["someSpecies"] = [0.0, 0.0]

        # Test without parameter table
        expected = [
            [
                (
                    {
                        "dynamicParameter1": 1.0,
                        "dynamicParameter2": 2.0,
                        "dynamicParameter3": 3.0,
                        "fixedParameter1": 1.0,
                    },
                    {
                        "dynamicParameter1": LIN,
                        "dynamicParameter2": LIN,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
            [
                (
                    {
                        "dynamicParameter1": 1.0,
                        "dynamicParameter2": 2.0,
                        "dynamicParameter3": 3.0,
                        "fixedParameter1": 2.0,
                    },
                    {
                        "dynamicParameter1": LIN,
                        "dynamicParameter2": LIN,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
        ]

        model = SbmlModel(sbml_model=ss_model.model)
        actual = petab.get_optimization_to_simulation_parameter_mapping(
            model=model,
            measurement_df=measurement_df,
            condition_df=condition_df,
            experiment_df=experiment_df,
        )
        assert actual == expected

        # Test with parameter table
        parameter_df = pd.DataFrame(
            data={
                PARAMETER_ID: [
                    "dynamicParameter1",
                    "dynamicParameter2",
                    "dynamicParameter3",
                ],
                ESTIMATE: [0, 1, 1],
                NOMINAL_VALUE: [11.0, 12.0, None],
                PARAMETER_SCALE: [LOG, LOG10, LIN],
            }
        )
        parameter_df.set_index(PARAMETER_ID, inplace=True)

        expected = [
            [
                (
                    {
                        "dynamicParameter1": 11.0,
                        "dynamicParameter2": "dynamicParameter2",
                        "dynamicParameter3": "dynamicParameter3",
                        "fixedParameter1": 1.0,
                    },
                    {
                        "dynamicParameter1": LIN,
                        "dynamicParameter2": LOG10,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
            [
                (
                    {
                        "dynamicParameter1": 11.0,
                        "dynamicParameter2": "dynamicParameter2",
                        "dynamicParameter3": "dynamicParameter3",
                        "fixedParameter1": 2.0,
                    },
                    {
                        "dynamicParameter1": LIN,
                        "dynamicParameter2": LOG10,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
        ]

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            model=model,
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            experiment_df=experiment_df,
        )

        assert actual == expected

        # Test with applied scaling

        expected = [
            [
                (
                    {
                        "dynamicParameter1": np.log(11.0),
                        "dynamicParameter2": "dynamicParameter2",
                        "dynamicParameter3": "dynamicParameter3",
                        "fixedParameter1": 1.0,
                    },
                    {
                        "dynamicParameter1": LOG,
                        "dynamicParameter2": LOG10,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
            [
                (
                    {
                        "dynamicParameter1": np.log(11.0),
                        "dynamicParameter2": "dynamicParameter2",
                        "dynamicParameter3": "dynamicParameter3",
                        "fixedParameter1": 2.0,
                    },
                    {
                        "dynamicParameter1": LOG,
                        "dynamicParameter2": LOG10,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
        ]

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            model=model,
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            scaled_parameters=True,
            experiment_df=experiment_df,
        )

        assert actual == expected

        # Test without fixed overrides

        expected = [
            [
                (
                    {
                        "dynamicParameter1": "dynamicParameter1",
                        "dynamicParameter2": "dynamicParameter2",
                        "dynamicParameter3": "dynamicParameter3",
                        "fixedParameter1": 1.0,
                    },
                    {
                        "dynamicParameter1": LOG,
                        "dynamicParameter2": LOG10,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
            [
                (
                    {
                        "dynamicParameter1": "dynamicParameter1",
                        "dynamicParameter2": "dynamicParameter2",
                        "dynamicParameter3": "dynamicParameter3",
                        "fixedParameter1": 2.0,
                    },
                    {
                        "dynamicParameter1": LOG,
                        "dynamicParameter2": LOG10,
                        "dynamicParameter3": LIN,
                        "fixedParameter1": LIN,
                    },
                ),
            ],
        ]

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            model=model,
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            fill_fixed_parameters=False,
            experiment_df=experiment_df,
        )

        assert actual == expected

    @staticmethod
    def test_all_override(condition_df_2_conditions):
        # Condition-specific parameters overriding original parameters
        condition_df = condition_df_2_conditions
        experiment_df = petab.get_experiment_df(
            pd.DataFrame(
                data={
                    EXPERIMENT_ID: ["experiment1", "experiment2"],
                    EXPERIMENT: [
                        "0:condition1",
                        "0:condition2",
                    ],
                }
            )
        )

        import simplesbml

        ss_model = simplesbml.SbmlModel()
        ss_model.addParameter("dynamicParameter1", 0.0)
        ss_model.addParameter("dynamicParameter2", 0.0)
        model = SbmlModel(sbml_model=ss_model.model)

        measurement_df = pd.DataFrame(
            data={
                OBSERVABLE_ID: ["obs1", "obs2", "obs1", "obs2"],
                EXPERIMENT_ID: [
                    "experiment1",
                    "experiment1",
                    "experiment2",
                    "experiment2",
                ],
                OBSERVABLE_PARAMETERS: [
                    "obs1par1override;obs1par2cond1override",
                    "obs2par1cond1override",
                    "obs1par1override;obs1par2cond2override",
                    "obs2par1cond2override",
                ],
                NOISE_PARAMETERS: ["", "", "", ""],
            }
        )

        parameter_df = pd.DataFrame(
            data={
                PARAMETER_ID: [
                    "dynamicParameter1",
                    "dynamicParameter2",
                    "obs1par1override",
                    "obs1par2cond1override",
                    "obs1par2cond2override",
                    "obs2par1cond1override",
                    "obs2par1cond2override",
                ],
                ESTIMATE: [1] * 7,
            }
        )
        parameter_df.set_index(PARAMETER_ID, inplace=True)

        expected = [
            [
                (
                    {
                        "fixedParameter1": 1.0,
                        "dynamicParameter1": "dynamicParameter1",
                        "dynamicParameter2": "dynamicParameter2",
                        "observableParameter1_obs1": "obs1par1override",
                        "observableParameter2_obs1": "obs1par2cond1override",
                        "observableParameter1_obs2": "obs2par1cond1override",
                    },
                    {
                        "fixedParameter1": LIN,
                        "dynamicParameter1": LIN,
                        "dynamicParameter2": LIN,
                        "observableParameter1_obs1": LIN,
                        "observableParameter2_obs1": LIN,
                        "observableParameter1_obs2": LIN,
                    },
                ),
            ],
            [
                (
                    {
                        "fixedParameter1": 2.0,
                        "dynamicParameter1": "dynamicParameter1",
                        "dynamicParameter2": "dynamicParameter2",
                        "observableParameter1_obs1": "obs1par1override",
                        "observableParameter2_obs1": "obs1par2cond2override",
                        "observableParameter1_obs2": "obs2par1cond2override",
                    },
                    {
                        "fixedParameter1": LIN,
                        "dynamicParameter1": LIN,
                        "dynamicParameter2": LIN,
                        "observableParameter1_obs1": LIN,
                        "observableParameter2_obs1": LIN,
                        "observableParameter1_obs2": LIN,
                    },
                ),
            ],
        ]

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            model=model,
            parameter_df=parameter_df,
            experiment_df=experiment_df,
        )
        assert actual == expected

        # For one case we test parallel execution, which must yield the same
        # result
        os.environ[petab.ENV_NUM_THREADS] = "4"
        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            model=model,
            parameter_df=parameter_df,
            experiment_df=experiment_df,
        )
        assert actual == expected

    @staticmethod
    def test_partial_override(condition_df_2_conditions):
        # Condition-specific parameters, keeping original parameters
        condition_df = copy.deepcopy(condition_df_2_conditions)
        condition_df["fixedParameter2"] = [nan, 2.5]
        experiment_df = petab.get_experiment_df(
            pd.DataFrame(
                data={
                    EXPERIMENT_ID: ["experiment1", "experiment2"],
                    EXPERIMENT: [
                        "0:condition1",
                        "0:condition2",
                    ],
                }
            )
        )

        import simplesbml

        ss_model = simplesbml.SbmlModel()
        ss_model.addParameter("fixedParameter1", 0.5)
        ss_model.addParameter("fixedParameter2", 1.0)
        ss_model.addParameter("dynamicParameter1", 0.0)
        ss_model.addParameter("observableParameter1_obs1", 0.0)
        ss_model.addParameter("observableParameter2_obs1", 0.0)
        ss_model.addParameter("observableParameter1_obs2", 0.0)

        measurement_df = pd.DataFrame(
            data={
                OBSERVABLE_ID: ["obs1", "obs2", "obs1", "obs2"],
                EXPERIMENT_ID: [
                    "experiment1",
                    "experiment1",
                    "experiment2",
                    "experiment2",
                ],
                OBSERVABLE_PARAMETERS: [
                    "obs1par1override;obs1par2cond1override",
                    "",
                    "obs1par1override;obs1par2cond2override",
                    "obs2par1cond2override",
                ],
                NOISE_PARAMETERS: ["", "", "", ""],
            }
        )

        parameter_df = pd.DataFrame(
            data={
                PARAMETER_ID: [
                    "dynamicParameter1",
                    "obs1par1override",
                    "obs1par2cond1override",
                    "obs1par2cond2override",
                    "obs2par1cond2override",
                ],
                ESTIMATE: [1, 1, 1, 1, 1],
            }
        )
        parameter_df.set_index(PARAMETER_ID, inplace=True)

        expected = [
            [
                (
                    {
                        "fixedParameter1": 1.0,
                        "fixedParameter2": 1.0,
                        "dynamicParameter1": "dynamicParameter1",
                        "observableParameter1_obs1": "obs1par1override",
                        "observableParameter2_obs1": "obs1par2cond1override",
                        "observableParameter1_obs2": np.nan,
                    },
                    {
                        "fixedParameter1": LIN,
                        "fixedParameter2": LIN,
                        "dynamicParameter1": LIN,
                        "observableParameter1_obs1": LIN,
                        "observableParameter2_obs1": LIN,
                        "observableParameter1_obs2": LIN,
                    },
                ),
            ],
            [
                (
                    {
                        "fixedParameter1": 2.0,
                        "fixedParameter2": 2.5,
                        "dynamicParameter1": "dynamicParameter1",
                        "observableParameter1_obs1": "obs1par1override",
                        "observableParameter2_obs1": "obs1par2cond2override",
                        "observableParameter1_obs2": "obs2par1cond2override",
                    },
                    {
                        "fixedParameter1": LIN,
                        "fixedParameter2": LIN,
                        "dynamicParameter1": LIN,
                        "observableParameter1_obs1": LIN,
                        "observableParameter2_obs1": LIN,
                        "observableParameter1_obs2": LIN,
                    },
                ),
            ],
        ]

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            model=petab.models.sbml_model.SbmlModel(ss_model.model),
            parameter_df=parameter_df,
            experiment_df=experiment_df,
        )

        # Comparison with NaN containing expected results fails after pickling!
        # Need to test first for correct NaNs, then for the rest.
        assert np.isnan(expected[0][0][0]["observableParameter1_obs2"])
        assert np.isnan(actual[0][0][0]["observableParameter1_obs2"])
        expected[0][0][0]["observableParameter1_obs2"] = 0.0
        actual[0][0][0]["observableParameter1_obs2"] = 0.0

        assert actual == expected

    @staticmethod
    def test_parameterized_condition_table():
        condition_df = pd.DataFrame(
            data={
                CONDITION_ID: ["condition1", "condition2", "condition3"],
                CONDITION_NAME: ["", "Condition 2", ""],
                "dynamicParameter1": [
                    "dynamicOverride1_1",
                    "dynamicOverride1_2",
                    0,
                ],
            }
        )
        condition_df.set_index(CONDITION_ID, inplace=True)

        experiment_df = petab.get_experiment_df(
            pd.DataFrame(
                data={
                    EXPERIMENT_ID: [
                        "experiment1",
                        "experiment2",
                        "experiment3",
                    ],
                    EXPERIMENT: [
                        "0:condition1",
                        "0:condition2",
                        "0:condition3",
                    ],
                }
            )
        )

        measurement_df = pd.DataFrame(
            data={
                EXPERIMENT_ID: [
                    "experiment1",
                    "experiment2",
                    "experiment3",
                ],
                OBSERVABLE_ID: ["obs1", "obs2", "obs1"],
                OBSERVABLE_PARAMETERS: "",
                NOISE_PARAMETERS: "",
            }
        )

        parameter_df = pd.DataFrame(
            data={
                PARAMETER_ID: ["dynamicOverride1_1", "dynamicOverride1_2"],
                PARAMETER_NAME: ["", "..."],
                ESTIMATE: [1, 1],
            }
        )
        parameter_df.set_index(PARAMETER_ID, inplace=True)

        import simplesbml

        ss_model = simplesbml.SbmlModel()
        ss_model.addParameter("dynamicParameter1", 1.0)

        assert petab.get_model_parameters(ss_model.model) == [
            "dynamicParameter1"
        ]

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            model=petab.models.sbml_model.SbmlModel(ss_model.model),
            experiment_df=experiment_df,
        )

        expected = [
            [
                (
                    {"dynamicParameter1": "dynamicOverride1_1"},
                    {"dynamicParameter1": LIN},
                ),
            ],
            [
                (
                    {"dynamicParameter1": "dynamicOverride1_2"},
                    {"dynamicParameter1": LIN},
                ),
            ],
            [
                (
                    {"dynamicParameter1": 0},
                    {"dynamicParameter1": LIN},
                ),
            ],
        ]

        assert actual == expected

    @staticmethod
    def test_parameterized_condition_table_changed_scale():
        """Test overriding a dynamic parameter `overridee` with
        - a log10 parameter to be estimated (condition 1)
        - lin parameter not estimated (condition2)
        - log10 parameter not estimated (condition 3)
        - constant override (condition 4)
        """
        # overridden parameter
        overridee_id = "overridee"

        # set up model
        import simplesbml

        ss_model = simplesbml.SbmlModel()
        ss_model.addParameter(overridee_id, 2.0)
        assert petab.get_model_parameters(ss_model.model) == [overridee_id]
        assert petab.get_model_parameters(
            ss_model.model, with_values=True
        ) == {overridee_id: 2.0}

        # set up condition table
        condition_df = pd.DataFrame(
            data={
                CONDITION_ID: [
                    "condition1",
                    "condition2",
                    "condition3",
                    "condition4",
                ],
                overridee_id: [
                    "dynamicOverrideLog10",
                    "fixedOverrideLin",
                    "fixedOverrideLog10",
                    10.0,
                ],
            }
        )
        condition_df.set_index("conditionId", inplace=True)

        experiment_df = petab.get_experiment_df(
            pd.DataFrame(
                data={
                    EXPERIMENT_ID: [
                        "experiment1",
                        "experiment2",
                        "experiment3",
                        "experiment4",
                    ],
                    EXPERIMENT: [
                        "0:condition1",
                        "0:condition2",
                        "0:condition3",
                        "0:condition4",
                    ],
                }
            )
        )

        # set up measurement table
        measurement_df = pd.DataFrame(
            data={
                EXPERIMENT_ID: [
                    "experiment1",
                    "experiment2",
                    "experiment3",
                    "experiment4",
                ],
                OBSERVABLE_ID: ["obs1", "obs2", "obs1", "obs2"],
                OBSERVABLE_PARAMETERS: "",
                NOISE_PARAMETERS: "",
            }
        )

        # set up parameter table
        parameter_df = pd.DataFrame(
            data={
                PARAMETER_ID: [
                    "dynamicOverrideLog10",
                    "fixedOverrideLin",
                    "fixedOverrideLog10",
                ],
                ESTIMATE: [1, 0, 0],
                NOMINAL_VALUE: [np.nan, -2, 1000],
                PARAMETER_SCALE: [LOG10, LIN, LOG10],
            }
        )
        parameter_df.set_index(PARAMETER_ID, inplace=True)

        # test without preequilibration condition; unscaled known parameters

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            model=petab.models.sbml_model.SbmlModel(ss_model.model),
            experiment_df=experiment_df,
        )

        expected = [
            [
                (
                    {overridee_id: "dynamicOverrideLog10"},
                    {overridee_id: LOG10},
                ),
            ],
            [
                (
                    {overridee_id: -2.0},
                    {overridee_id: LIN},
                ),
            ],
            [
                # not scaled:
                (
                    {overridee_id: 1000.0},
                    {overridee_id: LIN},
                ),
            ],
            [
                (
                    {overridee_id: 10.0},
                    {overridee_id: LIN},
                ),
            ],
        ]

        assert actual == expected

        # test without preequilibration condition; scaled known parameters

        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            model=petab.models.sbml_model.SbmlModel(ss_model.model),
            scaled_parameters=True,
            experiment_df=experiment_df,
        )

        expected = [
            [
                (
                    {overridee_id: "dynamicOverrideLog10"},
                    {overridee_id: LOG10},
                ),
            ],
            [
                (
                    {overridee_id: -2.0},
                    {overridee_id: LIN},
                ),
            ],
            [
                # scaled fixedOverrideLog10:
                (
                    {overridee_id: 3.0},
                    {overridee_id: LOG10},
                ),
            ],
            [
                (
                    {overridee_id: 10.0},
                    {overridee_id: LIN},
                ),
            ],
        ]

        assert actual == expected

        # Add preeq condition

        experiment_df[EXPERIMENT] = [
            "-inf:condition1;0:condition1",
            "-inf:condition1;0:condition2",
            "-inf:condition3;0:condition3",
            "-inf:condition3;0:condition4",
        ]
        actual = petab.get_optimization_to_simulation_parameter_mapping(
            measurement_df=measurement_df,
            condition_df=condition_df,
            parameter_df=parameter_df,
            model=petab.models.sbml_model.SbmlModel(ss_model.model),
            experiment_df=experiment_df,
        )

        expected = [
            [
                (
                    {overridee_id: "dynamicOverrideLog10"},
                    {overridee_id: LOG10},
                ),
                (
                    {overridee_id: "dynamicOverrideLog10"},
                    {overridee_id: LOG10},
                ),
            ],
            [
                (
                    {overridee_id: "dynamicOverrideLog10"},
                    {overridee_id: LOG10},
                ),
                (
                    {overridee_id: -2.0},
                    {overridee_id: LIN},
                ),
            ],
            [
                # not rescaled:
                (
                    {overridee_id: 1000.0},
                    {overridee_id: LIN},
                ),
                (
                    {overridee_id: 1000.0},
                    {overridee_id: LIN},
                ),
            ],
            [
                (
                    {overridee_id: 1000.0},
                    {overridee_id: LIN},
                ),
                (
                    {overridee_id: 10.0},
                    {overridee_id: LIN},
                ),
            ],
        ]

        assert actual == expected


def test_fill_in_nominal_values():
    parameter_df = pd.DataFrame(
        data={
            PARAMETER_ID: ["estimated", "not_estimated"],
            NOMINAL_VALUE: [0.0, 2.0],
            ESTIMATE: [1, 0],
        }
    )
    parameter_df.set_index([PARAMETER_ID], inplace=True)

    par_mapping = {"estimated": "estimated", "not_estimated": "not_estimated"}
    scale_mapping = {"estimated": LIN, "not_estimated": LIN}
    _apply_parameter_table(par_mapping, scale_mapping, parameter_df)

    expected_par = {"estimated": "estimated", "not_estimated": 2.0}
    expected_scale = {"estimated": LIN, "not_estimated": LIN}

    assert expected_par == par_mapping
    assert expected_scale == scale_mapping

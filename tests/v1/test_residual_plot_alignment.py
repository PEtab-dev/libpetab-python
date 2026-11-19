"""Regression tests for matching plotted simulations to measurements."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from petab.v1 import Problem
from petab.v1.visualize import (
    plot_goodness_of_fit,
    plot_residuals_vs_simulation,
)

plt.switch_backend("agg")


@pytest.mark.parametrize(
    "plot", [plot_goodness_of_fit, plot_residuals_vs_simulation]
)
@pytest.mark.parametrize("replicates", [False, True])
def test_simulation_order_does_not_change_plot(plot, replicates):
    """Reordering simulation rows must not change pairs or statistics."""
    measurements = pd.DataFrame(
        {
            "observableId": ["obs", "other", "obs"],
            "simulationConditionId": ["c1", "c2", "c1"],
            "time": [0, 0, 1],
            "measurement": [2.0, 7.0, 12.0],
            "preequilibrationConditionId": [np.nan] * 3,
        },
        index=[5, 5, 9],
    )
    simulations = measurements.rename(
        columns={"measurement": "simulation"}
    ).copy()
    simulations["simulation"] = [1.0, 3.0, 5.0]
    if replicates:
        measurements = pd.concat([measurements, measurements.iloc[[0]]])
    problem = Problem(
        measurement_df=measurements,
        observable_df=pd.DataFrame(
            {"noiseFormula": ["1", "1"]}, index=["obs", "other"]
        ),
        parameter_df=pd.DataFrame(),
    )
    shuffled = simulations.iloc[[2, 0, 1]]
    before = shuffled.copy(deep=True)
    measurements_before = measurements.copy(deep=True)
    try:
        result = plot(problem, shuffled)
        ax = result if plot is plot_goodness_of_fit else result[0]
        x = [1, 3, 5] + ([1] if replicates else [])
        y = [2, 7, 12] if plot is plot_goodness_of_fit else [1, 4, 7]
        if replicates:
            y += y[:1]
        np.testing.assert_allclose(
            ax.collections[
                0 if plot is plot_goodness_of_fit else 1
            ].get_offsets(),
            np.column_stack([x, y]),
        )
        if plot is plot_goodness_of_fit and not replicates:
            assert "slope: 2.50" in ax.texts[0].get_text()
        pd.testing.assert_frame_equal(shuffled, before)
        pd.testing.assert_frame_equal(measurements, measurements_before)
    finally:
        plt.close("all")

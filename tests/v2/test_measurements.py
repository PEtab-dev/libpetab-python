import pandas as pd

from petab.v2.C import EXPERIMENT_ID
from petab.v2.measurements import get_measured_experiments


def test_get_measured_experiments():
    """Test get_measured_experiments"""
    measurement_df = pd.DataFrame(
        data={
            EXPERIMENT_ID: ["c0", "c1", "c0", "c1"],
        }
    )
    expected = ["c0", "c1"]
    actual = get_measured_experiments(measurement_df)
    assert actual == expected

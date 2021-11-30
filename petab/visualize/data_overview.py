"""
Functions for creating an overview report of a PEtab problem
"""

from pathlib import Path
from shutil import copyfile
from typing import Union

import pandas as pd
import petab
from petab.C import *

__all__ = ['create_report']


def create_report(
        problem: petab.Problem,
        model_name: str,
        output_path: Union[str, Path] = ''
) -> None:
    """Create an HTML overview data / model overview report

    Arguments:
        problem: PEtab problem
        model_name: Name of the model, used for file name for report
        output_path: Output directory
    """

    template_dir = Path(__file__).absolute().parent / 'templates'
    output_path = Path(output_path)
    template_file = "report.html"

    data_per_observable = get_data_per_observable(problem.measurement_df)
    num_conditions = len(problem.condition_df.index)

    # Setup template engine
    import jinja2
    template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(template_file)

    # Render and save
    output_text = template.render(problem=problem, model_name=model_name,
                                  data_per_observable=data_per_observable,
                                  num_conditions=num_conditions)
    with open(output_path / f'{model_name}.html', 'w') as html_file:
        html_file.write(output_text)
    copyfile(template_dir / 'mystyle.css', output_path / 'mystyle.css')


def get_data_per_observable(measurement_df: pd.DataFrame) -> pd.DataFrame:
    """Get table with number of data points per observable and condition

    Arguments:
        measurement_df: PEtab measurement data frame
    Returns:
        Pivot table with number of data points per observable and condition
    """

    my_measurements = measurement_df.copy()
    index = [SIMULATION_CONDITION_ID]
    if PREEQUILIBRATION_CONDITION_ID in my_measurements:
        my_measurements[PREEQUILIBRATION_CONDITION_ID].fillna('', inplace=True)
        index.append(PREEQUILIBRATION_CONDITION_ID)

    data_per_observable = pd.pivot_table(
        my_measurements, values=MEASUREMENT, aggfunc='count',
        index=index,
        columns=[OBSERVABLE_ID], fill_value=0)

    # Add row and column sums
    data_per_observable.loc['SUM', :] = data_per_observable.sum(axis=0).values
    data_per_observable['SUM'] = data_per_observable.sum(axis=1).values

    data_per_observable = data_per_observable.astype(int)

    return data_per_observable

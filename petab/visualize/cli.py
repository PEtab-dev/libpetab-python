"""Command-line interface for visualization."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .plot_data_and_simulation import plot_problem
from .. import Problem, get_simulation_df, get_visualization_df


def _parse_cli_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description='Create PEtab visualizations.')

    parser.add_argument('-y', '--yaml', dest='yaml_file_name', required=True,
                        help='PEtab problem YAML filename')
    parser.add_argument('-s', '--simulations', dest='simulation_file_name',
                        required=False,
                        help='PEtab simulation filename')
    parser.add_argument('-o', '--output-directory', dest='output_directory',
                        required=True, help='Output directory')
    parser.add_argument('-v', '--visualizations', required=False,
                        dest='visualization_file_name',
                        help='PEtab visualization specification filename')
    parser.add_argument('--style', required=False,
                        dest='style_file_name',
                        help='Matplotlib style file')
    return parser.parse_args()


def _petab_visualize_main():
    """Entrypoint for visualization command-line interface."""
    args = _parse_cli_args()

    petab_problem = Problem.from_yaml(args.yaml_file_name)
    simulations_df = None
    if args.simulation_file_name:
        simulations_df = get_simulation_df(args.simulation_file_name)

    if args.visualization_file_name:
        petab_problem.visualization_df = get_visualization_df(
            args.visualization_file_name)

    if args.style_file_name:
        plt.style.use(args.style_file_name)

    # Avoid errors when plotting without X server
    plt.switch_backend('agg')

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    plot_problem(
        petab_problem=petab_problem,
        simulations_df=simulations_df,
        subplot_dir=args.output_directory,
    )

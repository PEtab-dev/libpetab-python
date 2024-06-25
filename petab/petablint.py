#!/usr/bin/env python3

"""Command line tool to check for correct format"""

import argparse
import logging
import sys

from colorama import Fore
from colorama import init as init_colorama

import petab

logger = logging.getLogger(__name__)


class LintFormatter(logging.Formatter):
    """Custom log formatter"""

    formats = {
        logging.DEBUG: Fore.CYAN + "%(message)s",
        logging.INFO: Fore.GREEN + "%(message)s",
        logging.WARN: Fore.YELLOW + "%(message)s",
        logging.ERROR: Fore.RED + "%(message)s",
    }

    def format(self, record):
        # pylint: disable=protected-access
        format_orig = self._style._fmt
        self._style._fmt = LintFormatter.formats.get(record.levelno, self._fmt)
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


def parse_cli_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Check if a set of files adheres to the PEtab format."
    )

    # General options:
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="More verbose output",
    )

    # Call with set of files
    group = parser.add_argument_group("Check individual files *DEPRECATED*. Please contact us via GitHub issues, if you need this.")
    group.add_argument(
        "-s", "--sbml", dest="sbml_file_name", help="SBML model filename"
    )
    group.add_argument(
        "-o",
        "--observables",
        dest="observable_file_name",
        help="Observable table",
    )
    group.add_argument(
        "-m",
        "--measurements",
        dest="measurement_file_name",
        help="Measurement table",
    )
    group.add_argument(
        "-c",
        "--conditions",
        dest="condition_file_name",
        help="Conditions table",
    )
    group.add_argument(
        "-p",
        "--parameters",
        dest="parameter_file_name",
        help="Parameter table",
    )
    group.add_argument(
        "--vis",
        "--visualizations",
        dest="visualization_file_name",
        help="Visualization table",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-y",
        "--yaml",
        dest="yaml_file_name_deprecated",
        help="PEtab YAML problem filename. "
        "*DEPRECATED* pass the file name as positional argument instead.",
    )
    group.add_argument(
        dest="yaml_file_name",
        help="PEtab YAML problem filename",
        nargs="?",
    )

    args = parser.parse_args()
    if (args.yaml_file_name or args.yaml_file_name_deprecated) and any(
        (
            args.sbml_file_name,
            args.condition_file_name,
            args.measurement_file_name,
            args.parameter_file_name,
        )
    ):
        parser.error(
            "When providing a yaml file, no other files may be specified."
        )

    if args.yaml_file_name_deprecated:
        logger.warning(
            "The -y/--yaml option is deprecated. "
            "Please provide the YAML file as a positional argument."
        )
        if args.yaml_file_name:
            parser.error(
                "Please provide only one of --yaml or positional argument."
            )

    args.yaml_file_name = args.yaml_file_name or args.yaml_file_name_deprecated

    return args


def main():
    """Run PEtab validator"""
    init_colorama(autoreset=True)
    ch = logging.StreamHandler()
    ch.setFormatter(LintFormatter())
    logging.basicConfig(level=logging.DEBUG, handlers=[ch])

    args = parse_cli_args()

    if args.verbose:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.WARN)

    if args.yaml_file_name:
        from jsonschema.exceptions import ValidationError

        from petab.yaml import validate

        try:
            validate(args.yaml_file_name)
        except ValidationError as e:
            logger.error(
                "Provided YAML file does not adhere to PEtab " f"schema: {e}"
            )
            sys.exit(1)

        if petab.is_composite_problem(args.yaml_file_name):
            # TODO: further checking:
            #  https://github.com/ICB-DCM/PEtab/issues/191
            #  problem = petab.CompositeProblem.from_yaml(args.yaml_file_name)
            return

        problem = petab.Problem.from_yaml(args.yaml_file_name)

    else:
        # DEPRECATED
        logger.debug("Looking for...")
        if args.sbml_file_name:
            logger.debug(f"\tSBML model: {args.sbml_file_name}")
        if args.condition_file_name:
            logger.debug(f"\tCondition table: {args.condition_file_name}")
        if args.observable_file_name:
            logger.debug(f"\tObservable table: {args.observable_file_name}")
        if args.measurement_file_name:
            logger.debug(f"\tMeasurement table: {args.measurement_file_name}")
        if args.parameter_file_name:
            logger.debug(f"\tParameter table: {args.parameter_file_name}")
        if args.visualization_file_name:
            logger.debug(
                "\tVisualization table: " f"{args.visualization_file_name}"
            )

        try:
            problem = petab.Problem.from_files(
                sbml_file=args.sbml_file_name,
                condition_file=args.condition_file_name,
                measurement_file=args.measurement_file_name,
                parameter_file=args.parameter_file_name,
                observable_files=args.observable_file_name,
                visualization_files=args.visualization_file_name,
            )
        except FileNotFoundError as e:
            logger.error(e)
            sys.exit(1)

    ret = petab.lint.lint_problem(problem)
    sys.exit(ret)


if __name__ == "__main__":
    main()

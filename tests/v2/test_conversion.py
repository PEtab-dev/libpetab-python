import logging
import tempfile

import pytest

from petab.v2.petab1to2 import petab1to2


def test_petab1to2_remote():
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )

    with tempfile.TemporaryDirectory(prefix="test_petab1to2") as tmpdirname:
        # TODO verify that the v2 files match "ground truth"
        # in `petabtests/cases/v2.0.0/sbml/0001/_0001.yaml`
        petab1to2(yaml_url, tmpdirname)


try:
    import benchmark_models_petab

    parametrize_or_skip = pytest.mark.parametrize(
        "problem_id", benchmark_models_petab.MODELS
    )
except ImportError:
    parametrize_or_skip = pytest.mark.skip(
        reason="benchmark_models_petab not installed"
    )


@parametrize_or_skip
def test_benchmark_collection(problem_id):
    """Test that we can upgrade all benchmark collection models."""
    logging.basicConfig(level=logging.DEBUG)

    if problem_id == "Froehlich_CellSystems2018":
        pytest.skip("Too slow. Re-enable once we are faster.")

    yaml_path = benchmark_models_petab.get_problem_yaml_path(problem_id)
    with tempfile.TemporaryDirectory(
        prefix=f"test_petab1to2_{problem_id}"
    ) as tmpdirname:
        petab1to2(yaml_path, tmpdirname)

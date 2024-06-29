import logging
import tempfile

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


def test_benchmark_collection():
    """Test that we can upgrade all benchmark collection models."""
    import benchmark_models_petab

    logging.basicConfig(level=logging.DEBUG)

    for problem_id in benchmark_models_petab.MODELS:
        if problem_id == "Lang_PLOSComputBiol2024":
            # Does not pass initial linting
            continue

        yaml_path = benchmark_models_petab.get_problem_yaml_path(problem_id)
        with tempfile.TemporaryDirectory(
            prefix=f"test_petab1to2_{problem_id}"
        ) as tmpdirname:
            petab1to2(yaml_path, tmpdirname)

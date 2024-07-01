from petab.v2 import Problem


def test_load_remote():
    """Test loading remote files"""
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v2.0.0/sbml/0001/_0001.yaml"
    )
    petab_problem = Problem.from_yaml(yaml_url)

    assert (
        petab_problem.measurement_df is not None
        and not petab_problem.measurement_df.empty
    )

    assert petab_problem.validate() == []


def test_auto_upgrade():
    yaml_url = (
        "https://raw.githubusercontent.com/PEtab-dev/petab_test_suite"
        "/main/petabtests/cases/v1.0.0/sbml/0001/_0001.yaml"
    )
    problem = Problem.from_yaml(yaml_url)
    # TODO check something specifically different in a v2 problem
    assert isinstance(problem, Problem)

import benchmark_models_petab
import pytest

import petab.v1

# from pypesto.petab import PetabImporter
from petab.v1.priors import priors_to_measurements


@pytest.mark.parametrize(
    "problem_id", ["Schwen_PONE2014", "Isensee_JCB2018", "Raimundez_PCB2020"]
)
def test_priors_to_measurements(problem_id):
    petab_problem_priors: petab.v1.Problem = (
        benchmark_models_petab.get_problem(problem_id)
    )
    assert petab.v1.lint_problem(petab_problem_priors) is False

    petab_problem_measurements = priors_to_measurements(petab_problem_priors)
    assert petab.v1.lint_problem(petab_problem_measurements) is False
    assert (
        petab_problem_measurements.parameter_df.shape[0]
        == petab_problem_priors.parameter_df.shape[0]
    )
    assert (
        petab_problem_measurements.observable_df.shape[0]
        > petab_problem_priors.observable_df.shape[0]
    )
    assert (
        petab_problem_measurements.measurement_df.shape[0]
        > petab_problem_priors.measurement_df.shape[0]
    )

    # test via pypesto
    # different model IDs to avoid collision of amici modules
    petab_problem_measurements.model.model_id += "_measurements"
    petab_problem_priors.model.model_id += "_priors"

    from pypesto.petab import PetabImporter

    importer = PetabImporter(petab_problem_priors)
    problem_p = importer.create_problem(force_compile=True)

    importer = PetabImporter(petab_problem_measurements)
    problem_m = importer.create_problem(force_compile=True)

    xs = problem_m.startpoint_method(n_starts=10, problem=problem_m)

    for x in [petab_problem_priors.x_nominal_free_scaled, *xs]:
        fx_p = problem_p.objective(x)
        fx_m = problem_m.objective(x)
        print(
            fx_p, fx_m, fx_p - fx_m, (fx_p - fx_m) / max(abs(fx_p), abs(fx_m))
        )

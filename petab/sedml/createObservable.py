# create the observable data file in the PEtab format

from petab import *
from petab.migrations import sbml_observables_to_table
from ..yaml import *



def update(model_name, model_name_2):
    """Update model"""
    model_dir = f'{model_name}/'
    yaml_file = f'{model_name_2}/{model_name}.yaml'
    yaml_config = load_yaml(yaml_file)

    if is_composite_problem(yaml_config):
        print(f"SKIPPING {model_name}: Cannot handle PEtab problems with "
              "multiple models yet")
        return

    problem = Problem.from_yaml(yaml_file)
    sbml_observables_to_table(problem)

    # save updated files
    p = yaml_config['problems'][0]
    for field in [SBML_FILES, CONDITION_FILES, MEASUREMENT_FILES,
                  VISUALIZATION_FILES]:
        if field in p and not len(p[field]) <= 1:
            raise NotImplementedError(f"Cannot handle multiple {field}")

    problem.to_files(
        sbml_file=model_dir + p[SBML_FILES][0],
        # condition_file=model_dir + p[CONDITION_FILES][0],
        measurement_file=model_dir + p[MEASUREMENT_FILES][0],
        # parameter_file=model_dir + yaml_config[PARAMETER_FILE]
        observable_file=model_dir + f"observables_{model_name}.tsv",
    )


def observablesPETAB(sedml_file_name):

    benchmark_model = sedml_file_name
    benchmark_model_2 = './sedml2petab/' + sedml_file_name + '/' + sedml_file_name
    print('# ', benchmark_model)
    try:
        update(benchmark_model, benchmark_model_2)
    except RuntimeError as e:
        print(e)

    print('=' * 100)
    # break

    obsdatafile_save_path = ''
    return obsdatafile_save_path
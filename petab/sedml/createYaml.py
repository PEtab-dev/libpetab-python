# create .yaml file for simulation in COPASI

import os
import pandas as pd
import yaml
import petab
from copy import deepcopy


def create_petab_yaml(model_dir, model_dir_2):

    filename = f'{model_dir}.yaml'
    config = {
        'format_version': petab.__format_version__,
        'parameter_file':
            petab.get_default_parameter_file_name(model_dir),
        'problems': [
            {
                'sbml_files':
                    [
                        petab.get_default_sbml_file_name(model_dir),
                    ],
                'condition_files':
                    [
                        petab.get_default_condition_file_name(model_dir),
                    ],
                'measurement_files':
                    [
                        petab.get_default_measurement_file_name(model_dir),
                    ],
            },
        ]
    }

    # Add observable file if exists
    obs_file_name = f"observables_{model_dir}.tsv"
    if os.path.isfile(os.path.join(model_dir, obs_file_name)):
        config['problems'][0]['observable_files'] = [obs_file_name]

    # Add visualization file if exists
    vis_file_name =  f"visualizationSpecification_{model_dir}.tsv"
    if os.path.isfile(os.path.join(model_dir, vis_file_name)):
        config['problems'][0]['visualization_files'] = [vis_file_name]

    data = [
        (filename, config),
    ]

    # exceptions
    if model_dir == 'Becker_Science2010':
        config['problems'].append(deepcopy(config['problems'][0]))
        config['problems'][0]['sbml_files'] = [
            'model_Becker_Science2010__BaF3_Exp.xml']
        config['problems'][0]['condition_files'] = [
            'experimentalCondition_Becker_Science2010__BaF3_Exp.tsv']
        config['problems'][0]['measurement_files'][0] = \
            'measurementData_Becker_Science2010__BaF3_Exp.tsv'
        config['problems'][1]['sbml_files'] = [
            'model_Becker_Science2010__binding.xml']
        config['problems'][1]['condition_files'] = [
            'experimentalCondition_Becker_Science2010__binding.tsv']
        config['problems'][1]['measurement_files'][0] = \
            'measurementData_Becker_Science2010__binding.tsv'
        # simulationData_Becker_Science2010__BaF3_Exp.tsv
        # simulationData_Becker_Science2010__binding.tsv

    elif model_dir in ('Casaletto_PNAS2019', 'Merkle_PCB2016'):
        print(f"# ERROR: {model_dir}: model missing")
        return

    for filename, config in data:
        petab.validate(config, path_prefix=model_dir_2)

        out_file = os.path.join(model_dir_2, filename)
        print('git add ' + out_file)
        with open(out_file, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    return outfile


def yamlCOPASI(sedml_file_name):

    benchmark_model = sedml_file_name
    benchmark_model_2 = 'sedml2petab/' + sedml_file_name + '/' + sedml_file_name
    print('# ', benchmark_model)
    try:
        yaml_save_path = create_petab_yaml(benchmark_model, benchmark_model_2)
    except ValueError as e:
        print(e)

    # print('='*100)
    # break

    return yaml_save_path
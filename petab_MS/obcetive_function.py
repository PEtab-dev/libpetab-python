from . import problem
import pandas as pd
import os
import importlib
import sys


def get_objective_function(objective_func_path: str):
    type = problem.check_value_type(objective_func_path)
    if type is "python_file":
        head, tail = os.path.split(objective_func_path)
        tail_info = tuple(val for val in tail.split(';'))
        filename, ext = os.path.splitext(tail_info[0])
        module_path = head + '/' + tail_info[0]
        module_name = filename
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        new_module = getattr(module, tail_info[1])
    return new_module
    if type is "r_file":
        measurement_df_dict = {}
        filename, functionname = name.split(";")
        path = os.path.abspath(os.path.join(measurement_file, os.pardir))
        r = R(os.path.join(path, filename))
        tmp_measurement_df = r.observation(functionname)
        measurement_df_dict[condition] = tmp_measurement_df
        return 0
    return 0

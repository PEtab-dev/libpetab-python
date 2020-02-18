import libsedml
import libsbml
import pandas
import os
from .recreateExperimental import *
from .buildSEDML import *

def petab2sedml(petab_folder_path):                                                                                     #exp_save_path, meas_save_path, par_save_path, sbml_save_path):

    # create the basic experimental data file
    sbml_file_name, base_path, exp_con_save_path, new_exp_save_path, new_meas_save_path, new_par_save_path, new_sbml_save_path = recreateExpDataFile(petab_folder_path)                                                                              #exp_save_path, meas_save_path, par_save_path, sbml_save_path)

    # create the basic SEDML file
    new_sbml_file_name, sedml_save_path = createSEDML(sbml_file_name, base_path, exp_con_save_path, new_sbml_save_path)
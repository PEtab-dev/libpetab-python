from .sedml_import import *
import shutil
import libsedml
import libsbml
from .downloadSEDML import *
from .downloadSBML import *
from .getExperimentalData import *
from .rearrangeExperimentalData import *
from .getObservables import *
from .createExperimental import *
from .createMeasurement import *
from .createParameters import *
from .extendSBML import *
from .petabFolder import *
from .createObservable import *
from .createYaml import *
from ..C import *


def sedml2petab(sedml_path, sedml_file_name, output_folder=None):

    # download sedml file
    sedml_save_path, sbml_save_path = downloadAllSEDML(sedml_path, sedml_file_name)

    # download sbml files
    sbml_save_path, sbml_id = downloadAllSBML(sedml_save_path, sbml_save_path)

    # download experimental data
    getAllExperimentalDataFiles(sedml_path, sedml_file_name)

    # rearrange experimental data file into petab format
    exp_rearrange_save_path = rearrange2PEtab(sedml_path, sedml_file_name)

   # add observables to sbml file
    new_sbml_save_path = getAllObservables(sedml_save_path, sbml_save_path, sedml_file_name, sbml_id)

    # create experimental_condition file
    expconfile_save_path = experimentalPETAB(sedml_save_path, sedml_file_name)

    # create measurement_date file
    measdatafile_save_path = measurementPETAB(exp_rearrange_save_path, sedml_file_name)

    # create parameters file
    parfile_save_path = parameterPETAB(new_sbml_save_path, sedml_file_name, measdatafile_save_path)

    # extend the sbml_file_with_observables by 'noise_', 'sigma_', 'observable_' as [parameters] and [assignment_rules]
    newest_sbml_save_path = editSBML(new_sbml_save_path, sedml_file_name, parfile_save_path)

    # create petab folder with all ingredients
    final_petab_order_path = restructureFiles(expconfile_save_path, measdatafile_save_path, parfile_save_path, newest_sbml_save_path, sedml_file_name)

    # yaml file --- first version (needed for observablePETAB)
    yaml_save_path = yamlCOPASI(sedml_file_name)

    # create observable data file by writing sbml_observables to new table
    obsdatafile_save_path = observablesPETAB(sedml_file_name)

    # yaml file --- final version (with observables from observablesPETAB)
    yaml_save_path = yamlCOPASI(sedml_file_name)
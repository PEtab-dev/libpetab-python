import libsedml
import os
import urllib.request
import shutil
import json
import logging
from .logging_util import LOGGER_NAME


logger = logging.getLogger(LOGGER_NAME)


BASE_URL = "https://jjj.bio.vu.nl"
BASE_FOLDER = "./sedml_models"


def get_sedml_infos_from_rest_api():
    """
    Get list of all jws sedml models infos.
    """
    rest_url = BASE_URL + "/rest/experiments/?format=json"
    response = urllib.request.urlopen(rest_url)
    json_string = response.read().decode('utf-8')
    sedml_infos = json.loads(json_string)
    return sedml_infos


def download_all_sedml_models_from_jws(base_folder=BASE_FOLDER):
    """
    Download all sedml models to xml files.
    """
    # download list of sedml model infos
    sedml_infos = get_sedml_infos_from_rest_api()

    # download every single sedml model
    for sedml_info in sedml_infos:
        # model identifier
        sedml_slug = sedml_info['slug']
        # an own folder for the sedml model
        sedml_folder = base_folder + "/" + sedml_slug
        sedml_file = sedml_folder + "/" + sedml_slug + ".sedml"
        download_sedml_model(sedml_slug, sedml_file)


def download_sedml_model(sedml_slug, sedml_file):
    """
    Download the sedml model specified by `sedml_slug` from jws
    to `sedml_file`.
    """
    # url of the sedml model
    sedml_url = BASE_URL + "/models/experiments/" + sedml_slug \
        + "/export/sedml"
    
    logger.info(f"Downloading sedml model {sedml_slug} from "
                f"{sedml_url} to file {sedml_file}.")


    '''
    # create folder
    sedml_folder = os.path.dirname(sedml_file)
    if not os.path.exists(sedml_folder):
        os.makedirs(sedml_folder)
    
    try:
        with urllib.request.urlopen(sedml_url) as response, \
                open(sedml_file, 'wb') as f:
            shutil.copyfileobj(response, f)

        # folder for sbml models
        sbml_folder = sedml_folder + "/" + "sbml_models"

        # download all sbml models for the sedml model to file
        download_all_sbml_models_for_sedml_model(sedml_file, sbml_folder)

    except Exception as e:
        logger.warn(f"Failed to download sedml model {sedml_slug} "
                    f"from {sedml_url}, {e}.")
    '''

def download_all_sbml_models_for_sedml_model(sedml_file, sbml_folder):
    """
    Download all sbml models used in a sedml model to a subfolder.
    """
    # read sedml file
    sedml_model = libsedml.readSedML(sedml_file)

    # create sbml model folder if not exists
    if not os.path.exists(sbml_folder):
        os.makedirs(sbml_folder)

    # extract sbml entries
    n_sbml_models = sedml_model.getNumModels()
    for i_sbml_model in range(n_sbml_models):
        sbml_entry = sedml_model.getModel(i_sbml_model)
        sbml_id = sbml_entry.getId()
        sbml_url = sbml_entry.getSource()
        sbml_file = sbml_folder + "/" + sbml_id + ".sbml"
        download_sbml_model(sbml_url, sbml_file)
    return sbml_file                                                                        # returns only the last save path --- only for i = 1

def download_sbml_model(sbml_url, sbml_file):
    """
    Download one sbml model from `sbml_url` to `sbml_file`.
    """
    logger.info(f"  Downloading sbml model from {sbml_url} to file "
                f"{sbml_file}.")

    try:
        with urllib.request.urlopen(sbml_url) as response, \
                open(sbml_file, 'wb') as f:
            shutil.copyfileobj(response, f)
    except Exception as e:
        logger.warn(f"Failed to download sbml model from {sbml_url}, {e}.")

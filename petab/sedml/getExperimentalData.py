# script to get the .csv file of experimental data

import os
import libsedml
import urllib.request
import sys


def getAllExperimentalDataFiles(sedml_path, sedml_file_name):

    # create new folder to save experimental data file
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/experimental_data'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/experimental_data')

    # load sedml
    sedml_file = libsedml.readSedML(sedml_path)

    # get all experimental data files
    for iData in range(0, sedml_file.getNumDataDescriptions()):
        try:
            # parse source url from data description
            data = sedml_file.getDataDescription(iData)
            data_id = data.getId()
            data_source = data.getSource()

            # download file
            urllib.request.urlretrieve(data_source, './sedml2petab/' + sedml_file_name + '/experimental_data/' + data_id + '.xls')

        except:
            print('No experimental data files!')

    # delete empty folders of experimental data
    if len(os.listdir('./sedml2petab/' + sedml_file_name + '/experimental_data')) == 0:
        print('The experimental data file could not be downloaded!')
        sys.exit()
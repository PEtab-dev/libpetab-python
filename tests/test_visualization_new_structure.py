import warnings
from os import path
from tempfile import TemporaryDirectory
import pytest
from petab.C import *
from petab import Problem
from petab.visualize.plotter import (MPLPlotter)
from petab.visualize.plotting import Figure, VisualisationSpec
import matplotlib.pyplot as plt


@pytest.fixture
def data_file_Fujita():
    return "doc/example/example_Fujita/Fujita_measurementData.tsv"


@pytest.fixture
def condition_file_Fujita():
    return "doc/example/example_Fujita/Fujita_experimentalCondition.tsv"


@pytest.fixture
def data_file_Fujita_wrongNoise():
    return "doc/example/example_Fujita/Fujita_measurementData_wrongNoise.tsv"


@pytest.fixture
def data_file_Fujita_nanData():
    return "doc/example/example_Fujita/Fujita_measurementData_nanData.tsv"


@pytest.fixture
def simu_file_Fujita():
    return "doc/example/example_Fujita/Fujita_simulatedData.tsv"


@pytest.fixture
def data_file_Fujita_minimal():
    return "doc/example/example_Fujita/Fujita_measurementData_minimal.tsv"


@pytest.fixture
def visu_file_Fujita_small():
    return "doc/example/example_Fujita/Fujita_visuSpec_small.tsv"


@pytest.fixture
def visu_file_Fujita_wo_dsid():
    return "doc/example/example_Fujita/visuSpecs/Fujita_visuSpec_1.tsv"


@pytest.fixture
def visu_file_Fujita_minimal():
    return "doc/example/example_Fujita/visuSpecs/Fujita_visuSpec_mandatory.tsv"


@pytest.fixture
def visu_file_Fujita_empty():
    return "doc/example/example_Fujita/visuSpecs/Fujita_visuSpec_empty.tsv"


@pytest.fixture
def data_file_Isensee():
    return "doc/example/example_Isensee/Isensee_measurementData.tsv"


@pytest.fixture
def condition_file_Isensee():
    return "doc/example/example_Isensee/Isensee_experimentalCondition.tsv"


@pytest.fixture
def vis_spec_file_Isensee():
    return "doc/example/example_Isensee/Isensee_visualizationSpecification.tsv"


@pytest.fixture
def simulation_file_Isensee():
    return "doc/example/example_Isensee/Isensee_simulationData.tsv"


def test_visualization(data_file_Isensee,
                       condition_file_Isensee,
                       vis_spec_file_Isensee,
                       simulation_file_Isensee):

    figure = Figure(condition_file_Isensee,
                    data_file_Fujita,
                    vis_spec_file_Isensee)
    plotter = MPLPlotter(figure)
    plotter.generate_plot()


def test_VisualizationSpec():
    test_spec = {'plotName':'test_plot',
                 'plotTypeSimulation':'test_plot_type',
                 'plotTypeData': 'test_data_type',
                 'xValues': 'test_xValues',
                 'xScale': 'test_xScale',
                 'yScale': 'test_yScale',
                 'legendEntry': 'test_legend',
                 'datasetId': ['test_dataset_id'],
                 'yValues': ['test_yValue'],
                 'yOffset': ['test_yOffset'],
                 'xOffset': ['test_xOffset'],
                 'xLabel': 'test_xLabel',
                 'yLabel': 'test_yLabel'
                  }
    assert {**{'figureId': 'fig0', PLOT_ID: 'plot0'}, **test_spec} == \
        VisualisationSpec(plot_id='plot0', plot_settings=test_spec).__dict__


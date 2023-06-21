import subprocess
from os import path
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pytest

import petab
from petab.C import *
from petab.visualize import plot_with_vis_spec, plot_without_vis_spec, \
    plot_residuals_vs_simulation, plot_goodness_of_fit
from petab.visualize.plotting import VisSpecParser
from petab.visualize.lint import validate_visualization_df

# Avoid errors when plotting without X server
plt.switch_backend('agg')

EXAMPLE_DIR = Path(__file__).parents[1] / "doc" / "example"


@pytest.fixture(scope="function")
def close_fig():
    """Close all open matplotlib figures"""
    yield
    plt.close('all')


@pytest.fixture
def data_file_Fujita():
    return EXAMPLE_DIR / "example_Fujita" / "Fujita_measurementData.tsv"


@pytest.fixture
def condition_file_Fujita():
    return EXAMPLE_DIR / "example_Fujita" / "Fujita_experimentalCondition.tsv"


@pytest.fixture
def data_file_Fujita_wrongNoise():
    return EXAMPLE_DIR / "example_Fujita" \
           / "Fujita_measurementData_wrongNoise.tsv"


@pytest.fixture
def data_file_Fujita_nanData():
    return EXAMPLE_DIR / "example_Fujita" \
           / "Fujita_measurementData_nanData.tsv"


@pytest.fixture
def simu_file_Fujita():
    return EXAMPLE_DIR / "example_Fujita" \
                      / "Fujita_simulatedData.tsv"


@pytest.fixture
def simu_file_Fujita_t_inf():
    return EXAMPLE_DIR / "example_Fujita" \
                      / "Fujita_simulatedData_t_inf.tsv"


@pytest.fixture
def data_file_Fujita_minimal():
    return EXAMPLE_DIR / "example_Fujita"\
           / "Fujita_measurementData_minimal.tsv"


@pytest.fixture
def data_file_Fujita_t_inf():
    return EXAMPLE_DIR / "example_Fujita"\
           / "Fujita_measurementData_t_inf.tsv"


@pytest.fixture
def visu_file_Fujita_small():
    return EXAMPLE_DIR / "example_Fujita" / "Fujita_visuSpec_small.tsv"


@pytest.fixture
def visu_file_Fujita_wo_dsid_wo_yvalues():
    return EXAMPLE_DIR / "example_Fujita" / "visuSpecs" \
           / "Fujita_visuSpec_1.tsv"


@pytest.fixture
def visu_file_Fujita_all_obs_with_diff_settings():
    return EXAMPLE_DIR / "example_Fujita" / "visuSpecs" \
           / "Fujita_visuSpec_3.tsv"


@pytest.fixture
def visu_file_Fujita_minimal():
    return EXAMPLE_DIR / "example_Fujita" / "visuSpecs"\
           / "Fujita_visuSpec_mandatory.tsv"


@pytest.mark.filterwarnings("ignore:Visualization table is empty")
@pytest.fixture
def visu_file_Fujita_empty():
    return EXAMPLE_DIR / "example_Fujita" / "visuSpecs" \
           / "Fujita_visuSpec_empty.tsv"


@pytest.fixture
def visu_file_Fujita_replicates():
    return EXAMPLE_DIR / "example_Fujita" / "visuSpecs" \
           / "Fujita_visuSpec_replicates.tsv"


@pytest.fixture
def data_file_Isensee():
    return EXAMPLE_DIR / "example_Isensee" / "Isensee_measurementData.tsv"


@pytest.fixture
def condition_file_Isensee():
    return EXAMPLE_DIR / "example_Isensee" \
           / "Isensee_experimentalCondition.tsv"


@pytest.fixture
def vis_spec_file_Isensee():
    return EXAMPLE_DIR / "example_Isensee" \
           / "Isensee_visualizationSpecification.tsv"


@pytest.fixture
def vis_spec_file_Isensee_replicates():
    return EXAMPLE_DIR / "example_Isensee" \
           / "Isensee_visualizationSpecification_replicates.tsv"


@pytest.fixture
def vis_spec_file_Isensee_scatterplot():
    return EXAMPLE_DIR / "example_Isensee" \
           / "Isensee_visualizationSpecification_scatterplot.tsv"


@pytest.fixture
def simulation_file_Isensee():
    return EXAMPLE_DIR / "example_Isensee" / "Isensee_simulationData.tsv"


def test_visualization_with_vis_and_sim(data_file_Isensee,
                                        condition_file_Isensee,
                                        vis_spec_file_Isensee,
                                        simulation_file_Isensee,
                                        close_fig):
    validate_visualization_df(
        petab.Problem(
            condition_df=petab.get_condition_df(condition_file_Isensee),
            visualization_df=petab.get_visualization_df(vis_spec_file_Isensee),
        )
    )
    plot_with_vis_spec(vis_spec_file_Isensee, condition_file_Isensee,
                       data_file_Isensee, simulation_file_Isensee)


def test_visualization_replicates(data_file_Isensee,
                                  condition_file_Isensee,
                                  vis_spec_file_Isensee_replicates,
                                  simulation_file_Isensee,
                                  close_fig):
    plot_with_vis_spec(vis_spec_file_Isensee_replicates,
                       condition_file_Isensee,
                       data_file_Isensee, simulation_file_Isensee)


def test_visualization_scatterplot(data_file_Isensee,
                                   condition_file_Isensee,
                                   vis_spec_file_Isensee_scatterplot,
                                   simulation_file_Isensee,
                                   close_fig):
    plot_with_vis_spec(vis_spec_file_Isensee_scatterplot,
                       condition_file_Isensee,
                       data_file_Isensee, simulation_file_Isensee)


def test_visualization_small_visu_file_w_datasetid(data_file_Fujita,
                                                   condition_file_Fujita,
                                                   visu_file_Fujita_small,
                                                   close_fig):
    """
    Test: visualization specification file only with few columns in
    particular datasetId
    (optional columns are optional)
    """
    plot_with_vis_spec(visu_file_Fujita_small, condition_file_Fujita,
                       data_file_Fujita)


def test_visualization_small_visu_file_wo_datasetid(
        data_file_Fujita,
        condition_file_Fujita,
        visu_file_Fujita_wo_dsid_wo_yvalues,
        close_fig):
    """
    Test: visualization specification file only with few columns in
    particular no datasetId column
    (optional columns are optional)
    """
    plot_with_vis_spec(visu_file_Fujita_wo_dsid_wo_yvalues,
                       condition_file_Fujita, data_file_Fujita)


def test_visualization_all_obs_with_diff_settings(
        data_file_Fujita,
        condition_file_Fujita,
        visu_file_Fujita_all_obs_with_diff_settings,
        close_fig):
    """
    Test: visualization specification file only with few columns. In
    particular, no datasetId column and no yValues column, but more than one
    plot id. Additionally, having plot id different from 'plot\\d+' for the
    case of vis_spec expansion is tested.
    """
    plot_with_vis_spec(visu_file_Fujita_all_obs_with_diff_settings,
                       condition_file_Fujita, data_file_Fujita)


def test_visualization_minimal_visu_file(data_file_Fujita,
                                         condition_file_Fujita,
                                         visu_file_Fujita_minimal,
                                         close_fig):
    """
    Test: visualization specification file only with mandatory column plotId
    (optional columns are optional)
    """
    plot_with_vis_spec(visu_file_Fujita_minimal, condition_file_Fujita,
                       data_file_Fujita)


def test_visualization_empty_visu_file(data_file_Fujita,
                                       condition_file_Fujita,
                                       visu_file_Fujita_empty,
                                       close_fig):
    """
    Test: Empty visualization specification file should default to routine
    for no file at all
    """
    with pytest.warns(UserWarning, match="Visualization table is empty."):
        plot_with_vis_spec(visu_file_Fujita_empty, condition_file_Fujita,
                           data_file_Fujita)


def test_visualization_minimal_data_file(data_file_Fujita_minimal,
                                         condition_file_Fujita,
                                         visu_file_Fujita_wo_dsid_wo_yvalues,
                                         close_fig):
    """
    Test visualization, with the case: data file only with mandatory columns
    (optional columns are optional)
    """
    plot_with_vis_spec(visu_file_Fujita_wo_dsid_wo_yvalues,
                       condition_file_Fujita, data_file_Fujita_minimal)


def test_visualization_with_dataset_list(data_file_Isensee,
                                         condition_file_Isensee,
                                         simulation_file_Isensee,
                                         close_fig):
    datasets = [['JI09_150302_Drg345_343_CycNuc__4_ABnOH_and_ctrl',
                 'JI09_150302_Drg345_343_CycNuc__4_ABnOH_and_Fsk'],
                ['JI09_160201_Drg453-452_CycNuc__ctrl',
                 'JI09_160201_Drg453-452_CycNuc__Fsk',
                 'JI09_160201_Drg453-452_CycNuc__Sp8_Br_cAMPS_AM']]

    # TODO: is condition_file needed here
    plot_without_vis_spec(condition_file_Isensee, datasets, 'dataset',
                          data_file_Isensee)

    plot_without_vis_spec(condition_file_Isensee, datasets, 'dataset',
                          data_file_Isensee, simulation_file_Isensee)


def test_visualization_without_datasets(data_file_Fujita,
                                        condition_file_Fujita,
                                        simu_file_Fujita,
                                        close_fig):

    sim_cond_id_list = [['model1_data1'], ['model1_data2', 'model1_data3'],
                        ['model1_data4', 'model1_data5'], ['model1_data6']]

    observable_id_list = [['pS6_tot'], ['pEGFR_tot'], ['pAkt_tot']]

    plot_without_vis_spec(condition_file_Fujita, sim_cond_id_list,
                          'simulation', data_file_Fujita,
                          plotted_noise=PROVIDED)

    plot_without_vis_spec(condition_file_Fujita, observable_id_list,
                          'observable', data_file_Fujita,
                          plotted_noise=PROVIDED)

    # with simulations

    plot_without_vis_spec(condition_file_Fujita, sim_cond_id_list,
                          'simulation', data_file_Fujita, simu_file_Fujita,
                          plotted_noise=PROVIDED)

    plot_without_vis_spec(condition_file_Fujita, observable_id_list,
                          'observable', data_file_Fujita, simu_file_Fujita,
                          plotted_noise=PROVIDED)


def test_visualization_only_simulations(condition_file_Fujita,
                                        simu_file_Fujita,
                                        close_fig):

    sim_cond_id_list = [['model1_data1'], ['model1_data2', 'model1_data3'],
                        ['model1_data4', 'model1_data5'], ['model1_data6']]

    observable_id_list = [['pS6_tot'], ['pEGFR_tot'], ['pAkt_tot']]

    plot_without_vis_spec(condition_file_Fujita, sim_cond_id_list,
                          'simulation', simulations_df=simu_file_Fujita,
                          plotted_noise=PROVIDED)

    plot_without_vis_spec(condition_file_Fujita, observable_id_list,
                          'observable', simulations_df=simu_file_Fujita,
                          plotted_noise=PROVIDED)


def test_simple_visualization(
        data_file_Fujita, condition_file_Fujita, close_fig
):
    plot_without_vis_spec(condition_file_Fujita,
                          measurements_df=data_file_Fujita)
    plot_without_vis_spec(condition_file_Fujita,
                          measurements_df=data_file_Fujita,
                          plotted_noise=PROVIDED)


def test_visualization_with__t_inf(data_file_Fujita_t_inf,
                                   simu_file_Fujita_t_inf,
                                   condition_file_Fujita,
                                   visu_file_Fujita_replicates,
                                   close_fig):
    # plot only measurements
    plot_without_vis_spec(condition_file_Fujita,
                          measurements_df=data_file_Fujita_t_inf)

    # plot only simulation
    plot_without_vis_spec(condition_file_Fujita,
                          simulations_df=simu_file_Fujita_t_inf)

    # plot both measurements and simulation
    plot_without_vis_spec(condition_file_Fujita,
                          measurements_df=data_file_Fujita_t_inf,
                          simulations_df=simu_file_Fujita_t_inf)

    # plot both measurements and simulation
    plot_with_vis_spec(visu_file_Fujita_replicates,
                       condition_file_Fujita,
                       measurements_df=data_file_Fujita_t_inf,
                       simulations_df=simu_file_Fujita_t_inf)


def test_save_plots_to_file(data_file_Isensee, condition_file_Isensee,
                            vis_spec_file_Isensee, simulation_file_Isensee,
                            close_fig):
    with TemporaryDirectory() as temp_dir:
        plot_with_vis_spec(vis_spec_file_Isensee, condition_file_Isensee,
                           data_file_Isensee, simulation_file_Isensee,
                           subplot_dir=temp_dir)


def test_save_visu_file(data_file_Isensee,
                        condition_file_Isensee):

    with TemporaryDirectory() as temp_dir:

        vis_spec_parser = VisSpecParser(condition_file_Isensee,
                                        data_file_Isensee)
        figure, _ = vis_spec_parser.parse_from_id_list()

        with pytest.warns(
                UserWarning,
                match="please check that datasetId column corresponds to"
        ):
            figure.save_to_tsv(path.join(temp_dir, "visuSpec.tsv"))

        datasets = [['JI09_150302_Drg345_343_CycNuc__4_ABnOH_and_ctrl',
                     'JI09_150302_Drg345_343_CycNuc__4_ABnOH_and_Fsk'],
                    ['JI09_160201_Drg453-452_CycNuc__ctrl',
                     'JI09_160201_Drg453-452_CycNuc__Fsk',
                     'JI09_160201_Drg453-452_CycNuc__Sp8_Br_cAMPS_AM']]

        vis_spec_parser = VisSpecParser(condition_file_Isensee,
                                        data_file_Isensee)
        figure, _ = vis_spec_parser.parse_from_id_list(datasets,
                                                       group_by='dataset')
        with pytest.warns(
                UserWarning,
                match="please check that datasetId column corresponds to"
        ):
            figure.save_to_tsv(path.join(temp_dir, "visuSpec1.tsv"))


def test_residuals_plot(simu_file_Fujita, close_fig):
    fujita_yaml = EXAMPLE_DIR / "example_Fujita" / "Fujita.yaml"
    fujita_petab_problem = petab.Problem.from_yaml(fujita_yaml)
    plot_residuals_vs_simulation(fujita_petab_problem, simu_file_Fujita)


def test_goodness_of_fit_plot(simu_file_Fujita, close_fig):
    fujita_yaml = EXAMPLE_DIR / "example_Fujita" / "Fujita.yaml"
    fujita_petab_problem = petab.Problem.from_yaml(fujita_yaml)
    plot_goodness_of_fit(fujita_petab_problem, simu_file_Fujita)


def test_cli():
    fujita_dir = EXAMPLE_DIR / "example_Fujita"

    with TemporaryDirectory() as temp_dir:
        args = [
            "petab_visualize",
            "-y", str(fujita_dir / "Fujita.yaml"),
            "-s", str(fujita_dir / "Fujita_simulatedData.tsv"),
            "-o", temp_dir
        ]
        subprocess.run(args, check=True)


@pytest.mark.filterwarnings("ignore:Visualization table is empty")
@pytest.mark.parametrize(
    "vis_file",
    (
            "vis_spec_file_Isensee",
            "vis_spec_file_Isensee_replicates",
            "vis_spec_file_Isensee_scatterplot",
            "visu_file_Fujita_wo_dsid_wo_yvalues",
            "visu_file_Fujita_all_obs_with_diff_settings",
            "visu_file_Fujita_empty",
            "visu_file_Fujita_minimal",
            "visu_file_Fujita_replicates",
            "visu_file_Fujita_small",
    )
)
def test_validate(vis_file, request):
    """Check that all test files pass validation."""
    vis_file = request.getfixturevalue(vis_file)
    assert False is validate_visualization_df(
        petab.Problem(visualization_df=petab.get_visualization_df(vis_file))
    )

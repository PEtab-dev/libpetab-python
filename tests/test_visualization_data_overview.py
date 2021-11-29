from pathlib import Path
from tempfile import TemporaryDirectory

import petab
from petab.visualize.data_overview import create_report


def test_data_overview():
    """Data overview generation with Fujita example data from this repository
    """
    with TemporaryDirectory() as temp_dir:
        outfile = Path(temp_dir) / 'Fujita.html'
        repo_root = Path(__file__).parent.parent
        yaml_filename = (repo_root / 'doc' / 'example' / 'example_Fujita'
                         / 'Fujita.yaml')
        problem = petab.Problem.from_yaml(yaml_filename)
        create_report(problem, 'Fujita', output_path=temp_dir)
        assert outfile.is_file()

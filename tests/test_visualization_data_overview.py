from pathlib import Path
from tempfile import TemporaryDirectory

from petab.visualize.data_overview import main


def test_data_overview():
    # Ensure report creation succeeds for Fujita example model
    with TemporaryDirectory() as temp_dir:
        outfile = Path(temp_dir) / 'Fujita.html'
        main(outdir=temp_dir)
        assert outfile.is_file()

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt

# Avoid errors when plotting without X server
plt.switch_backend('agg')


def test_cli():
    example_dir = Path(__file__).parent.parent / "doc" / "example"
    fujita_dir = example_dir / "example_Fujita"

    with TemporaryDirectory() as temp_dir:
        args = [
            "petab_visualize",
            "-y", str(fujita_dir / "Fujita.yaml"),
            "-s", str(fujita_dir / "Fujita_simulatedData.tsv"),
            "-o", temp_dir
        ]
        subprocess.run(args, check=True)

import tempfile
from pathlib import Path

from petab.v2.core import ObservablesTable


def test_observables_table():
    file = (
        Path(__file__).parents[2]
        / "doc/example/example_Fujita/Fujita_observables.tsv"
    )

    # read-write-read round trip
    observables = ObservablesTable.from_tsv(file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "observables.tsv"
        observables.to_tsv(tmp_file)
        observables2 = ObservablesTable.from_tsv(tmp_file)
        assert observables == observables2

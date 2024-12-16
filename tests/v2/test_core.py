import tempfile
from pathlib import Path

from petab.v2.core import ConditionsTable, ObservablesTable
from petab.v2.petab1to2 import petab1to2

example_dir_fujita = Path(__file__).parents[2] / "doc/example/example_Fujita"


def test_observables_table():
    file = example_dir_fujita / "Fujita_observables.tsv"

    # read-write-read round trip
    observables = ObservablesTable.from_tsv(file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / "observables.tsv"
        observables.to_tsv(tmp_file)
        observables2 = ObservablesTable.from_tsv(tmp_file)
        assert observables == observables2


def test_conditions_table():
    with tempfile.TemporaryDirectory() as tmp_dir:
        petab1to2(example_dir_fujita / "Fujita.yaml", tmp_dir)
        file = Path(tmp_dir, "Fujita_experimentalCondition.tsv")
        # read-write-read round trip
        conditions = ConditionsTable.from_tsv(file)
        tmp_file = Path(tmp_dir) / "conditions.tsv"
        conditions.to_tsv(tmp_file)
        conditions2 = ConditionsTable.from_tsv(tmp_file)
        assert conditions == conditions2

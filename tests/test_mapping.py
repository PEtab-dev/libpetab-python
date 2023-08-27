"""Tests related to petab.mapping"""
import tempfile

import pandas as pd
import pytest

from petab.C import *  # noqa: F403
from petab.mapping import *


def test_get_mapping_df():
    """Test parameters.get_mapping_df."""
    # Missing columns
    mapping_df = pd.DataFrame(
        data={
            PETAB_ENTITY_ID: ["e1"],
        }
    )

    with pytest.raises(KeyError):
        get_mapping_df(mapping_df)

    # check index is correct
    mapping_df = pd.DataFrame(
        data={
            PETAB_ENTITY_ID: ["e1"],
            MODEL_ENTITY_ID: ["m1"],
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fh:
        file_name = fh.name
        write_mapping_df(mapping_df, file_name)

    assert get_mapping_df(file_name).index == ["e1"]

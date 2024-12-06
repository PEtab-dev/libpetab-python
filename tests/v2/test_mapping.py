"""Tests related to petab.v2.mapping"""
import tempfile

import pandas as pd
import pytest

from petab.v2 import get_mapping_df, write_mapping_df
from petab.v2.C import *  # noqa: F403


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

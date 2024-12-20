"""Tests related to petab.versions"""

from petab.versions import *


def test_parse_version():
    assert parse_version("1.2.3") == (1, 2, 3, "")
    assert parse_version("1.2.3a") == (1, 2, 3, "a")
    assert parse_version("1.2") == (1, 2, 0, "")
    assert parse_version("1") == (1, 0, 0, "")
    assert parse_version(1) == (1, 0, 0, "")
    assert parse_version("1.2.3.a") == (1, 2, 3, ".a")
    assert parse_version("1.2.3.4") == (1, 2, 3, ".4")

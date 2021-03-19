![CI tests](https://github.com/PEtab-dev/libpetab-python/workflows/CI%20tests/badge.svg)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/fd7dd5cee68e449983be5c43f230c7f3)](https://www.codacy.com/gh/PEtab-dev/libpetab-python)
[![codecov](https://codecov.io/gh/PEtab-dev/libpetab-python/branch/master/graph/badge.svg)](https://codecov.io/gh/PEtab-dev/libpetab-python)
[![PyPI version](https://badge.fury.io/py/petab.svg)](https://badge.fury.io/py/petab)

# petab - a Python package for handling PEtab files

[PEtab](https://petab.readthedocs.io/) is a data format for specifying
parameter estimation problems in systems biology. This repository provides
the `petab` Python package for reading, writing and validating PEtab files.

## Documentation

Documentation of the `petab` Python package is available at
[https://libpetab-python.readthedocs.io/en/latest/](https://libpetab-python.readthedocs.io/en/latest/).
Documentation of the PEtab format in general is available at
[https://petab.readthedocs.io/en/latest/](https://petab.readthedocs.io/en/latest/).

## Installation

The PEtab library is available on [pypi](https://pypi.org/project/petab/)
and the easiest way to install it is running

    pip3 install petab
    
It will require Python>=3.7.1 to run.

Development versions of the PEtab library can be installed using

    pip3 install https://github.com/PEtab-dev/libpetab-python/archive/develop.zip

(replace `develop` by the branch or commit you would like to install).

When setting up a new parameter estimation problem, the most useful tools will
be:

  - The **PEtab validator**, which is now automatically installed using Python
    entrypoints to be available as a shell command from anywhere called
    `petablint`

  - `petab.create_parameter_df` to create the parameter table, once you
    have set up the model, condition table, observable table and measurement
    table

  - `petab.create_combine_archive` to create a
    [COMBINE Archive](https://combinearchive.org/index/) from PEtab files

## Examples

Examples for PEtab Python library usage:

* [Validation](https://github.com/PEtab-dev/libpetab-python/blob/master/doc/example/example_petablint.ipynb)
* [Visualization](https://github.com/PEtab-dev/libpetab-python/blob/master/doc/example/example_visualization.ipynb)


## Getting help

If you have any question or problems with this package, feel free to post them
at our GitHub [issue tracker](https://github.com/PEtab-dev/libpetab-python/issues/).

## Contributing

Contributions and feedback to this package are very welcome, see our
[contribution guide](CONTRIBUTING.md).

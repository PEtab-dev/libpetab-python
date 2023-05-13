[![CI](https://github.com/PEtab-dev/libpetab-python/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/PEtab-dev/libpetab-python/actions/workflows/ci_tests.yml)
[![codecov](https://codecov.io/gh/PEtab-dev/libpetab-python/branch/main/graph/badge.svg)](https://codecov.io/gh/PEtab-dev/libpetab-python)
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
    
It will require Python>=3.9 to run. (We are following the
[numpy Python support policy](https://numpy.org/neps/nep-0029-deprecation_policy.html)).

Development versions of the PEtab library can be installed using

    pip3 install https://github.com/PEtab-dev/libpetab-python/archive/develop.zip

(replace `develop` by the branch or commit you would like to install).

When setting up a new parameter estimation problem, the most useful tools will
be:

  - The [PEtab validator](https://petab.readthedocs.io/projects/libpetab-python/en/latest/example/example_petablint.html),
    which is automatically installed using Python
    entrypoints to be available as a shell command from anywhere, called
    `petablint`

  - [`petab.create_parameter_df`](https://petab.readthedocs.io/projects/libpetab-python/en/latest/build/_autosummary/petab.parameters.html#petab.parameters.create_parameter_df)
    to create the parameter table, once you have set up the model, 
    condition table, observable table and measurement table

  - [`petab.create_combine_archive`](https://petab.readthedocs.io/projects/libpetab-python/en/latest/build/_autosummary/petab.core.html#petab.core.create_combine_archive)
    to create a [COMBINE Archive](https://combinearchive.org/index/) from PEtab
    files

## Examples

Examples for PEtab Python library usage:

* [Validation](https://github.com/PEtab-dev/libpetab-python/blob/main/doc/example/example_petablint.ipynb)
* [Visualization](https://github.com/PEtab-dev/libpetab-python/blob/main/doc/example/example_visualization.ipynb)


## Getting help

If you have any question or problems with this package, feel free to post them
at our GitHub [issue tracker](https://github.com/PEtab-dev/libpetab-python/issues/).

## Contributing

Contributions and feedback to this package are very welcome, see our
[contribution guide](CONTRIBUTING.md).

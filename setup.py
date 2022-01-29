from setuptools import setup, find_packages
import sys
import os
import re


def read(fname):
    """Read a file."""
    return open(fname).read()


def absolute_links(txt):
    """Replace relative petab github links by absolute links."""

    raw_base = "(https://raw.githubusercontent.com/"\
               "petab-dev/libpetab-python/master/"
    embedded_base = "(https://github.com/petab-dev/libpetab-python/"\
                    "tree/master/"
    # iterate over links
    for var in re.findall(r'\[.*?\]\((?!http).*?\)', txt):
        if re.match(r'.*?.(png|svg)\)', var):
            # link to raw file
            rep = var.replace("(", raw_base)
        else:
            # link to github embedded file
            rep = var.replace("(", embedded_base)
        txt = txt.replace(var, rep)
    return txt


# Python version check. We need >= 3.6 due to e.g. f-strings
if sys.version_info < (3, 7, 1):
    sys.exit('PEtab requires at least Python version 3.7.1')

# read version from file
__version__ = ''
version_file = os.path.join('petab', 'version.py')
# sets __version__
exec(read(version_file))  # pylint: disable=W0122 # nosec

ENTRY_POINTS = {
    'console_scripts': [
        'petablint = petab.petablint:main',
        'petab_visualize = petab.visualize.cli:_petab_visualize_main',
    ]
}

# project metadata
# noinspection PyUnresolvedReferences
setup(name='petab',
      version=__version__,
      description='Parameter estimation tabular data',
      long_description=absolute_links(read('README.md')),
      long_description_content_type="text/markdown",
      author='The PEtab developers',
      author_email='daniel.weindl@helmholtz-muenchen.de',
      url='https://github.com/PEtab-dev/libpetab-python',
      packages=find_packages(exclude=['doc*', 'test*']),
      install_requires=['numpy>=1.15.1',
                        'pandas>=1.2.0',
                        'matplotlib>=3.5.0',
                        'python-libsbml>=5.17.0',
                        'sympy',
                        'colorama',
                        'seaborn',
                        'pyyaml',
                        'jsonschema',
                        ],
      include_package_data=True,
      tests_require=['flake8', 'pytest', 'python-libcombine'],
      python_requires='>=3.7.1',
      entry_points=ENTRY_POINTS,
      extras_require={
          'reports': ['Jinja2'],
          'combine': ['python-libcombine>=0.2.6'],
          'doc': [
              'sphinx>=3.5.3',
              'sphinxcontrib-napoleon>=0.7',
              'sphinx-markdown-tables>=0.0.15',
              'sphinx-rtd-theme>=0.5.1',
              'myst-parser>=0.16.1',
              'nbsphinx>=0.8.2',
              'm2r>=0.2.1',
              'ipython>=7.21.0',
          ]
      }
      )

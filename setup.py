import os
import re

from setuptools import setup


def read(fname):
    """Read a file."""
    return open(fname).read()


def absolute_links(txt):
    """Replace relative petab github links by absolute links."""
    raw_base = (
        "(https://raw.githubusercontent.com/petab-dev/libpetab-python/master/"
    )
    embedded_base = (
        "(https://github.com/petab-dev/libpetab-python/tree/master/"
    )
    # iterate over links
    for var in re.findall(r"\[.*?\]\((?!http).*?\)", txt):
        if re.match(r".*?.(png|svg)\)", var):
            # link to raw file
            rep = var.replace("(", raw_base)
        else:
            # link to github embedded file
            rep = var.replace("(", embedded_base)
        txt = txt.replace(var, rep)
    return txt


# read version from file
__version__ = ""
version_file = os.path.join("petab", "version.py")
# sets __version__
exec(read(version_file))  # pylint: disable=W0122 # nosec # noqa: S102

# project metadata
# noinspection PyUnresolvedReferences
setup(
    long_description=absolute_links(read("README.md")),
    long_description_content_type="text/markdown",
    version=__version__,
)

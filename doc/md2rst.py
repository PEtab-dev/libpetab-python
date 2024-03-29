import re

import m2r2


def read(fname):
    """Read a file."""
    return open(fname).read()


def absolute_links(txt):
    """Replace relative petab github links by absolute links."""
    repo = "petab-dev/libpetab-python"
    raw_base = f"(https://raw.githubusercontent.com/{repo}/master/"
    embedded_base = f"(https://github.com/{repo}/tree/master/"
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


def md2rst(source: str, target: str):
    txt = absolute_links(read(source))
    txt = m2r2.convert(txt)
    with open(target, "w") as f:
        f.write(txt)


if __name__ == "__main__":
    # parse readme
    md2rst("../README.md", "_static/README.rst")

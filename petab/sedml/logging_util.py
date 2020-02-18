import logging


LOGGER_NAME = "JWS_IMPORTER"


def log_to_console(level=None):
    """
    Log to console.
    Parameters
    ----------
    level: int
        The output level to use. Default: logging.DEBUG.
    """
    if level is None:
        level = logging.DEBUG

    logger = logging.getLogger(LOGGER_NAME)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)


def log_to_file(level=None, filename=None):
    """
    Log to file.
    Parameters
    ----------
    level: int
        The output level to use. Default: logging.DEBUG.
    filename: str
        The name of the file to append to.
        Default: .pypesto_logging.log.
    """

    if level is None:
        level = logging.DEBUG

    if filename is None:
        filename = ".logging.log"

    logger = logging.getLogger(LOGGER_NAME)
    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    logger.addHandler(fh)

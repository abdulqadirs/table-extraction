import logging
from pathlib import Path

LOG_LEVEL = logging.INFO
logger = logging.getLogger("table-extraction")

def setup_logging(logfile = None):
    """
    Sets up logging to stout and to the given file
    Args:
        logfile (Path): The path of logging file
        level: The level of logging(logging.info)
    
    Raises:
        InvalidPath: An error if the given file path is invalid.
    """
    global logger
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s', "%H:%M:%S")
    logger.setLevel(logging.INFO)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)

    if logfile:
        try:
            filehandler = logging.FileHandler(logfile)
        except FileNotFoundError as e:
            logger.warning(e)
        else:
            filehandler.setLevel(logging.INFO)
            filehandler.setFormatter(formatter)
            logger.addHandler(filehandler)
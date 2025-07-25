import logging
import os
import sys
import traceback
from pathlib import Path


class ErrorFormatter(logging.Formatter):
    """Custom formatter that adds file and line info for ERROR and higher levels"""

    def format(self, record):
        # Use different formats based on log level
        if record.levelno >= logging.ERROR:
            self._style._fmt = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        else:
            self._style._fmt = '%(asctime)s - %(levelname)s - %(message)s'
        return super().format(record)


def setup_logger(name='pyMAOS', log_file=None, level=logging.INFO):
    """
    Set up logger to output to both console and file with enhanced error formatting

    Parameters
    ----------
    name : str
        Name of the logger
    log_file : str, optional
        Path to log file. If None, logging will only be to console.
    level : int
        Logging level (default: logging.INFO)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatter
    formatter = ErrorFormatter()

    # Create console handler and set level
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Create file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        os.makedirs(log_path.parent, exist_ok=True)  # Create directory if needed
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_exception(logger, exc_info=None, message="An exception occurred"):
    """
    Log an exception with traceback showing file and line numbers

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    exc_info : tuple, optional
        Exception info from sys.exc_info(). If None, current exception is used.
    message : str, optional
        Custom message to prepend to the exception details
    """
    if exc_info is None:
        exc_info = sys.exc_info()

    exc_type, exc_value, exc_tb = exc_info
    tb_details = traceback.extract_tb(exc_tb)

    if tb_details:
        last_frame = tb_details[-1]
        file_name = os.path.basename(last_frame.filename)
        line_no = last_frame.lineno
        logger.error(f"{message}: {exc_type.__name__}: {exc_value} [in {file_name}:{line_no}]")

        # Log the full traceback at debug level
        tb_formatted = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.debug(f"Full traceback:\n{tb_formatted}")
    else:
        logger.error(f"{message}: {exc_type.__name__}: {exc_value}")


# Default logger instance
default_logger = setup_logger()
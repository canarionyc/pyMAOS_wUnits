import logging
import os
import sys
import traceback


class ErrorFormatter(logging.Formatter):
    """Custom formatter that adds file and line info for ERROR and higher levels"""

    def format(self, record):
        # Use different formats based on log level
        if record.levelno >= logging.ERROR:
            self._style._fmt = '%(asctime)s - %(levelname)s - %(message)s'
        else:
            self._style._fmt = '%(asctime)s - %(levelname)s - %(message)s'
        return super().format(record)


def setup_logger(name='pyMAOS', log_file=None, level=logging.INFO):
    """Set up logger to output to both console and file with enhanced error formatting"""
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
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Create directory if needed
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_exception(logger, exc_info=None, message="An exception occurred"):
    """Log an exception with traceback showing file and line numbers"""
    if exc_info is None:
        exc_info = sys.exc_info()

    exc_type, exc_value, exc_tb = exc_info
    tb_details = traceback.extract_tb(exc_tb)

    if tb_details:
        # Get the frame where the actual exception occurred
        error_frame = tb_details[-1]
        file_name = os.path.basename(error_frame.filename)
        line_no = error_frame.lineno
        function = error_frame.name

        # Include the error location in the message
        error_msg = f"{message}: {exc_type.__name__} in {file_name}:{line_no} (function: {function}): {exc_value}"
        logger.error(error_msg)

        # Print the entire call stack with clickable file links
        print(f"DEBUG - Full call stack for exception:")
        for i, frame in enumerate(tb_details):
            frame_file = os.path.basename(frame.filename)
            print(f"  Frame {i}:\n{frame_file}:{frame.lineno} in {frame.name}()")
            # Put clickable link at beginning of line
            print(f"{frame.filename}:{frame.lineno}")
            if frame.line:
                print(f"    Code: {frame.line.strip()}")

        # Log the full traceback at debug level
        tb_formatted = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.debug(f"Full traceback:\n{tb_formatted}")
    else:
        logger.error(f"{message}: {exc_type.__name__}: {exc_value}")


# Default logger instance
default_logger = setup_logger()
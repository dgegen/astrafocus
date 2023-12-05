import logging
import time

def configure_logger():
    logging.basicConfig(
        format='%(levelname)s,%(asctime)s.%(msecs)03d,%(process)d,%(name)s,(%(filename)s:%(lineno)d),%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    logging.Formatter.converter = time.gmtime

    return logging.getLogger(__name__)


def configure_logger(
    log_file: str = "main.log",
    stream_handler_level: int = logging.INFO,
    file_handler_level: int = logging.DEBUG,
    run_from_ipython: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with console and optionally file handlers.

    Parameters
    ----------
    log_file : str, optional
        The name of the log file. Defaults to 'main.log'.
    stream_handler_level : int, optional
        Log level for the console (stream) handler. Defaults to logging.INFO.
    file_handler_level : int, optional
        Log level for the file handler. Defaults to logging.DEBUG.
    run_from_ipython : bool, optional
        Flag indicating if the script is run from IPython environment. Defaults to True.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Notes
    -----
    This function creates a logger with a console (stream) handler that formats log messages
    with a simple format and an optional file handler that uses a more detailed format.

    The console handler is set to use the CustomFormatter class for color-coded output.
    If the logger has no handlers, a console handler is added to it. If not running from an IPython
    environment and the logger has only one handler (assumed to be the console handler), a file handler
    is added to the logger for writing logs to the specified log file.

    Example
    -------
    >>> logger = setup_logger(log_file='mylog.log', stream_handler_level=logging.INFO)
    >>> logger.info('This is an informational message.')
    >>> logger.debug('This is a debug message.')

    """
    # Create a logger
    logger = logging.getLogger(__name__)  # Gets python log_name
    logger.setLevel(logging.INFO)

    # Create a formatter

    stream_formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)-8s :: %(message)s", datefmt="%H:%M:%S"
    )
    log_file_formatter = logging.Formatter(
        "%(asctime)s :: %(name)-8s, %(levelname)s , %(module)s:%(lineno)d :: %(message)s"
    )

    # Create a console handler
    if len(logger.handlers) == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_handler_level)
        # stream_handler.setFormatter(stream_formatter)
        stream_handler.setFormatter(stream_formatter)
        stream_handler.setFormatter(CustomFormatter())
        logger.addHandler(stream_handler)

    # Create a file handler
    if not run_from_ipython and len(logger.handlers) == 1:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_handler_level)
        file_handler.setFormatter(log_file_formatter)
        logger.addHandler(file_handler)

    return logger


class CustomFormatter(logging.Formatter):
    """
    A custom logging formatter that allows customizable formatting and color-coded output.

    Parameters
    ----------
    fmt : str, optional
        The log message format. Defaults to '%(asctime)s :: %(levelname)-8s :: %(message)s'.
    datefmt : str, optional
        The date format for log timestamps. Defaults to '%H:%M:%S'.

    Attributes
    ----------
    grey : str
        ANSI escape code for grey text.
    yellow : str
        ANSI escape code for yellow text.
    red : str
        ANSI escape code for red text.
    bold_red : str
        ANSI escape code for bold red text.
    reset : str
        ANSI escape code to reset text formatting.

    Methods
    -------
    format(record)
        Format the log record according to the specified log level's formatting.

    Usage
    -----
    formatter = CustomFormatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    Note
    ----
    The ANSI escape codes are used to colorize the output text in supported terminals.
    """

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(
        self,
        fmt=None,
        datefmt=None,
    ) -> None:
        """
        Initialize the CustomFormatter instance.

        Parameters
        ----------
        fmt : str, optional
            The log message format. Defaults to '%(asctime)s :: %(levelname)-8s :: %(message)s'.
        datefmt : str, optional
            The date format for log timestamps. Defaults to '%H:%M:%S'.
        """
        if fmt is None:
            fmt = "%(asctime)s :: %(levelname)-8s :: %(message)s"
        if datefmt is None:
            datefmt = "%H:%M:%S"
        self.custom_format = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.custom_format + self.reset,
            logging.INFO: self.grey + self.custom_format + self.reset,
            logging.WARNING: self.yellow + self.custom_format + self.reset,
            logging.ERROR: self.red + self.custom_format + self.reset,
            logging.CRITICAL: self.bold_red + self.custom_format + self.reset,
        }
        super().__init__(fmt, datefmt)

    def format(self, record):
        """Format the log record according to the specified log level's formatting."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)


if __name__ == "__main__":
    # Call the setup_logger function to set up the logger
    logger = configure_logger()

    # Start logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # logger.exception("...") # to get traceback of an error.

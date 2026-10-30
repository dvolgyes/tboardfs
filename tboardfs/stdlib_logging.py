"""Send the records of the standard logging module to loguru."""

import logging

from loguru import logger

from tboardfs.constants import LOG_LEVELS


class StdlibToLoguru(logging.Handler):
    """Write each standard logging record to loguru.

    mfusepy logs a failed FUSE operation, with its traceback, through the
    standard logging module; this puts it in the same outputs as our logs.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write one record with its level and traceback."""
        level = record.levelname if record.levelname in LOG_LEVELS else record.levelno
        logger.opt(exception=record.exc_info).log(level, record.getMessage())

    @staticmethod
    def install() -> None:
        """Send all standard logging records to loguru, which filters the levels."""
        logging.basicConfig(
            handlers=[StdlibToLoguru()], level=logging.NOTSET, force=True
        )

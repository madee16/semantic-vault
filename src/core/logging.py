"""
Structured logging configuration for SemanticVault.
"""
import logging
import sys


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return the root application logger."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("semanticvault")
    logger.setLevel(log_level)
    return logger


logger = setup_logging()

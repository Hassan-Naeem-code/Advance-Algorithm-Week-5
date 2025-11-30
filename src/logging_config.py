"""Centralized logging configuration for the project.
Call `configure_logging()` near process entrypoint.
"""
import logging
from logging import StreamHandler


def configure_logging(level: int = logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level)
    handler = StreamHandler()
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)


__all__ = ["configure_logging"]

import logging
from typing import TYPE_CHECKING

from agentlib.core import environment

if TYPE_CHECKING:
    from agentlib import Environment


class CustomLogger(logging.Logger):
    """Subclass of Logger that adds the env_time to the record, allowing it to print
    the current time."""

    def __init__(self, name, env: "Environment", level=logging.NOTSET):
        super().__init__(name, level)
        self.env = env

    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        record = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, extra, sinfo
        )
        _until = self.env.pretty_until()
        _time = self.env.pretty_time()
        if _until is environment.UNTIL_UNSET:
            # Add "INIT" prefix to clearly indicate initialization phase
            record.env_time = f"<INIT>"
        elif _until is None:
            record.env_time = _time
        else:
            record.env_time = _time + "/" + _until
        return record


def create_logger(env: "Environment", name: str) -> CustomLogger:
    """Creates a logger that displays the environment time when logging."""
    # Create a custom logger
    custom_logger = CustomLogger(name, env=env)
    custom_logger.setLevel(logging.root.getEffectiveLevel())

    # Create a formatter
    formatter = logging.Formatter("%(env_time)s %(levelname)s: %(name)s: %(message)s")

    # Create a StreamHandler and add it to the logger
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    custom_logger.addHandler(stream_handler)

    # Check if root logger has any FileHandlers and add similar ones to our custom logger
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            # Create a similar FileHandler for our custom logger
            file_handler = logging.FileHandler(
                handler.baseFilename, mode=handler.mode, encoding=handler.encoding
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(handler.level)
            custom_logger.addHandler(file_handler)

    return custom_logger

import logging
from typing import TYPE_CHECKING

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
        record.env_time = self.env.pretty_time()
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
    return custom_logger

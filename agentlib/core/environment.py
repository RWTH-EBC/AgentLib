"""This module contains the Environment class, used by all Agents and Modules."""
import json
import logging
import time
from datetime import datetime
from typing import Union, Any, Optional
import simpy
from pathlib import Path
from simpy.core import SimTime, Event
from pydantic import (
    ConfigDict,
    PositiveFloat,
    BaseModel,
    Field,
)

logger = logging.getLogger(name=__name__)


class EnvironmentConfig(BaseModel):
    """Config for the Environment"""

    rt: bool = False
    factor: PositiveFloat = 1.0
    strict: bool = False
    t_sample: PositiveFloat = Field(
        title="t_sample",
        default=1,
        description="Used to increase the now-time of"
        "the environment using the clock function.",
    )
    offset: float = Field(
        title="offset",
        default=0,
        description="Used to offset the now-time of"
        "the environment using the clock function.",
    )
    clock: bool = False
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"
    )


class Environment:
    """TODO"""

    def __new__(
        cls, *args, **kwargs
    ) -> Union["RealtimeEnvironment", "InstantEnvironment"]:
        config = make_env_config(kwargs["config"])
        if config.rt:
            return RealtimeEnvironment(config=config)
        else:
            return InstantEnvironment(config=config)


def make_env_config(
    config: Union[dict, EnvironmentConfig, str, None]
) -> EnvironmentConfig:
    if config is None:
        return EnvironmentConfig()
    if isinstance(config, EnvironmentConfig):
        return config
    elif isinstance(config, (str, Path)):
        if Path(config).exists():
            with open(config, "r") as f:
                config = json.load(f)
    return EnvironmentConfig.model_validate(config)


class CustomSimpyEnvironment(simpy.Environment):

    _config: EnvironmentConfig

    @property
    def config(self) -> EnvironmentConfig:
        """Return the config of the environment"""
        return self._config

    @property
    def time(self) -> float:
        """Get the current time of the environment.
        If RT is enabled, the unix-time is returned."""
        return self.now

    def clock(self):
        """Define a clock loop to increase the now-timer every other milisecond
        (Or whatever t_sample is)"""
        # if log level is not info or debug, this process can terminate
        while True:
            logger.info("Current simulation time: %s", self.pretty_time())
            yield self.timeout(self.config.t_sample)

    def pretty_time(self):
        ...


class InstantEnvironment(CustomSimpyEnvironment):
    """
    Custom environment to meet the needs
    of the agentlib.
    """

    def __init__(self, *, config: EnvironmentConfig):
        super().__init__(initial_time=config.offset)
        self._config = config
        self.process(self.clock())

    def pretty_time(self) -> str:
        """Returns the time in a nice format. Datetime if realtime, seconds if not
        realtime. Implemented as rt_time or sim_time"""
        return f"{self.now:.2f}s"


class RealtimeEnvironment(simpy.RealtimeEnvironment, CustomSimpyEnvironment):
    """
    Custom environment to meet the needs
    of the agentlib.
    """

    def __init__(self, *, config: EnvironmentConfig):
        super().__init__(
            initial_time=config.offset, factor=config.factor, strict=config.strict
        )
        self._config = config
        self.process(self.clock())

    def run(self, until: Optional[Union[SimTime, Event]] = None) -> Optional[Any]:
        self._t_start = time.time()
        self.sync()
        return super().run(until=until)

    @property
    def time(self) -> float:
        """Get the current time of the environment.
        If RT is enabled, the unix-time is returned."""
        try:
            return self.now + self._t_start
        except AttributeError:
            raise RuntimeError(
                "Environment time was accessed before start of the environment. Make "
                "sure your modules do not access the environment time during __init__. "
            )

    def pretty_time(self) -> str:
        """Returns the time in a nice format. Datetime if realtime, seconds if not
        realtime. Implemented as rt_time or sim_time"""
        return datetime.fromtimestamp(self.now).strftime("%d-%b-%Y %H:%M:%S")

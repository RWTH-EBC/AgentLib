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
    field_validator,
    ConfigDict,
    PositiveFloat,
    BaseModel,
    FilePath,
    Field,
)

logger = logging.getLogger(name=__name__)


class EnvironmentConfig(BaseModel):
    """Config for the Environment"""

    rt: bool = False
    factor: PositiveFloat = 1.0
    strict: bool = False
    initial_time: Union[PositiveFloat, float] = 0
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

    @field_validator("initial_time")
    @classmethod
    def check_time(cls, initial_time):
        """Check if initial time is correct"""
        if initial_time >= 0:
            return initial_time
        raise ValueError("Time has to be greater than 0.")


class Environment(simpy.RealtimeEnvironment):
    """
    Custom environment to meet the needs
    of the agentlib.
    """

    def __init__(self, *, config: dict = None):
        # pylint: disable=super-init-not-called
        self.t_start = None
        if config is None:
            self.config = EnvironmentConfig()
        else:
            self.config = config

    @property
    def offset(self):
        """Offset of the now-time of the environment"""
        return self.config.offset

    @offset.setter
    def offset(self, offset):
        """Set the offset of the environment"""
        self.config.offset = offset

    @property
    def config(self) -> EnvironmentConfig:
        """Return the config of the environment"""
        return self._config

    @config.setter
    def config(self, config: Union[EnvironmentConfig, dict, str, FilePath]):
        """Set the config/settings of the environment"""
        if isinstance(config, EnvironmentConfig):
            self._config = config
        elif isinstance(config, (str, Path)):
            if Path(config).exists():
                with open(config, "r") as f:
                    config = json.load(f)
        self._config = EnvironmentConfig.model_validate(config)
        self._create_env()

    def _create_env(self):
        """
        Sets the runtime environment as discrete event dispatcher. We Use the
        simpy package for this:
        https://simpy.readthedocs.io/en/latest/contents.html

        The environment is initialized according to configuration. If an
        existing one is parsed it will only check if the local configuration
        aligns with external configuration.

        Args:
            env(simpy.Environment): Environment where the local processes
            should be added to

        Returns:

        """
        if self.config.rt:
            logger.info("Initializing real time runtime environment...")
            simpy.RealtimeEnvironment.__init__(
                self,
                factor=self.config.factor,
                initial_time=self.config.initial_time,
                strict=self.config.strict,
            )
            self.pretty_time = self.rt_time
        else:
            logger.info("Initializing runtime environment...")
            simpy.Environment.__init__(
                self, **self.config.model_dump(include={"initial_time"})
            )
            self.pretty_time = self.sim_time
        # regularily print sim time, if clock is on, and log level is sufficient
        if self.config.clock and logger.level <= logging.INFO:
            self.process(self.clock())

    def step(self) -> None:
        if self.config.rt:
            simpy.RealtimeEnvironment.step(self)
        else:
            simpy.Environment.step(self)

    def run(self, until: Optional[Union[SimTime, Event]] = None) -> Optional[Any]:
        self.t_start = time.time()
        if isinstance(self, simpy.RealtimeEnvironment):
            self.sync()
        return super().run(until=until)

    @property
    def time(self) -> float:
        """Get the current time of the environment.
        If RT is enabled, the unix-time is returned."""
        if self.config.rt:
            return self.now + self.t_start + self.offset
        # Else return
        return self.now + self.offset

    def clock(self):
        """Define a clock loop to increase the now-timer every other milisecond
        (Or whatever t_sample is)"""
        # if log level is not info or debug, this process can terminate
        while True:
            logger.info("Current simulation time: %s", self.pretty_time())
            yield self.timeout(self.config.t_sample)

    def pretty_time(self) -> str:
        """Returns the time in a nice format. Datetime if realtime, seconds if not
        realtime. Implemented as rt_time or sim_time"""
        if self.config.rt:
            return self.rt_time()
        return self.sim_time()

    def rt_time(self) -> str:
        """Returns the current real time in datetime format, based on unix timestamp."""
        try:
            return datetime.fromtimestamp(self.time).strftime("%d-%b-%Y %H:%M:%S")
        except TypeError:
            # when the rt environment has not yet started, we cannot have a time
            return ""

    def sim_time(self) -> str:
        """Returns the current simulation time in seconds, with two decimals of
        precision."""
        return f"{self.time:.2f}s"

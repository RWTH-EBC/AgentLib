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
        default=60,
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
    """Simpy Environment Distributor. Handles synchronous processes."""

    def __new__(
        cls, *args, **kwargs
    ) -> Union["RealtimeEnvironment", "InstantEnvironment"]:
        config = make_env_config(kwargs.get("config"))
        if not config.rt:
            return InstantEnvironment(config=config)
        if config.factor == 1:
            return RealtimeEnvironment(config=config)
        return ScaledRealtimeEnvironment(config=config)


def make_env_config(
    config: Union[dict, EnvironmentConfig, str, None]
) -> EnvironmentConfig:
    """Creates the environment config from different sources."""
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
    """A customized version of the simpy environment. Handles execution of modules
    processes and manages time for instant execution mode."""

    _config: EnvironmentConfig

    @property
    def config(self) -> EnvironmentConfig:
        """Return the config of the environment"""
        return self._config

    @property
    def offset(self) -> EnvironmentConfig:
        """Start time offset of the environment."""
        return self.config.offset

    @property
    def time(self) -> float:
        """Get the current time of the environment.
        If RT is enabled, the unix-time is returned."""
        return self.now

    def clock(self):
        """Define a clock loop to increase the now-timer every other second
        (Or whatever t_sample is)"""
        # if log level is not info or debug, this process can terminate
        while True:
            logger.info("Current simulation time: %s", self.pretty_time())
            yield self.timeout(self.config.t_sample)

    def pretty_time(self):
        ...


class InstantEnvironment(CustomSimpyEnvironment):
    """A customized version of the simpy environment. Handles execution of modules
    processes and manages time for instant execution mode."""

    def __init__(self, *, config: EnvironmentConfig):
        super().__init__(initial_time=config.offset)
        self._config = config
        if self.config.clock:
            self.process(self.clock())

    def pretty_time(self) -> str:
        """Returns the time in a nice format. Datetime if realtime, seconds if not
        realtime. Implemented as rt_time or sim_time"""
        return f"{self.time:.2f}s"


class RealtimeEnvironment(simpy.RealtimeEnvironment, CustomSimpyEnvironment):
    """A customized version of the simpy environment. Handles execution of modules
    processes and manages time for real time execution mode."""

    def __init__(self, *, config: EnvironmentConfig):
        super().__init__(
            initial_time=config.offset, factor=config.factor, strict=config.strict
        )
        self._config = config
        if self.config.clock:
            self.process(self.clock())
        else:
            self.process(self.silent_clock())

    def run(self, until: Optional[Union[SimTime, Event]] = None) -> Optional[Any]:
        self.sync()
        return super().run(until=until)

    @property
    def time(self) -> float:
        """Get the current time of the environment.
        If RT is enabled, the unix-time is returned."""
        return time.time() + self.config.offset

    def pretty_time(self) -> str:
        """Returns the time in a nice format. Datetime if realtime, seconds if not
        realtime. Implemented as rt_time or sim_time"""
        return datetime.fromtimestamp(self.time).strftime("%d-%b-%Y %H:%M:%S")

    def silent_clock(self):
        """A silent clock, which does not log anything."""
        # if log level is not info or debug, this process can terminate
        while True:
            yield self.timeout(self.config.t_sample)


class ScaledRealtimeEnvironment(simpy.RealtimeEnvironment, CustomSimpyEnvironment):
    """A customized version of the simpy environment. Handles execution of modules
    processes and manages time for scaled real time execution mode."""

    def __init__(self, *, config: EnvironmentConfig):
        super().__init__(
            initial_time=config.offset, factor=config.factor, strict=config.strict
        )
        self._config = config
        if self.config.clock:
            self.process(self.clock())
        else:
            self.process(self.silent_clock())

    def run(self, until: Optional[Union[SimTime, Event]] = None) -> Optional[Any]:
        self.sync()
        return super().run(until=until)

    @property
    def time(self) -> float:
        """Get the current time of the environment.
        If RT is enabled, the unix-time is returned."""
        return self.now

    def pretty_time(self) -> str:
        """Returns the time in a nice format. Datetime if realtime, seconds if not
        realtime. Implemented as rt_time or sim_time"""
        return f"{self.time:.2f}s"

    def silent_clock(self):
        """A silent clock, which does not log anything."""
        # if log level is not info or debug, this process can terminate
        while True:
            yield self.timeout(self.config.t_sample)


def monkey_patch_simpy_process():
    """Removes the exception catching in simpy processes. This removes some of simpys
    features that we do not need. In return, it improves debugging and makes error
    messages more concise."""

    def _describe_frame(frame) -> str:
        """Print filename, line number and function name of a stack frame."""
        filename, name = frame.f_code.co_filename, frame.f_code.co_name
        lineno = frame.f_lineno

        with open(filename) as f:
            for no, line in enumerate(f):
                if no + 1 == lineno:
                    return (
                        f'  File "{filename}", line {lineno}, in {name}\n'
                        f"    {line.strip()}\n"
                    )
            return f'  File "{filename}", line {lineno}, in {name}\n'

    def new_resume(self, event: Event) -> None:
        """Resumes the execution of the process with the value of *event*. If
        the process generator exits, the process itself will get triggered with
        the return value or the exception of the generator."""
        # Mark the current process as active.
        self.env._active_proc = self

        while True:
            # Get next event from process
            try:
                if event._ok:
                    event = self._generator.send(event._value)
                else:
                    # The process has no choice but to handle the failed event
                    # (or fail itself).
                    event._defused = True

                    # Create an exclusive copy of the exception for this
                    # process to prevent traceback modifications by other
                    # processes.
                    exc = type(event._value)(*event._value.args)
                    exc.__cause__ = event._value
                    event = self._generator.throw(exc)
            except StopIteration as e:
                # Process has terminated.
                event = None  # type: ignore
                self._ok = True
                self._value = e.args[0] if len(e.args) else None
                self.env.schedule(self)
                break

            # Process returned another event to wait upon.
            try:
                # Be optimistic and blindly access the callbacks attribute.
                if event.callbacks is not None:
                    # The event has not yet been triggered. Register callback
                    # to resume the process if that happens.
                    event.callbacks.append(self._resume)
                    break
            except AttributeError:
                # Our optimism didn't work out, figure out what went wrong and
                # inform the user.
                if hasattr(event, "callbacks"):
                    raise

                msg = f'Invalid yield value "{event}"'
                descr = _describe_frame(self._generator.gi_frame)
                raise RuntimeError(f"\n{descr}{msg}") from None

        self._target = event
        self.env._active_proc = None

    simpy.Process._resume = new_resume


monkey_patch_simpy_process()

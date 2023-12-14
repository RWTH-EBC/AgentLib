"""
This modules defines re-use able controller modules,
such as the standard Controller and the SISOController
"""

import abc
import logging
from math import inf
from typing import Generator

from pydantic import field_validator, Field
from pydantic_core.core_schema import FieldValidationInfo

from agentlib.core import BaseModule, Agent, BaseModuleConfig
from agentlib.core.datamodels import AgentVariable

logger = logging.getLogger(__name__)


class Controller(BaseModule):
    """
    Base class for all controller tasks within an agent
    """

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self.step = self.loop_sim()

    @property
    def step(self) -> Generator:
        """Return the generator for the do_step function"""
        return self._step

    @step.setter
    def step(self, step: Generator):
        """Set the generator for the do_step function"""
        self._step = step

    def process(self):
        """ "Only called on run() to initialize the step."""
        # pylint: disable=stop-iteration-return
        next(self.step)
        yield self.env.event()

    def loop_sim(self):
        """Loop over the do_step function"""
        raise NotImplementedError("Needs to be implemented by derived modules")

    @abc.abstractmethod
    def do_step(self, inp_var: AgentVariable):
        """Controller step function. Needs to be a generator function,
        thus using yield instead of return"""
        raise NotImplementedError("Needs to be implemented by derived modules")


class SISOControllerConfig(BaseModuleConfig):
    """Check all inputs of a SISO-Contoller

    Parameters used in all SISO Controllers.
    ub: Upper bound of controller output
    lb: Lower bound of controller output
    reverser: Change of sign.
    """

    input: AgentVariable = AgentVariable(name="u", type="float", value=0)
    output: AgentVariable = AgentVariable(name="y", type="float", value=0)
    ub: float = Field(title="Upper bound", default=inf)
    lb: float = Field(title="Lower bound", default=-inf)
    reverse: bool = Field(title="Change of sign", default=False)

    @field_validator("lb")
    @classmethod
    def check_bounds(cls, lb, info: FieldValidationInfo):
        """Check if upper and lower bound values are correct"""
        assert info.data["ub"] > lb, "Upper limit must be greater than lower limit"
        return lb

    @field_validator("output", "input")
    @classmethod
    def check_value_type(cls, var):
        if var.value is None:
            var.value = 0.0
        return var


class SISOController(Controller):
    """
    Base class for all controller having one single input and one single output
    """

    config: SISOControllerConfig

    @property
    def ub(self):
        """The ub value"""
        return self.config.ub

    @property
    def lb(self):
        """The lb value"""
        return self.config.lb

    @property
    def reverse(self):
        """The reverse value"""
        return self.config.reverse

    def register_callbacks(self):
        """A SISO controller has only one input and only reacts to this input."""
        inp = self.get(self.config.input.name)
        self.agent.data_broker.register_callback(
            alias=inp.alias,
            source=inp.source,
            callback=self._siso_callback,
            name=inp.name,
        )

    def _siso_callback(self, inp: AgentVariable, name: str):
        self.logger.debug("Received input %s=%s", name, inp.value)
        out_val = self.step.send(inp)
        if out_val is None:
            self.logger.error("Output value is None. Won't send it.")
        else:
            out_name = self.config.output.name
            self.logger.debug("Sending output %s=%s", out_name, out_val)
            self.set(name=out_name, value=out_val)

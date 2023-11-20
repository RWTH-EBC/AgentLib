from pydantic import Field
from agentlib.core import Agent
from agentlib.modules.controller import SISOController, SISOControllerConfig
from agentlib.core.datamodels import AgentVariable


class BangBangConfig(SISOControllerConfig):
    """Special config for a BangBang-Controller"""

    gain: float = Field(title="Gain of output", default=1)


class BangBang(SISOController):
    """
    A bang–bang controller (2 step or on–off controller), also known as a
    hysteresis controller, that switches abruptly between two states.
    """

    config: BangBangConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self._last_out_val = self.get(self.config.output.name).value

    @property
    def gain(self):
        """Get the gain of the BangBang controller"""
        return self.config.gain

    @property
    def last_out_val(self):
        """Last output value of the controller"""
        return self._last_out_val

    @last_out_val.setter
    def last_out_val(self, out_val):
        """Set the last output value of the controller"""
        self._last_out_val = out_val

    def loop_sim(self):
        out_val = None
        while True:
            inp = yield out_val
            out_val = self.do_step(inp_var=inp)
            self.last_out_val = out_val
            out_val *= self.gain

    def do_step(self, inp_var: AgentVariable):
        # y = not pre(y) and u > uHigh or pre(y) and u >= uLow
        if inp_var.value <= self.ub and self.last_out_val == int(not self.reverse):
            return int(not self.reverse)
        if inp_var.value > self.ub and self.last_out_val == int(not self.reverse):
            return int(self.reverse)
        if inp_var.value < self.lb and self.last_out_val == int(self.reverse):
            return int(not self.reverse)
        if inp_var.value >= self.lb and self.last_out_val == int(self.reverse):
            return int(self.reverse)

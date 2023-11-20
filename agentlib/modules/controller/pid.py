from math import inf, isclose
from typing import Union

from pydantic import field_validator
from pydantic_core.core_schema import FieldValidationInfo

from agentlib.modules.controller import SISOController, SISOControllerConfig
from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import ConfigurationError


class PIDConfig(SISOControllerConfig):
    """
    Pydantic data model for pid controller configuration parser
    """

    setpoint: Union[AgentVariable, float] = AgentVariable(
        name="setpoint", description="Pid Setpoint", type="float"
    )
    Kp: Union[AgentVariable, float] = AgentVariable(
        name="Kp", description="Proportional gain", type="float", value=1
    )
    Ti: Union[AgentVariable, float] = AgentVariable(
        name="Ti", description="Integration time in s", type="float", value=inf
    )
    Td: Union[AgentVariable, float] = AgentVariable(
        name="Td",
        description="Derivative time in s",
        type="float",
        value=0,
        unit="seconds",
    )

    @field_validator("Kp", "Ti", "Td", "setpoint", mode="before")
    @classmethod
    def convert_to_variable(cls, parameter, info: FieldValidationInfo):
        if isinstance(parameter, (float, int)):
            default = cls.default(info.field_name)
            parameter = default.copy(update={"value": parameter})
        if isinstance(parameter, AgentVariable):
            value = parameter.value
        else:
            value = parameter.get("value")
        if value is None:
            raise ConfigurationError(
                f"PID needs a value for the variable '{parameter.name}'."
            )
        return parameter


class PID(SISOController):
    """
    A proportional–integral–derivative controller (PID
    controller or three-term controller) with anti-wind up.
    It continuously calculates an error value e(t)  as the difference between a
    desired set point and a measured process variable and applies a correction
    based on proportional, integral, and derivative terms
    (denoted P, I, and D respectively)
    +--------------------+---------------+---------------+---------------------+
    | Parameter          | Kp            | Ki=(Kp/Ti)    | Kd=(Kp*Td)          |
    +--------------------+---------------+---------------+---------------------+
    | Rise time          | Decrease      | Decrease      | Minor change        |
    | Overshoot          | Increase      | Increase      | Decrease            |
    | Settling time      | Small change  | Increase      | Decrease            |
    | Steady-state error | Decrease      | Eliminate     | No effect in theory |
    | Stability          | Degrade       | Degrade       | Improve if Kd small |
    +--------------------+---------------+---------------+---------------------+

    Configs:
        setpoint (float): Set point of the controller
        Kp (float): proportional gain
        Ti (float): integration time in s
        Td (float): derivative time in s
        ub (float): high control limit
        lb (float): low control limit
        reverse(boolean): change of sign

    """

    config: PIDConfig

    def __init__(self, *, config, agent):
        self.integral: float = 0
        self.e_last: float = 0
        self.last_time: float = 0
        self.e_der_t: float = 0
        super().__init__(config=config, agent=agent)

    @property
    def setpoint(self):
        """Get the current setpoint value from data_broker or config"""
        return self.get(self.config.setpoint.name).value

    @property
    def Kp(self):
        """Get the current Kp value from data_broker or config"""
        return self.get(self.config.Kp.name).value

    @property
    def Ti(self):
        """Get the current Ti value from data_broker or config"""
        return self.get(self.config.Ti.name).value

    @property
    def Td(self):
        """Get the current Td value from data_broker or config"""
        return self.get(self.config.Td.name).value

    def loop_sim(self):
        self.last_time = self.env.time
        y = None
        while True:
            inp_var = yield y
            y = self.do_step(inp_var)

    def do_step(self, inp_var: AgentVariable):
        u = inp_var.value
        curr_time = inp_var.timestamp

        t_sample = curr_time - self.last_time
        if t_sample <= 0:
            self.logger.error(
                "t_sample is smaller equal zero. %s" " Can't compute integral part.",
                t_sample,
            )
            return

        # calculate control difference
        if self.reverse:
            e = u - self.setpoint
        else:
            e = self.setpoint - u

        # calculate integral
        if self.Ti != 0:
            self.integral += 1 / self.Ti * e * t_sample

        # calculate differential.
        # Assert that t_sample is numerically feasible to use in division
        if isclose(t_sample, 0.0, rel_tol=1e-12, abs_tol=0.0):
            self.logger.error(
                "Sample rate to high! t_sample: %s. "
                "Can't calculate differential part of PID",
                t_sample,
            )
            self.e_der_t = 0
        else:
            self.e_der_t = self.Td * (e - self.e_last) / t_sample

        # PID output
        y = self.Kp * (e + self.integral + self.e_der_t)

        # Limiter
        if y < self.lb:
            y = self.lb
            self.integral = y / self.Kp - e
        elif y > self.ub:
            y = self.ub
            self.integral = y / self.Kp - e

        self.e_last = e
        self.last_time = curr_time  # Set the value for the next iteration
        return y

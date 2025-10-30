"""This module contains the ScipyStateSpaceModel class."""

import logging
from typing import Union

import numpy as np
from pydantic import ValidationError, model_validator

from agentlib.core.errors import OptionalDependencyError

try:
    from scipy import signal
    from scipy import interpolate, integrate
except ImportError as err:
    raise OptionalDependencyError(
        dependency_name="scipy", dependency_install="scipy", used_object="scipy-model"
    ) from err


from agentlib.core import Model, ModelConfig


logger = logging.getLogger(__name__)


class ScipyStateSpaceModelConfig(ModelConfig):
    """Customize config of Model."""

    system: Union[dict, list, tuple, signal.StateSpace]

    @model_validator(mode="before")
    @classmethod
    def check_system(cls, values):
        """Root validator to check if the given system is valid."""
        # pylint: disable=no-self-argument,no-self-use
        system = values.get("system")
        if isinstance(system, (tuple, list)):
            # Check correct input size
            assert (
                len(system) == 4
            ), "State space representation requires exactly 4 matrices"
        elif isinstance(system, dict):
            assert "A" in system, "State space representation requires key 'A'"
            assert "B" in system, "State space representation requires key 'B'"
            assert "C" in system, "State space representation requires key 'C'"
            assert "D" in system, "State space representation requires key 'D'"
            system = [system["A"], system["B"], system["C"], system["D"]]
        elif isinstance(system, signal.ltisys.StateSpaceContinuous):
            return values
        else:
            logger.error(
                "Given system is of type %s but should be list, tuple or dict",
                type(system),
            )
            raise ValidationError
        # Setup the system
        system = signal.StateSpace(*system)
        # Check dimensions with inputs, states and outputs:
        n_inputs = len(values.get("inputs", []))
        n_outputs = len(values.get("outputs", []))
        n_states = len(values.get("states", []))
        assert (
            system.A.shape[0] == n_states
        ), "Given system matrix A does not match size of states"
        assert (
            system.A.shape[1] == n_states
        ), "Given system matrix A does not match size of states"
        assert (
            system.B.shape[0] == n_states
        ), "Given system matrix B does not match size of states"
        assert (
            system.B.shape[1] == n_inputs
        ), "Given system matrix B does not match size of inputs"
        assert (
            system.C.shape[0] == n_outputs
        ), "Given system matrix C does not match size of outputs"
        assert (
            system.C.shape[1] == n_states
        ), "Given system matrix C does not match size of states"
        assert (
            system.D.shape[0] == n_outputs
        ), "Given system matrix D does not match size of outputs"
        assert (
            system.D.shape[1] == n_inputs
        ), "Given system matrix D does not match size of inputs"
        values["system"] = system
        return values


class ScipyStateSpaceModel(Model):
    """
    This class holds a scipy StateSpace model.
    It uses scipy.signal.lti as a system and the
    odeint as integrator.
    """

    config: ScipyStateSpaceModelConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check if system was correctly set up
        assert isinstance(self.config.system, signal.StateSpace)

    def do_step(self, *, t_start, t_sample=None):
        if t_sample is None:
            t_sample = self.dt
        t = self._create_time_samples(t_sample=t_sample) + t_start
        u = np.array([[inp.value for inp in self.inputs] for _ in t])
        x0 = np.array([sta.value for sta in self.states])

        ufunc = interpolate.interp1d(t, u, kind="linear", axis=0, bounds_error=False)

        def f_dot(x, t, sys, ufunc):
            """The vector field of the linear system."""
            return np.dot(sys.A, x) + np.squeeze(
                np.dot(sys.B, np.nan_to_num(ufunc([t])).flatten())
            )

        x = integrate.odeint(f_dot, x0, t, args=(self.config.system, ufunc))
        y = np.dot(self.config.system.C, np.transpose(x)) + np.dot(
            self.config.system.D, np.transpose(u)
        )

        y = np.squeeze(np.transpose(y))

        # Set states based on shape:
        if len(y.shape) == 1:
            self._set_output_values(
                names=self.get_output_names(), values=[y[-1].item()]
            )
        else:
            self._set_output_values(
                names=self.get_output_names(), values=y[-1, :].tolist()
            )
        if len(x.shape) == 1:
            self._set_state_values(names=self.get_state_names(), values=[x[-1].item()])
        else:
            self._set_state_values(
                names=self.get_state_names(), values=x[-1, :].tolist()
            )
        return True

    def initialize(self, **kwargs):
        pass

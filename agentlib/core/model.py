"""This module contains just the basic Model."""

import abc
import os
import json
import logging
from copy import deepcopy
from itertools import chain
from typing import Union, List, Dict, Any, Optional, get_type_hints, Type
from pydantic import ConfigDict, BaseModel, Field, field_validator
import numpy as np
from pydantic.fields import PrivateAttr
from pydantic_core.core_schema import FieldValidationInfo

from agentlib.core.datamodels import (
    ModelVariable,
    ModelInputs,
    ModelStates,
    ModelOutputs,
    ModelParameters,
    ModelState,
    ModelParameter,
    ModelOutput,
    ModelInput,
)

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """
    Pydantic data model for controller configuration parser
    """

    user_config: dict = Field(
        default=None,
        description="The config given by the user to instantiate this class."
        "Will be stored to enable a valid overwriting of the "
        "default config and to better restart modules."
        "Is also useful to debug validators and the general BaseModuleConfig.",
    )
    name: Optional[str] = Field(default=None, validate_default=True)
    description: str = Field(default="You forgot to document your model!")
    sim_time: float = Field(default=0, title="Current simulation time")
    dt: Union[float, int] = Field(default=1, title="time increment")
    validate_variables: bool = Field(
        default=True,
        title="Validate Variables",
        description="If true, the validator of a variables value is called whenever a "
        "new value is set. Disabled by default for performance reasons.",
    )

    inputs: ModelInputs = Field(default=list())
    outputs: ModelOutputs = Field(default=list())
    states: ModelStates = Field(default=list())
    parameters: ModelParameters = Field(default=list())

    _types: Dict[str, type] = PrivateAttr(
        default={
            "inputs": ModelInput,
            "outputs": ModelOutput,
            "states": ModelState,
            "parameters": ModelParameter,
        }
    )
    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="forbid"
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, name):
        """
        Check if name of model is given. If not, use the
        name of the model class.
        """
        if name is None:
            name = str(cls).replace("Config", "")
        return name

    @field_validator("parameters", "inputs", "outputs", "states", mode="after")
    @classmethod
    def include_default_model_variables(
        cls, _: List[ModelVariable], info: FieldValidationInfo
    ):
        """
        Validator building block to merge default variables with config variables in a standard validator.
        Updates default variables when a variable with the same name is present in the config.
        Then returns the union of the default variables and the external config variables.

        This validator ensures default variables are kept
        when the config provides new variables
        """
        default = cls.model_fields[info.field_name].get_default()
        user_config = info.data["user_config"].get(info.field_name, [])
        variables: List[ModelVariable] = deepcopy(default)
        user_variables_dict = {d["name"]: d for d in user_config}

        for i, var in enumerate(variables):
            if var.name in user_variables_dict:
                var_to_update_with = user_variables_dict[var.name]
                user_config.remove(var_to_update_with)
                var_dict = var.dict()
                var_dict.update(var_to_update_with)
                variables[i] = cls._types.get_default()[info.field_name](**var_dict)
        variables.extend(
            [cls._types.get_default()[info.field_name](**var) for var in user_config]
        )
        return variables

    def get_variable_names(self):
        """
        Returns the names of every variable as list
        """
        return [
            var.name
            for var in self.inputs + self.outputs + self.states + self.parameters
        ]

    def __init__(self, **kwargs):
        kwargs["user_config"] = kwargs.copy()
        super().__init__(**kwargs)


class Model(abc.ABC):
    """
    Base class for simulation models. To implement your
    own model, inherit from this class.
    """

    config: ModelConfig

    # pylint: disable=too-many-public-methods

    def __init__(self, **kwargs):
        """
        Initializes model class
        """
        self._inputs = {}
        self._outputs = {}
        self._states = {}
        self._parameters = {}

        self.config = self.get_config_type()(**kwargs)

    @classmethod
    def get_config_type(cls) -> Type[ModelConfig]:
        return get_type_hints(cls)["config"]

    @abc.abstractmethod
    def do_step(self, *, t_start: float, t_sample: float):
        """
        Performing one simulation step
        Args:
            t_start: start time for integration
            t_sample: increment of solver integration
        Returns:
        """
        raise NotImplementedError(
            "The Model class does not implement this "
            "because it is individual to the subclasses"
        )

    @abc.abstractmethod
    def initialize(self, **kwargs):
        """
        Abstract method to define what to
        do in order to initialize the model in use.
        """
        raise NotImplementedError(
            "The Model class does not implement this "
            "because it is individual to the subclasses"
        )

    def terminate(self):
        """Terminate the model if applicable by subclass."""

    def __getattr__(self, item):
        if item in self._inputs:
            return self._inputs.get(item)
        if item in self._outputs:
            return self._outputs.get(item)
        if item in self._parameters:
            return self._parameters.get(item)
        if item in self._states:
            return self._states.get(item)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def generate_variables_config(self, filename: str = None, **kwargs) -> str:
        """
        Generate a config file (.json) to enable an user friendly
        configuration of the model.


        Args:
            filename (str): Optional path where to store the config.
                            If None, current model name and workdir are used.
            kwargs: Kwargs directly passed to the json.dump method.
        Returns:
            filepath (str): Filepath where the json is stored
        """
        if filename is None:
            filename = os.path.join(os.getcwd(), f"{self.__class__.__name__}.json")
        model_config = {
            "inputs": [inp.dict() for inp in self.inputs],
            "outputs": [out.dict() for out in self.outputs],
            "states": [sta.dict() for sta in self.states],
            "parameters": [par.dict() for par in self.parameters],
        }
        with open(filename, "w") as file:
            json.dump(obj=model_config, fp=file, **kwargs)
        return filename

    @property
    def config(self) -> ModelConfig:
        """Get the current config, which is
        a ModelConfig object."""
        return self._config

    @config.setter
    def config(self, config: Union[dict, ModelConfig]):
        """
        Set a new config.

        Args:
            config (dict, ModelConfig): The config dict or ModelConfig object.
        """
        # Instantiate the ModelConfig.
        if isinstance(config, self.get_config_type()):
            self._config = config
        else:
            self._config = self.get_config_type()(**config)
        # Update model variables.
        self._inputs = {var.name: var for var in self.config.inputs.copy()}
        self._outputs = {var.name: var for var in self.config.outputs.copy()}
        self._states = {var.name: var for var in self.config.states.copy()}
        self._parameters = {var.name: var for var in self.config.parameters.copy()}

    @property
    def description(self):
        """Get model description"""
        return self.config.description

    @description.setter
    def description(self, description: str):
        """Set model description"""
        self.config.description = description

    @description.deleter
    def description(self):
        """Delete model description. Default is then used."""
        # todo fwu do we have a use for this, or should we just get rid of deleters, and these properties alltogether?
        self.config.description = (
            self.get_config_type().model_fields["description"].default
        )

    @property
    def name(self):
        """Get model name"""
        return self.config.name

    @name.setter
    def name(self, name: str):
        """
        Set the model name
        Args:
            name (str): Name of the model
        """
        self.config.name = name

    @name.deleter
    def name(self):
        """Delete the model name"""
        self.config.name = self.get_config_type().model_fields["name"].default

    @property
    def sim_time(self):
        """Get the current simulation time"""
        return self.config.sim_time

    @sim_time.setter
    def sim_time(self, sim_time: float):
        """Set the current simulation time"""
        self.config.sim_time = sim_time

    @sim_time.deleter
    def sim_time(self):
        """Reset the current simulation time to the default value"""
        self.config.sim_time = self.get_config_type().model_fields["sim_time"].default

    @property
    def dt(self):
        """Get time increment of simulation"""
        return self.config.dt

    @property
    def variables(self):
        """Get all model variables as a list"""
        return list(
            chain.from_iterable(
                [self.inputs, self.outputs, self.parameters, self.states]
            )
        )

    @property
    def inputs(self) -> ModelInputs:
        """Get all model inputs as a list"""
        return list(self._inputs.values())

    @property
    def outputs(self) -> ModelOutputs:
        """Get all model outputs as a list"""
        return list(self._outputs.values())

    @property
    def states(self) -> ModelStates:
        """Get all model states as a list"""
        return list(self._states.values())

    @property
    def parameters(self) -> ModelParameters:
        """Get all model parameters as a list"""
        return list(self._parameters.values())

    def _create_time_samples(self, t_sample):
        """
        Function to generate an array of time samples
        using the current self.dt object.
        Note that, if self.dt is not a true divider of t_sample,
        the output array is not equally samples.

        Args:
            t_sample (float): Sample

        Returns:

        """
        samples = np.arange(0, t_sample, self.dt)
        if samples[-1] == t_sample:
            return samples
        return np.append(samples, t_sample)

    ##########################################################################################
    # Getter and setter function using names for easier access
    ##########################################################################################
    def get_outputs(self, names: List[str]):
        """Get model outputs based on given names."""
        assert isinstance(names, list), "Given names are not a list"
        return [self._outputs[name] for name in names if name in self._outputs]

    def get_inputs(self, names: List[str]):
        """Get model inputs based on given names."""
        assert isinstance(names, list), "Given names are not a list"
        return [self._inputs[name] for name in names if name in self._inputs]

    def get_parameters(self, names: List[str]):
        """Get model parameters based on given names."""
        assert isinstance(names, list), "Given names are not a list"
        return [self._parameters[name] for name in names if name in self._parameters]

    def get_states(self, names: List[str]):
        """Get model states based on given names."""
        assert isinstance(names, list), "Given names are not a list"
        return [self._states[name] for name in names if name in self._states]

    def get_output(self, name: str):
        """Get model output based on given name."""
        return self._outputs.get(name, None)

    def get_input(self, name: str):
        """Get model input based on given name."""
        return self._inputs.get(name, None)

    def get_state(self, name: str):
        """Get model state based on given name."""
        return self._states.get(name, None)

    def get_parameter(self, name: str):
        """Get model parameter based on given name."""
        return self._parameters.get(name, None)

    def set_input_value(self, name: str, value: Union[float, int, bool]):
        """Just used from external modules like simulator to set new input values"""
        self.set_input_values(names=[name], values=[value])

    def set_input_values(self, names: List[str], values: List[Union[float, int, bool]]):
        """Just used from external modules like simulator to set new input values"""
        self.__setter(variables=self._inputs, values=values, names=names)

    def _set_output_value(self, name: str, value: Union[float, int, bool]):
        """Just used internally to write output values"""
        self._set_output_values(names=[name], values=[value])

    def _set_output_values(
        self, names: List[str], values: List[Union[float, int, bool]]
    ):
        """Just used internally to write output values"""
        self.__setter(variables=self._outputs, values=values, names=names)

    def _set_state_value(self, name: str, value: Union[float, int, bool]):
        """Just used internally to write state values"""
        self._set_state_values(names=[name], values=[value])

    def _set_state_values(
        self, names: List[str], values: List[Union[float, int, bool]]
    ):
        """Just used internally to write state values"""
        self.__setter(variables=self._states, values=values, names=names)

    def set_parameter_value(self, name: str, value: Union[float, int, bool]):
        """Used externally to write new parameter values from e.g. a calibration process"""
        self.set_parameter_values(names=[name], values=[value])

    def set_parameter_values(
        self, names: List[str], values: List[Union[float, int, bool]]
    ):
        """Used externally to write new parameter values from e.g. a calibration process"""
        self.__setter(variables=self._parameters, values=values, names=names)

    def __setter(
        self,
        variables: Dict[str, ModelVariable],
        values: List[Union[float, int, bool]],
        names: List[str],
    ):
        """General setter of model values."""
        assert len(names) == len(
            values
        ), "Length of names has to equal length of values"
        for name, value in zip(names, values):
            if value is None:
                logger.warning(
                    "Tried to override variable '%s' in model '%s' "
                    "with None. Keeping the previous value of %s",
                    name,
                    self.name,
                    variables[name].value,
                )
                continue
            variables[name].set_value(
                value=value, validate=self.config.validate_variables
            )

    def get(self, name: str) -> ModelVariable:
        """
        Get any variable from using name:

        Args:
            name (str): The item to get from config by name of Variable.
                        Hence, item=ModelVariable.name
        Returns:
            var (ModelVariable): The matching variable
        Raises:
            AttributeError: If the item was not found in the variables of the
                            module.
        """
        if name in self._inputs:
            return self._inputs[name]
        if name in self._outputs:
            return self._outputs[name]
        if name in self._parameters:
            return self._parameters[name]
        if name in self._states:
            return self._states[name]
        raise ValueError(
            f"'{self.__class__.__name__}' has "
            f"no ModelVariable with the name '{name}' "
            f"in the config."
        )

    def set(self, name: str, value: Any):
        """
        Set any variable from using name:

        Args:
            name (str): The item to get from data_broker by name of Variable.
                        Hence, item=AgentVariable.name
            value (Any): Any value to set to the Variable
        Raises:
            AttributeError: If the item was not found in the variables of the
                            module.
        """
        if name in self._inputs:
            self.set_input_value(name=name, value=value)
        elif name in self._outputs:
            self._set_output_value(name=name, value=value)
        elif name in self._parameters:
            self.set_parameter_value(name=name, value=value)
        elif name in self._states:
            self._set_state_value(name=name, value=value)
        else:
            raise ValueError(
                f"'{self.__class__.__name__}' has "
                f"no ModelVariable with the name '{name}' "
                f"in the config."
            )

    def get_input_names(self):
        """
        Returns:
            names (list): A list containing all input names
        """
        return list(self._inputs.keys())

    def get_output_names(self):
        """
        Returns:
            names (list): A list containing all output names
        """
        return list(self._outputs.keys())

    def get_state_names(self):
        """
        Returns:
            names (list): A list containing all state names
        """
        return list(self._states.keys())

    def get_parameter_names(self):
        """
        Returns:
            names (list): A list containing all state names
        """
        return list(self._parameters.keys())

"""
Module contains the Simulator, used to simulate any model.
"""

import os
import warnings
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Union, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import field_validator, Field
from pydantic_core.core_schema import FieldValidationInfo

from agentlib.core import (
    BaseModule,
    BaseModuleConfig,
    Agent,
    Causality,
    AgentVariable,
    AgentVariables,
    ModelVariable,
    Model,
)
from agentlib.core.errors import OptionalDependencyError
from agentlib.models import get_model_type, UNINSTALLED_MODEL_TYPES
from agentlib.utils import custom_injection, create_time_samples


@dataclass
class SimulatorResults:
    """Class to organize in-memory simulator results."""

    index: List[float]
    columns: pd.MultiIndex
    data: List[List[float]]

    def __init__(self, variables: List[ModelVariable]):
        """
        Initializes results object input variables
        "u", outputs "x" and internal state variables "x".
        It uses a nested dict provided from model class.
        +---------------------------------------------------+
        |   |   inputs u    |  outputs y    |  states x     |
        | t |   u1  |   u2  |   y1  |   y2  |   x1  |   x2  |
        +---------------------------------------------------+
        | 1 |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |
        | 2 |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |
        |...|  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |
        |...
        """
        self.columns = pd.MultiIndex.from_arrays(
            arrays=np.array(
                [
                    [_var.causality.name for _var in variables],
                    [_var.name for _var in variables],
                    [_var.type for _var in variables],
                ]
            ),
            sortorder=0,
            names=["causality", "name", "type"],
        )
        self.index = []
        self.data = []

    def initialize(
            self,
            time: float,
    ):
        """Adds the first row to the data"""

    def df(self) -> pd.DataFrame:
        """Returns the current results as a dataframe."""
        # We do not return the last row, as it is always only half complete (since
        # inputs at time step k influence results of time step k+1. Writing in
        # incomplete dataframe would break the csv-file we append to.
        return pd.DataFrame(self.data, index=self.index, columns=self.columns)

    def write_results(self, file: str):
        """
        Dumps results which are currently in memory to a file.
        On creation of the file, the header columns are dumped, as well.
        """
        header = not Path(file).exists()
        self.df().to_csv(file, mode="a", header=header)
        # keep the last row of the results, as it is not finished (inputs missing)
        self.index = [self.index[-1]]
        self.data = [self.data[-1]]


def read_simulator_results(file: str):
    """Reads results from file with correct multi-column format."""
    return pd.read_csv(file, header=[0, 1, 2], index_col=0)


class SimulatorConfig(BaseModuleConfig):
    """
    Pydantic data model for simulator configuration parser
    """

    parameters: AgentVariables = []
    inputs: AgentVariables = []
    outputs: AgentVariables = []
    states: AgentVariables = []
    shared_variable_fields: List[str] = ["outputs"]
    t_start: Union[float, int] = Field(
        title="t_start", default=0.0, ge=0, description="Simulation start time"
    )
    t_stop: Union[float, int] = Field(
        title="t_stop", default=inf, ge=0, description="Simulation stop time"
    )
    t_sample: Union[float, int] = Field(
        title="t_sample",
        default=1,
        ge=0,
        description="Deprecated option."
    )
    t_sample_communication: Union[float, int] = Field(
        title="t_sample",
        default=1,
        validate_default=True,
        ge=0,
        description="Sample time of a full simulation step relevant for communication, including:"
                    "- Perform simulation with t_sample_simulation"
                    "- Update model results and send output values to other Agents or Modules."
    )
    t_sample_simulation: Union[float, int] = Field(
        title="t_sample_simulation",
        default=1,
        validate_default=True,
        ge=0,
        description="Sample time of the simulation itself. "
                    "The inputs of the models may be updated every other t_sample_simulation, "
                    "as long as the model supports this. Used to override dt of the model."
    )
    model: Dict

    # Model results
    save_results: bool = Field(
        title="save_results",
        default=False,
        description="If True, results are created and stored",
    )
    overwrite_result_file: bool = Field(
        title="overwrite_result",
        default=False,
        description="If True, and the result file already exists, the file is overwritten.",
    )
    result_filename: Optional[str] = Field(
        title="result_filename",
        default=None,
        description="If not None, results are stored in that filename."
                    "Needs to be a .csv file",
    )
    result_sep: str = Field(
        title="result_sep",
        default=",",
        description="Separator in the .csv file. Only relevant if "
                    "result_filename is passed",
    )
    result_causalities: List[Causality] = Field(
        title="result_causalities",
        default=[Causality.input, Causality.output],
        description="List of causalities to store. Default stores "
                    "only inputs and outputs",
    )
    write_results_delay: Optional[float] = Field(
        title="Write Results Delay",
        default=None,
        description="Sampling interval for which the results are written to disc in seconds.",
        validate_default=True,
        gt=0,
    )
    measurement_uncertainty: Union[Dict[str, float], float] = Field(
        title="measurement_uncertainty",
        default=0,
        description="Either pass a float and add the percentage uncertainty "
                    "to all measurements from the model."
                    "Or pass a Dict and specify the model variable name as key"
                    "and the associated uncertainty as a float",
    )
    validate_incoming_values: Optional[bool] = Field(
        default=False,  # we overwrite the default True in base, to be more efficient
        title="Validate Incoming Values",
        description="If true, the validator of the AgentVariable value is called when "
                    "receiving a new value from the DataBroker. In the simulator, this "
                    "is False by default, as we expect to receive a lot of measurements"
                    " and want to be efficient.",
    )
    update_inputs_on_callback: bool = Field(
        title="update_inputs_on_callback",
        default=True,
        description="Deprecated! Will be removed in future versions."
                    "If True, model inputs are updated if they are updated in data_broker."
                    "Else, the model inputs are updated before each simulation.",
    )

    @field_validator("result_filename")
    @classmethod
    def check_nonexisting_csv(cls, result_filename, info: FieldValidationInfo):
        """Check if the result_filename is a .csv file or an hf
        and assert that it does not exist."""
        if not info.data.get("save_results", False):
            # No need to check as filename will never be used anyways
            return None
        if result_filename is None:
            return result_filename
        if not result_filename.endswith(".csv"):
            raise TypeError(
                f"Given result_filename ends with "
                f'{result_filename.split(".")[-1]} '
                f"but should be a .csv file"
            )
        if os.path.isfile(result_filename):
            # remove result file, so a new one can be created
            if info.data["overwrite_result_file"]:
                os.remove(result_filename)
                return result_filename
            raise FileExistsError(
                f"Given result_filename at {result_filename} "
                f"already exists. We won't overwrite it automatically. "
                f"You can use the key word 'overwrite_result_file' to "
                f"activate automatic overwrite."
            )
        # Create path in case it does not exist
        fpath = os.path.dirname(result_filename)
        if fpath:
            os.makedirs(fpath, exist_ok=True)
        return result_filename

    @field_validator("t_stop")
    @classmethod
    def check_t_stop(cls, t_stop, info: FieldValidationInfo):
        """Check if stop is greater than start time"""
        t_start = info.data.get("t_start")
        assert t_stop > t_start, "t_stop must be greater than t_start"
        return t_stop

    @field_validator("t_sample_communication", "t_sample_simulation")
    @classmethod
    def check_t_sample(cls, t_sample, info: FieldValidationInfo):
        """Check if t_sample is smaller than stop-start time"""
        t_start = info.data.get("t_start")
        t_stop = info.data.get("t_stop")
        t_sample_old = info.data.get("t_sample")
        if t_sample_old != 1:  # A change in the default shows t_sample is still in the config of the user
            if info.field_name == "t_sample_simulation":
                t_sample = 1
            else:
                t_sample = t_sample_old
        assert (
                t_start + t_sample <= t_stop
        ), "t_stop-t_start must be greater than t_sample"
        return t_sample

    @field_validator("update_inputs_on_callback")
    @classmethod
    def deprecate_update_inputs_on_callback(cls, update_inputs_on_callback, info: FieldValidationInfo):
        """Check if t_sample is smaller than stop-start time"""
        if update_inputs_on_callback:
            warnings.warn("update_inputs_on_callback is deprecated, remove it from your config.",
                          category=DeprecationWarning)
        else:
            warnings.warn(
                "update_inputs_on_callback is deprecated, remove it from your config. "
                "Will use update_inputs_on_callback=True",
                category=DeprecationWarning
            )
        return True

    @field_validator("t_sample")
    @classmethod
    def deprecate_t_sample(cls, t_sample, info: FieldValidationInfo):
        """Deprecates the t_sample field in favor of t_sample_communication and t_sample_simulation."""
        warnings.warn(
            "t_sample is deprecated, use t_sample_communication, "
            "t_sample_simulation for a concise separation of the two. "
            "Will use the given t_sample for t_sample_communication and t_sample_simulation=1 s, "
            "the `model.dt` default.",
        )
        return t_sample

    @field_validator("write_results_delay")
    @classmethod
    def set_default_t_sample(cls, write_results_delay, info: FieldValidationInfo):
        t_sample = info.data["t_sample"]
        if write_results_delay is None:
            # 5 is an arbitrary default which should balance writing new results as
            # soon as possible to disk with saving file I/O overhead
            return 5 * t_sample
        if write_results_delay < t_sample:
            raise ValueError(
                "Saving results more frequently than you simulate makes no sense. "
                "Increase write_results_delay above t_sample."
            )
        return write_results_delay

    @field_validator("model")
    @classmethod
    def check_model(cls, model, info: FieldValidationInfo):
        """Validate the model input"""
        parameters = info.data.get("parameters")
        inputs = info.data.get("inputs")
        outputs = info.data.get("outputs")
        states = info.data.get("states")
        dt = info.data.get("t_sample_simulation")
        if "dt" in model and dt != model["dt"]:
            warnings.warn(
                f"Given model {model['dt']=} differs from {dt=} of simulator. "
                f"Using models dt, consider switching to t_sample_simulation."
            )
        else:
            model["dt"] = dt
        if "type" not in model:
            raise KeyError(
                "Given model config does not " "contain key 'type' (type of the model)."
            )
        _type = model.pop("type")
        if isinstance(_type, dict):
            custom_cls = custom_injection(config=_type)
            model = custom_cls(**model)
        elif isinstance(_type, str):
            if _type in UNINSTALLED_MODEL_TYPES:
                raise OptionalDependencyError(
                    dependency_name=_type,
                    dependency_install=UNINSTALLED_MODEL_TYPES[_type],
                    used_object=f"model {_type}",
                )
            model = get_model_type(_type)(
                **model,
                parameters=convert_agent_vars_to_list_of_dicts(parameters),
                inputs=convert_agent_vars_to_list_of_dicts(inputs),
                outputs=convert_agent_vars_to_list_of_dicts(outputs),
                states=convert_agent_vars_to_list_of_dicts(states),
            )
        # Check if model was correctly initialized
        assert isinstance(model, Model)
        return model


class Simulator(BaseModule):
    """
    The Simulator is the interface between simulation models
    and further other implementations. It contains all interface functions for
    interacting with the standard model class.
    """

    config: SimulatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        # Initialize instance attributes
        self._model = None
        self.model = self.config.model
        self._result: SimulatorResults = SimulatorResults(
            variables=self._get_result_model_variables()
        )
        self._save_count: int = 1  # tracks, how often results have been saved
        self._inputs_changed_since_last_results_saving = False
        self._register_input_callbacks()
        self.logger.info("%s initialized!", self.__class__.__name__)

    def terminate(self):
        """Terminate the model"""
        self.model.terminate()
        super().terminate()

    @property
    def model(self) -> Model:
        """
        Getter for current simulation model

        Returns:
            agentlib.core.model.Model: Current simulation model
        """
        return self._model

    @model.setter
    def model(self, model: Model):
        """
        Setter for current simulation model.
        Also initializes it if needed!
        Args:
            model (agentlib.core.model.Model): model to set as current simulation model
        """
        if not isinstance(model, Model):
            self.logger.error(
                "You forgot to pass a valid model to the simulator module!"
            )
            raise TypeError(
                f"Given model is of type {type(model)} "
                f"but should be an instance of Model or a valid subclass"
            )
        self._model = model
        if self.config.t_start and self.env.offset:
            self.logger.warning(
                "config.t_start and env.offset are both non-zero. "
                "This may cause unexpected behavior. Ensure that this "
                "is intended and you know what you are doing."
            )
        self.model.initialize(
            t_start=self.config.t_start + self.env.config.offset,
            t_stop=self.config.t_stop,
        )
        self.logger.info("Model successfully loaded model: %s", self.model.name)

    def run(self, until=None):
        """
        Runs the simulator in stand-alone mode if needed
        Attention: If the environment is connected to another environment
        all scheduled process will be started in this environment.
        """
        if until is None:
            self.env.run(until=self.config.t_stop - self.config.t_start)
        else:
            self.env.run(until=until)

    def register_callbacks(self):
        pass

    def _register_input_callbacks(self):
        """Register input callbacks"""
        # Possible inputs are Inputs and parameters.
        # Outputs and states are always the result of the model
        # "Complicated" double for-loop to avoid boilerplate code
        for _type, model_var_names, ag_vars, callback in zip(
                ["input", "parameter"],
                [self.model.get_input_names(), self.model.get_parameter_names()],
                [self.config.inputs, self.config.parameters],
                [self._callback_update_model_input, self._callback_update_model_parameter],
        ):
            for var in ag_vars:
                if var.name in model_var_names:
                    self.logger.info(
                        "Registered callback for model %s %s ", _type, var.name
                    )
                    self.agent.data_broker.register_callback(
                        alias=var.alias,
                        source=var.source,
                        callback=callback,
                        name=var.name,
                    )
                # Case for variable overwriting
                if var.value is not None:
                    self.logger.debug(
                        "Updating model %s %s=%s", _type, var.name, var.value
                    )
                    self.model.set(name=var.name, value=var.value)
                    self._inputs_changed_since_last_results_saving = True

    def _callback_update_model_input(self, inp: AgentVariable, name: str):
        """Set given model input value to the model"""
        self.logger.debug("Updating model input %s=%s", name, inp.value)
        self.model.set_input_value(name=name, value=inp.value)
        self._inputs_changed_since_last_results_saving = True

    def _callback_update_model_parameter(self, par: AgentVariable, name: str):
        """Set given model parameter value to the model"""
        self.logger.debug("Updating model parameter %s=%s", name, par.value)
        self.model.set_parameter_value(name=name, value=par.value)
        self._inputs_changed_since_last_results_saving = True

    def process(self):
        """
        This function creates a endless loop for the single simulation step event,
        updating inputs, simulating, model results and then outputs.

        In a simulation step following happens:
        1. Specify the end time of the simulation from the agents perspective.
        **Important note**: The agents use unix-time as a timestamp and start
        the simulation with the current datetime (represented by self.env.time),
        the model starts at 0 seconds (represented by self.env.now).
        2. Directly after the simulation we store the results with
        the output time and then call the timeout in the environment,
        hence actually increase the environment time.
        3. Once the environment time reached the simulation time,
        we send the updated output values to other modules and agents by setting
        them the data_broker.
        """
        self._update_results(timestamp_inputs=self.env.time, timestamp_outputs=self.env.time)
        while True:
            # Simulate
            t_samples = create_time_samples(
                t_end=self.config.t_sample_communication,
                dt=self.config.t_sample_simulation
            )
            _t_start_simulation_loop = self.env.time
            self.logger.debug("Doing simulation steps %s ...", t_samples)
            for _idx, _t_sample in enumerate(t_samples[:-1]):
                _t_start = self.env.now + self.config.t_start
                dt_sim = t_samples[_idx + 1] - _t_sample
                self.model.do_step(t_start=_t_start, t_sample=dt_sim)
                if _idx == len(t_samples) - 2 or self._inputs_changed_since_last_results_saving:
                    if not self._inputs_changed_since_last_results_saving:
                        # Did not change during simulation step
                        timestamp_inputs = _t_start_simulation_loop
                    else:
                        # The inputs are only applied at self.env.time, not when they are received by the communicator
                        timestamp_inputs = self.env.time
                    # Update the results
                    self._update_results(
                        timestamp_outputs=self.env.time + dt_sim,
                        timestamp_inputs=timestamp_inputs
                    )
                yield self.env.timeout(dt_sim)
            # Communicate
            self.update_module_vars()

    def update_module_vars(self):
        """
        Method to write current model output and states
        values to the module outputs and states.
        """
        # pylint: disable=logging-fstring-interpolation
        for _type, model_get, agent_vars in zip(
                ["state", "output"],
                [self.model.get_state, self.model.get_output],
                [self.config.states, self.config.outputs],
        ):
            for var in agent_vars:
                mo_var = model_get(var.name)
                if mo_var is None:
                    raise KeyError(f"Given variable {var.name} not found in model.")
                value = self._get_uncertain_value(model_variable=mo_var)
                self.logger.debug("Updating %s %s=%s", _type, var.name, value)
                self.set(name=var.name, value=value)

    def _get_uncertain_value(self, model_variable: ModelVariable) -> float:
        """Get the value with added uncertainty based on the value of the variable"""
        if isinstance(self.config.measurement_uncertainty, dict):
            bias = self.config.measurement_uncertainty.get(model_variable.name, 0)
        else:
            bias = self.config.measurement_uncertainty
        return model_variable.value * (1 + np.random.uniform(-bias, bias))

    def get_results(self) -> Optional[pd.DataFrame]:
        """
        Return the current results.

        Returns:
            pd.DataFrame: The results DataFrame.
        """
        if not self.config.save_results:
            return
        file = self.config.result_filename
        if file:
            self._result.write_results(self.config.result_filename)
            df = read_simulator_results(file)
        else:
            df = self._result.df()
        df = df.droplevel(level=2, axis=1).droplevel(level=0, axis=1)
        return df

    def cleanup_results(self):
        if not self.config.save_results or not self.config.result_filename:
            return
        os.remove(self.config.result_filename)

    def _update_results(self, timestamp_outputs, timestamp_inputs):
        """
        Adds model variables to the SimulationResult object
        at the given timestamp.
        """
        if not self.config.save_results:
            return

        inp_values = [var.value for var in self._get_result_input_variables()]
        self._inputs_changed_since_last_results_saving = False

        out_values = [var.value for var in self._get_result_output_variables()]
        _len_outputs = len(out_values)

        # Two cases:
        # - Either both timestamps are the same and new, add them both
        # - or the inputs are for an earlier timestamp -> first add the inputs and then append the new outputs index.
        if timestamp_inputs == timestamp_outputs:
            # self.logger.debug("Storing data at the same time stamp %s s", timestamp_outputs)
            self._result.index.append(timestamp_outputs)
            self._result.data.append(out_values + inp_values)
        elif timestamp_inputs < timestamp_outputs:
            # add inputs in the time stamp before adding outputs, as they are active from
            # the start of this interval
            if timestamp_inputs in self._result.index:
                if timestamp_inputs == self._result.index[-1]:
                    # self.logger.debug("Adding inputs to last time stamp %s s", timestamp_inputs)
                    self._result.data[-1] = self._result.data[-1][:_len_outputs] + inp_values
                # else: pass, as inputs are outdated (have been changed during simulation step)
            else:
                # This case may occur if inputs changed during simulation.
                # In this case, the inputs hold for current time - t_sample_simulation, but the outputs
                # hold for the current time. In this case, just add Nones as outputs.
                self._result.index.append(timestamp_inputs)
                self._result.data.append([None] * _len_outputs + inp_values)
                # self.logger.debug(
                #     "Storing inputs only due to changes during simulation at time stamp %s s",
                #     timestamp_inputs
                # )

            # self.logger.debug("Storing outputs at time stamp %s s", timestamp_outputs)
            self._result.index.append(timestamp_outputs)
            self._result.data.append(out_values + [None] * len(inp_values))
        else:
            raise ValueError("Storing inputs ahead of outputs is not supported.")

        if (
                self.config.result_filename is not None
                and timestamp_outputs // (self.config.write_results_delay * self._save_count) > 0
        ):
            self._save_count += 1
            self._result.write_results(self.config.result_filename)

    def _get_result_model_variables(self) -> AgentVariables:
        """
        Gets all variables to be saved in the result based
        on self.result_causalities.
        """

        # THE ORDER OF THIS CONCAT IS IMPORTANT. The _update_results function will
        # extend the outputs with the inputs
        return self._get_result_output_variables() + self._get_result_input_variables()

    def _get_result_input_variables(self) -> AgentVariables:
        """Gets all input variables to be saved in the results based on
        self.result_causalities. Input variables are added to the results at the time
        index before an interval, i.e. parameters and inputs."""
        _variables = []
        for causality in self.config.result_causalities:
            if causality == Causality.input:
                _variables.extend(self.model.inputs)
            elif causality in [Causality.parameter, Causality.calculatedParameter]:
                _variables.extend(self.model.parameters)
        return _variables

    def _get_result_output_variables(self) -> AgentVariables:
        """Gets all output variables to be saved in the results based on
        self.result_causalities. Input variables are added to the results at the time
        index after an interval, i.e. locals and outputs."""
        _variables = []
        for causality in self.config.result_causalities:
            if causality == Causality.output:
                _variables.extend(self.model.outputs)
            elif causality == Causality.local:
                _variables.extend(self.model.states)
        return _variables


def convert_agent_vars_to_list_of_dicts(var: AgentVariables) -> List[Dict]:
    """
    Function to convert AgentVariables to a list of dictionaries containing information for
    ModelVariables.
    """
    var_dict_list = [
        agent_var.dict(exclude={"source", "alias", "shared", "rdf_class"})
        for agent_var in var
    ]
    return var_dict_list

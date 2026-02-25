"""
Module contains the Simulator, used to simulate any model.
"""

import os
import warnings
from dataclasses import dataclass, field
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

    # Configuration
    filename: Optional[str] = None
    header_written: bool = False

    # Data Buffers
    index: List[float] = field(default_factory=list)
    data: List[List[float]] = field(default_factory=list)

    # State tracking
    _current_inputs: List[float] = field(default_factory=list)
    _current_outputs: List[float] = field(default_factory=list)
    _columns: pd.MultiIndex = None
    _input_count: int = 0
    _output_count: int = 0

    def setup(self, input_vars: List[ModelVariable], output_vars: List[ModelVariable]):
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
        Also initializes the internal buffers.
        """
        variables = output_vars + input_vars
        self._input_count = len(input_vars)
        self._output_count = len(output_vars)

        # Initialize current inputs with current values
        self._current_inputs = [var.value for var in input_vars]

        self._columns = pd.MultiIndex.from_arrays(
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

    def update_inputs(self, values: List[float], time: float, capture_all_inputs: bool):
        """
        Updates the result with the inputs creating a full result row (output + input).
        If capture_all_inputs is True, creates a row with NaN outputs.
        """
        self._current_inputs = values
        # Results can already hold the input (at t_sample_communication created by
        # the output writing) or the input time is new (created by an input callback)
        if not self.index or time != self.index[-1]:
            # For capture_all_inputs, append the inputs created by an input callback
            if capture_all_inputs:
                self.index.append(time)
                # index is not in data, if results have been written to disk
                # Create row: [NaN, NaN, ..., In1, In2, ...]
                row = [None] * self._output_count + self._current_inputs
                # If timestamp is new, this needs to be appended
                self.data.append(row)
        else:
            # Create row: [Out1, Out2, ..., In1, In2, ...]
            row = self.data[-1][:self._output_count] + self._current_inputs
            # Update timestamp with new inputs
            self.data[-1] = row

    def update_outputs(self, values: List[float], time: float):
        """
        Stores a result row at the end of a simulation step.
        Combines provided output values with None for input values, as these are
        updated in the next time step.
        """
        # Create row: [Out1, Out2, ..., None, None, ...]
        row = values + [None] * self._input_count
        self.index.append(time)
        self.data.append(row)

    def update_current_outputs(self, values: List[float]):
        """
        Stores the current output values of intermediate simulation steps.
        """
        self._current_outputs = values

    def initialize_outputs(self, time):
        """
        Initializes output data with Nones.
        """
        self.index.append(time)
        # Create row: [None, None, ..., In1, In2, ...]
        self.data.append([None] * self._output_count + self._current_inputs)

    def initialize_inputs(self, values: List[float]):
        """
        Initializes input data with Nones.
        """
        self._current_inputs = values

    def write_results(self):
        """
        Dumps results which are currently in memory to the file.
        Clears memory after writing to keep footprint low.
        """
        if not self.filename or not self.data:
            return

        df = pd.DataFrame(self.data, index=self.index, columns=self._columns)

        # Write header only once
        header = not self.header_written and not Path(self.filename).exists()
        df.to_csv(self.filename, mode="a", header=header)

        self.header_written = True

        # Clear buffers
        self.index.clear()
        self.data.clear()

    def df(self) -> pd.DataFrame:
        """Returns the current results as a dataframe."""
        return pd.DataFrame(self.data, index=self.index, columns=self._columns)


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
        title="t_sample", default=1, ge=0, description="Deprecated option."
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
    capture_all_inputs: bool = Field(
        title="capture_all_inputs",
        default=False,
        description="If True, results are stored immediately when "
                    "inputs change, even during simulation steps.",
    )
    write_results_delay: Optional[float] = Field(
        title="Write Results Delay",
        default=None,
        description="Sampling interval for which the results are written to disc in seconds.",
        validate_default=True,
        gt=0,
    )
    update_inputs_on_callback: bool = Field(
        title="update_inputs_on_callback",
        default=True,
        description="Deprecated! Will be removed in future versions."
                    "If True, model inputs are updated if they are updated in data_broker."
                    "Else, the model inputs are updated before each simulation.",
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

        # Handle legacy t_sample logic
        if t_sample_old != 1:
            if info.field_name == "t_sample_simulation":
                t_sample = 1
            else:
                t_sample = t_sample_old
        assert (
                t_start + t_sample <= t_stop
        ), "t_stop-t_start must be greater than t_sample"
        return t_sample

    @field_validator("t_sample_communication")
    @classmethod
    def check_t_comm_against_sim(cls, t_sample_communication,
                                 info: FieldValidationInfo):
        """Check if t_sample is smaller than stop-start time"""
        t_sample_simulation = info.data.get("t_sample_simulation")
        if t_sample_simulation is not None:
            if t_sample_simulation > t_sample_communication:
                warnings.warn(
                    f"{t_sample_communication=} is smaller than {t_sample_simulation=}",
                    category=UserWarning
                )
        return t_sample_communication

    @field_validator("update_inputs_on_callback")
    @classmethod
    def deprecate_update_inputs_on_callback(cls, update_inputs_on_callback,
                                            info: FieldValidationInfo):
        """Check if t_sample is smaller than stop-start time"""
        warnings.warn(
            "update_inputs_on_callback is deprecated, remove it from your config. "
            "Will use update_inputs_on_callback=True",
            category=DeprecationWarning
        )
        return True

    @field_validator("t_sample")
    @classmethod
    def deprecate_t_sample(cls, t_sample, info: FieldValidationInfo):
        """Deprecates the t_sample field in favor of t_sample_communication
        and t_sample_simulation."""
        warnings.warn(
            "t_sample is deprecated, use t_sample_communication for storing outputs "
            "and t_sample_simulation for the actual simulation step. "
            "Will use the given t_sample for t_sample_communication and "
            "t_sample_simulation=1 s, the `model.dt` default.",
        )
        return t_sample

    @field_validator("write_results_delay")
    @classmethod
    def set_default_t_sample(cls, write_results_delay, info: FieldValidationInfo):
        t_comm = info.data.get("t_sample_communication", 1)

        if write_results_delay is None:
            # Default to writing every 5 communication steps to balance I/O
            return t_comm * 5

        if write_results_delay < t_comm:
            raise ValueError("write_results_delay should be >= t_sample_communication")
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

        self._model = None
        self.model = self.config.model
        self._pending_input_change_time = None

        # Caching variables for performance (avoid list comprehensions in loop)
        self._input_vars = self._get_result_input_variables()
        self._output_vars = self._get_result_output_variables()

        # Initialize Result Handler
        self._result = SimulatorResults(filename=self.config.result_filename)
        if self.config.save_results:
            self._result.setup(input_vars=self._input_vars,
                               output_vars=self._output_vars)

        # Initialize local time trackers
        self._last_write_time = 0.0
        self._last_communication_time = self.env.time

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
                    self._pending_input_change_time = self.env.time

    def _callback_update_model_input(self, inp: AgentVariable, name: str):
        """Set given model input value to the model"""
        self.logger.debug("Updating model input %s=%s", name, inp.value)
        self.model.set_input_value(name=name, value=inp.value)
        self._pending_input_change_time = self.env.time

    def _callback_update_model_parameter(self, par: AgentVariable, name: str):
        """Set given model parameter value to the model"""
        self.logger.debug("Updating model parameter %s=%s", name, par.value)
        self.model.set_parameter_value(name=name, value=par.value)
        self._pending_input_change_time = self.env.time

    def process(self):
        """
        Main simulation loop.
        Handles simulation stepping, result logging, and synchronization.
        """
        # 1. Log Initial State (t=0)
        if self.config.save_results:
            # Ensure the result buffer has the correct initial inputs
            in_values = [var.value for var in self._input_vars]
            self._result.initialize_inputs(in_values)
            self._result.initialize_outputs(self.env.time)
            # Prevent false positive "input change" log at t=0 due to initialization callbacks
            self._pending_input_change_time = None
        while True:
            # Determine the time points for the next communication step
            t_samples = create_time_samples(
                t_end=self.config.t_sample_communication,
                dt=self.config.t_sample_simulation
            )

            # Iterate through simulation sub-steps
            for i in range(len(t_samples) - 1):
                dt_sim = float(t_samples[i + 1] - t_samples[i])

                # 2. Check for Input Changes (Pre-Step)
                # If inputs changed since the last step (or during the yield), we log them now.
                # This ensures the new inputs are recorded at the current timestamp,
                # separate from the outputs of the *previous* step (which were logged at
                # the end of the last loop).
                if self._pending_input_change_time:
                    if self.config.save_results:
                        # Create row: [t=Current, Out=NaN, In=New]
                        self._log_inputs(self._pending_input_change_time,
                                         capture_all_inputs=self.config.capture_all_inputs)
                    self._pending_input_change_time = None

                # 3. Perform Simulation Step
                self.model.do_step(
                    t_start=self.config.t_start + self.env.now,
                    t_sample=dt_sim
                )

                # 4. Store intermediate outputs
                if self.config.save_results:
                    out_values = [var.value for var in self._output_vars]
                    self._result.update_current_outputs(out_values)

                # 5. Write results
                if self.config.save_results:
                    # Since simulation has been performed, the model and its results are
                    # already a time step ahead
                    current_time = self.env.time + self.config.t_sample_simulation
                    if ((current_time - self._last_communication_time) >=
                            self.config.t_sample_communication):
                        # Update time tracker for communication
                        self._last_communication_time = ((current_time //
                                                         self.config.t_sample_communication) *
                                                         self.config.t_sample_communication)
                        # Check if we need to write to disk, do this before storing
                        # outputs, to initialize the new row after dumping the results
                        self._check_and_write_to_disk(self.env.time +
                                                      self.config.t_sample_simulation)

                        # Log the outputs resulting from the step we just finished.
                        # These will be paired with the inputs active for the next simulation step.
                        self._log_outputs(self._last_communication_time)

                # 6. Wait for the environment
                yield self.env.timeout(dt_sim)

            # 7. End of Communication Step (Post-Step)
            # Communicate
            self.update_module_vars()

    def _log_inputs(self, time: float, capture_all_inputs: bool):
        """
        Update the result object with current inputs.
        If capture_all_inputs is True, a row is added immediately.
        """
        values = [var.value for var in self._input_vars]
        self._result.update_inputs(values, time, capture_all_inputs=capture_all_inputs)

    def _log_outputs(self, time: float):
        """
        Add a full result row (Outputs + Last Inputs).
        """
        values = [var.value for var in self._output_vars]
        self._result.update_outputs(values, time)

    def _check_and_write_to_disk(self, time):
        """Check if write delay has passed and dump to disk."""
        if not self.config.result_filename:
            return

        # Inputs are written in the next time step, therefore results
        # are behind actual env time
        current_result_time = time - self.config.t_sample_communication
        if (current_result_time - self._last_write_time) >= self.config.write_results_delay:
            self._result.write_results()
            self._last_write_time = time

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
            self._result.write_results()
            df = read_simulator_results(file)
        else:
            df = self._result.df()
        df = df.droplevel(level=2, axis=1).droplevel(level=0, axis=1)
        return df

    def cleanup_results(self):
        if not self.config.save_results or not self.config.result_filename:
            return
        os.remove(self.config.result_filename)

    def _get_result_input_variables(self) -> List[ModelVariable]:
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

    def _get_result_output_variables(self) -> List[ModelVariable]:
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
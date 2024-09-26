"""
Module contains the Simulator, used to simulate any model.
"""

import os
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
from agentlib.utils import custom_injection


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
        return pd.DataFrame(self.data[:-1], index=self.index[:-1], columns=self.columns)

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
    model: Dict

    t_start: Union[float, int] = Field(
        title="t_start", default=0.0, ge=0, description="Simulation start time"
    )
    t_stop: Union[float, int] = Field(
        title="t_stop", default=inf, ge=0, description="Simulation stop time"
    )
    t_sample: Union[float, int] = Field(
        title="t_sample", default=1, ge=0, description="Simulation sample time"
    )
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
    update_inputs_on_callback: bool = Field(
        title="update_inputs_on_callback",
        default=True,
        description="If True, model inputs are updated if they are updated in data_broker."
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

    @field_validator("t_sample")
    @classmethod
    def check_t_sample(cls, t_sample, info: FieldValidationInfo):
        """Check if t_sample is smaller than stop-start time"""
        t_start = info.data.get("t_start")
        t_stop = info.data.get("t_stop")
        assert (
            t_start + t_sample <= t_stop
        ), "t_stop-t_start must be greater than t_sample"
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
        if self.config.update_inputs_on_callback:
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

    def _callback_update_model_input(self, inp: AgentVariable, name: str):
        """Set given model input value to the model"""
        self.logger.debug("Updating model input %s=%s", name, inp.value)
        self.model.set_input_value(name=name, value=inp.value)

    def _callback_update_model_parameter(self, par: AgentVariable, name: str):
        """Set given model parameter value to the model"""
        self.logger.debug("Updating model parameter %s=%s", name, par.value)
        self.model.set_parameter_value(name=name, value=par.value)

    def process(self):
        """
        This function creates a endless loop for the single simulation step event.
        The do_step() function needs to return a generator.
        """
        self._update_result_outputs(self.env.time)
        while True:
            self.do_step()
            yield self.env.timeout(self.config.t_sample)
            self.update_module_vars()

    def do_step(self):
        """
        Generator function to perform a simulation step,
        update inputs, outputs and model results.

        In a simulation step following happens:
        1. Update inputs (only necessary if self.update_inputs_on_callback = False)
        2. Specify the end time of the simulation from the agents perspective.
        **Important note**: The agents use unix-time as a timestamp and start
        the simulation with the current datetime (represented by self.env.time),
        the model starts at 0 seconds (represented by self.env.now).
        3. Directly after the simulation we send the updated output values
        to other modules and agents by setting them the data_broker.
        Even though the environment time is not already at the end time specified above,
        we explicitly add the timestamp to the variables.
        This way other agents and communication has the maximum time possible to
        process the outputs and send input signals to the simulation.
        4. Call the timeout in the environment,
        hence actually increase the environment time.
        """
        if not self.config.update_inputs_on_callback:
            # Update inputs manually
            self.update_model_inputs()
        # Simulate
        self.model.do_step(
            t_start=(self.env.now + self.config.t_start), t_sample=self.config.t_sample
        )
        # Update the results and outputs
        self._update_results()

    def update_model_inputs(self):
        """
        Internal method to write current data_broker to simulation model.
        Only update values, not other module_types.
        """
        model_input_names = (
            self.model.get_input_names() + self.model.get_parameter_names()
        )
        for inp in self.variables:
            if inp.name in model_input_names:
                self.logger.debug("Updating model variable %s=%s", inp.name, inp.value)
                self.model.set(name=inp.name, value=inp.value)

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

    def _update_results(self):
        """
        Adds model variables to the SimulationResult object
        at the given timestamp.
        """
        if not self.config.save_results:
            return
        timestamp = self.env.time + self.config.t_sample
        inp_values = [var.value for var in self._get_result_input_variables()]

        # add inputs in the time stamp before adding outputs, as they are active from
        # the start of this interval
        self._result.data[-1].extend(inp_values)
        # adding output results afterwards. If the order here is switched, the [-1]
        # above will point to the wrong entry
        self._update_result_outputs(timestamp)
        if (
            self.config.result_filename is not None
            and timestamp // (self.config.write_results_delay * self._save_count) > 0
        ):
            self._save_count += 1
            self._result.write_results(self.config.result_filename)

    def _update_result_outputs(self, timestamp: float):
        """Updates results with current values for states and outputs."""
        self._result.index.append(timestamp)
        out_values = [var.value for var in self._get_result_output_variables()]
        self._result.data.append(out_values)

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

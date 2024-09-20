"""This module contains the FMUModel class."""

import queue
import shutil
import os
import logging
import pathlib
import uuid
from itertools import chain
from typing import Union, List

import attrs
import pydantic

from pydantic import field_validator, FilePath
from agentlib.core import Model, ModelConfig
from agentlib.core.errors import OptionalDependencyError
from agentlib.core.datamodels import ModelVariable, Causality

try:
    import fmpy.fmi2
    import fmpy
    from fmpy.fmi1 import FMICallException
except ImportError as err:
    raise OptionalDependencyError(
        dependency_name="fmu", dependency_install="fmpy", used_object="FMU-model"
    ) from err

logger = logging.getLogger(__name__)


class FmuModelConfig(ModelConfig):
    """
    The Config of FMUModels overwrite the default
    ModelConfig to redefine the system and add relevant
    fields like path and tolerance of the simulation.
    """

    path: FilePath
    tolerance: float = 0.001
    extract_fmu: bool = False
    log_fmu: bool = True
    only_config_variables: bool = pydantic.Field(
        default=True,
        description="If True, only the variables passed to this model by a simulator "
        "will be read and written at each simulation step (specified by "
        "dt).",
    )

    @field_validator("path")
    @classmethod
    def check_path(cls, path):
        """Check if the path has the correct extension"""
        # check file extension
        assert path.suffix in [".fmu", ".mo"], "Unknown file-extension"
        return path


class FmuModel(Model):
    """Class to wrap any FMU Model into the Model-Standard
    of the agentlib.
    """

    config: FmuModelConfig

    def __init__(self, **kwargs):
        # Private map to link variable names (str) to the fmu value reference (int)
        self._variables_vr = {}
        self._unzip_dir = None
        # Initialize config
        super().__init__(**kwargs)
        # Actively set the path to trigger the automatic setup (evokes __init_fmu)
        self.system = self.__init_fmu()
        initial_write_values = self.inputs + self.parameters
        self._variables_to_write = queue.Queue()
        [self._variables_to_write.put(v) for v in initial_write_values]

    @property
    def tolerance(self):
        """Get the tolerance of FMU simulation"""
        return self.config.tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float):
        """Set the tolerance in the config."""
        self.config.tolerance = tolerance

    @tolerance.deleter
    def tolerance(self):
        """Delete the tolerance and restore the default value."""
        self.config.tolerance = self.get_config_type().tolerance

    @property
    def extract_fmu(self):
        """Get whether the fmu shall be extracted to a new
        directory or if the temp folder is used."""
        return self.config.extract_fmu

    def do_step(self, *, t_start, t_sample=None):
        if t_sample is None:
            t_sample = self.dt
        # Write current values to system
        while not self._variables_to_write.empty():
            self.__write_value(self._variables_to_write.get_nowait())
        t_samples = self._create_time_samples(t_sample=t_sample) + t_start
        try:
            for _idx, _t_sample in enumerate(t_samples[:-1]):
                self.system.doStep(
                    currentCommunicationPoint=_t_sample,
                    communicationStepSize=t_samples[_idx + 1] - _t_sample,
                )
        except FMICallException as e:
            # Raise a different error, as simpy does not work well with FMI Errors
            raise RuntimeError(
                "The fmu had an internal error. Please check the logs to analyze it."
            ) from e
        # Read current values from system
        self.__read_values()
        return True

    def initialize(self, **kwargs):
        """
        Initializes FMU model

        Required kwargs:
            t_start (float): Start time of simulation
            t_stop (float): Stop time of simulation
        """
        logger.info("Initializing model...")
        # Handle Logging of the FMU itself:
        try:
            callbacks = fmpy.fmi2.fmi2CallbackFunctions()
            callbacks.logger = fmpy.fmi2.fmi2CallbackLoggerTYPE(self._fmu_logger)
            callbacks.allocateMemory = fmpy.fmi2.fmi2CallbackAllocateMemoryTYPE(
                fmpy.calloc
            )
            callbacks.freeMemory = fmpy.fmi2.fmi2CallbackFreeMemoryTYPE(fmpy.free)
            from fmpy.logging import addLoggerProxy
            from ctypes import byref

            addLoggerProxy(byref(callbacks))
        except Exception as err:
            logger.error("Could not setup custom logger in FMU model: %s", err)
            callbacks = None

        self.system.instantiate(
            callbacks=callbacks,
            loggingOn=False,  # Only for debug of fmu itself
            visible=False,  # Only for debug of fmu itself
        )
        self.system.reset()
        self.system.setupExperiment(
            startTime=kwargs["t_start"],
            stopTime=kwargs["t_stop"],
            tolerance=self.tolerance,
        )
        self.system.enterInitializationMode()
        self.system.exitInitializationMode()
        logger.info("Model: %s initialized", self.name)

    def _fmu_logger(self, component, instanceName, status, category, message):
        """Print the FMU's log messages to the command line (works for both FMI 1.0 and 2.0)"""
        # pylint: disable=unused-argument, invalid-name
        if self.config.log_fmu:
            label = ["OK", "WARNING", "DISCARD", "ERROR", "FATAL", "PENDING"][status]
            _level_map = {
                "OK": logging.INFO,
                "WARNING": logging.WARNING,
                "DISCARD": logging.WARNING,
                "ERROR": logging.ERROR,
                "FATAL": logging.FATAL,
                "PENDING": logging.FATAL,
            }
            logger.log(level=_level_map[label], msg=message.decode("utf-8"))

    def __init_fmu(self) -> fmpy.fmi2.FMU2Slave:
        path = self.config.path
        # System setup:
        # extract the FMU
        if self.extract_fmu:
            # Create own unzip directory
            _path = pathlib.Path(path)
            if not _path.is_absolute():
                _path = pathlib.Path(os.getcwd()).joinpath(_path)
            _unzip_dir = _path.parents[0].joinpath(
                f'{_path.name.replace(".fmu", "")}' f"_extracted_{uuid.uuid4()}"
            )
            _unzip_dir = str(_unzip_dir)
        else:
            _unzip_dir = None
        cur_cwd = os.getcwd()
        self._unzip_dir = fmpy.extract(filename=path, unzipdir=_unzip_dir)
        os.chdir(cur_cwd)  # Reset cwd. fmpy changes it sometimes.
        # Read the model description
        _model_description = fmpy.read_model_description(self._unzip_dir, validate=True)
        _system = fmpy.fmi2.FMU2Slave(
            guid=_model_description.guid,
            unzipDirectory=self._unzip_dir,
            modelIdentifier=_model_description.coSimulation.modelIdentifier,
            instanceName=__name__,
            fmiCallLogger=None,
        )
        self.name = _model_description.modelName
        if _model_description.description is not None:
            self.description = _model_description.description

        # Variable setup:
        # Get the inputs, outputs, internals and parameters
        self._variables_vr = {}
        _vars = {
            Causality.input: [],
            Causality.parameter: [],
            Causality.calculatedParameter: [],
            Causality.output: [],
            Causality.local: [],
            Causality.independent: [],
        }
        config_vars = set(self.config.get_variable_names())
        for _model_var in _model_description.modelVariables:
            # Convert to an agentlib ModelVariable object
            if _model_var.type == "String":
                logger.warning(
                    "String variable %s omitted. Not supported in AgentLib.",
                    _model_var.name,
                )
                continue  # Don't allow string model variables

            # if desired, we skip adding variables to this model instance if they are
            # not specified from outside. They will remain within the
            # fmpy.fmi2.FMU2Slave instance
            if self.config.only_config_variables and _model_var.name not in config_vars:
                continue

            _vars[_model_var.causality].append(
                dict(
                    name=_model_var.name,
                    type=_model_var.type,
                    value=(
                        self._converter(_model_var.type, _model_var.start)
                        if (
                            _model_var.causality
                            in [
                                Causality.parameter,
                                Causality.calculatedParameter,
                                Causality.input,
                            ]
                            and _model_var.start is not None
                        )
                        else None
                    ),
                    unit=(
                        _model_var.unit
                        if _model_var.unit is not None
                        else attrs.fields(ModelVariable).unit.default
                    ),
                    description=(
                        _model_var.description
                        if _model_var.description is not None
                        else attrs.fields(ModelVariable).description.default
                    ),
                    causality=_model_var.causality,
                    variability=_model_var.variability,
                )
            )
            self._variables_vr.update({_model_var.name: _model_var.valueReference})
        # This sets the inputs, outputs, internals and parameters and variables
        self.config = {
            "inputs": _vars[Causality.input],
            "outputs": _vars[Causality.output],
            "parameters": _vars[Causality.parameter]
            + _vars[Causality.calculatedParameter],
            "states": _vars[Causality.local] + _vars[Causality.independent],
            **self.config.model_dump(
                exclude={"inputs", "outputs", "states", "parameters"}
            ),
        }
        return _system

    def __write_value(self, var: ModelVariable):
        # One can only set inputs and parameters!

        if var.value is None:
            logger.error(
                "Tried setting the value of %s to None. This will not be set in the "
                "FMU.",
                var.name,
            )
            return

        _vr = self._variables_vr[var.name]
        if var.type == "Real":
            self.system.setReal([_vr], [float(var.value)])
        elif var.type in ["Integer", "Enumeration"]:
            self.system.setInteger([_vr], [int(var.value)])
        elif var.type == "Boolean":
            self.system.setBoolean([_vr], [bool(var.value)])
        else:
            logger.error("Variable %s not valid for this model!", var.name)

    def set_input_values(self, names: List[str], values: List[Union[float, int, bool]]):
        """Sets input values in the model and in the FMU."""
        super().set_input_values(names=names, values=values)
        for name in names:
            var = self._inputs[name]
            self._variables_to_write.put(var)

    def set_parameter_values(
        self, names: List[str], values: List[Union[float, int, bool]]
    ):
        """Sets parameter values in the model and in the FMU."""
        super().set_parameter_values(names=names, values=values)
        for name in names:
            var = self._parameters[name]
            self._variables_to_write.put(var)

    def __read_values(self):
        for _var in chain.from_iterable([self.outputs, self.parameters, self.states]):
            _vr = self._variables_vr[_var.name]
            if _var.type == "Real":
                _var.value = self.system.getReal([_vr])[0]
            elif _var.type in ["Integer", "Enumeration"]:
                _var.value = self.system.getInteger([_vr])[0]
            elif _var.type == "Boolean":
                _var.value = self.system.getBoolean([_vr])[0]
            else:
                raise TypeError(
                    f"Unsupported type: {_var.type} for variable {_var.name}"
                )

    def __enter__(self):
        self._terminate_and_free_instance()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._terminate_and_free_instance()
        # clean up
        shutil.rmtree(self._unzip_dir, ignore_errors=True)

    def _terminate_and_free_instance(self):
        try:
            self.system.terminate()
        except Exception as err:
            logger.error("Could not terminate FMU instance %s", err)
        try:
            self.system.freeInstance()
        except Exception as err:
            logger.error("Could not terminate FMU instance %s", err)

    def terminate(self):
        """Overwrite base method"""
        self.__exit__(exc_type=None, exc_val=None, exc_tb=None)

    @staticmethod
    def _converter(type_of_var: str, value):
        _mapper = {"Boolean": bool, "Real": float, "Integer": int, "Enumeration": int}
        if type_of_var in _mapper:
            return _mapper[type_of_var](value)
        # else
        return value

"""This module contains the base AgentModule."""

from __future__ import annotations
import abc
import json
import logging
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    List,
    Dict,
    Union,
    Any,
    TypeVar,
    Optional,
    get_type_hints,
    Type,
)

import pydantic
from pydantic import field_validator, ConfigDict, BaseModel, Field, PrivateAttr
from pydantic_core import core_schema
from pydantic.json_schema import GenerateJsonSchema

from agentlib.core.environment import CustomSimpyEnvironment
from agentlib.core.errors import ConfigurationError
from agentlib.core.datamodels import (
    AgentVariable,
    Source,
    AgentVariables,
    AttrsToPydanticAdaptor,
)
from agentlib.core import datamodels
import agentlib.core.logging_ as agentlib_logging
from agentlib.utils.fuzzy_matching import fuzzy_match, RAPIDFUZZ_IS_INSTALLED
from agentlib.utils.validators import (
    include_defaults_in_root,
    update_default_agent_variable,
    is_list_of_agent_variables,
    is_valid_agent_var_config,
)

if TYPE_CHECKING:
    # this avoids circular import
    from agentlib.core import Agent


logger = logging.getLogger(__name__)


class BaseModuleConfig(BaseModel):
    """
    Pydantic data model for basic module configuration
    """

    # The type is relevant to load the correct module class.
    type: Union[str, Dict[str, str]] = Field(
        title="Type",
        description="The type of the Module. Used to find the Python-Object "
        "from all agentlib-core and plugin Module options. If a dict is given,"
        "it must contain the keys 'file' and 'class_name'. "
        "'file' is the filepath of a python file containing the Module."
        "'class_name' is the name of the Module class within this file.",
    )
    # A module is uniquely identified in the MAS using agent_id and module_id.
    # The module_id should be unique inside one agent.
    # This is checked inside the agent-class.
    module_id: str = Field(
        description="The unqiue id of the module within an agent, "
        "used only to communicate withing the agent."
    )
    validate_incoming_values: Optional[bool] = Field(
        default=True,
        title="Validate Incoming Values",
        description="If true, the validator of the AgentVariable value is called when "
        "receiving a new value from the DataBroker.",
    )
    log_level: Optional[str] = Field(
        default=None,
        description="The log level for this Module. "
        "Default uses the root-loggers level."
        "Options: DEBUG; INFO; WARNING; ERROR; CRITICAL",
    )
    shared_variable_fields: List[str] = Field(
        default=[],
        description="A list of strings with each string being a field of the Modules configs. "
        "The field must be or contain an AgentVariable. If the field is added to this list, "
        "all shared attributes of the AgentVariables will be set to True.",
        validate_default=True,
    )
    # Aggregation of all instances of an AgentVariable in this Config
    _variables: AgentVariables = PrivateAttr(default=[])

    # The config given by the user to instantiate this class.
    # Will be stored to enable a valid overwriting of the
    # default config and to better restart modules.
    # Is also useful to debug validators and the general BaseModuleConfig.
    _user_config: dict = PrivateAttr(default=None)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        frozen=True,
    )

    def get_variables(self):
        """Return the private attribute with all AgentVariables"""
        return self._variables

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        """
        Custom schema method to
        - Add JSON Schema for custom attrs types Source and AgentVariable
        - put log_level last, as it is the only optional field of the module config.
        Used to better display relevant options of children classes in GUIs.
        """
        if "schema_generator" in kwargs:
            raise ValueError("Custom schema_generator is not supported for BaseModule.")

        class CustomGenerateJsonSchema(GenerateJsonSchema):
            """
            This class in necessary, as the default object type
            AttrsToPydanticAdaptor (e.g. Source, AgentVariable) are
            not json serializable by default.
            """

            def default_schema(self, schema: core_schema.WithDefaultSchema):
                if "default" in schema:
                    _default = schema["default"]
                    if isinstance(_default, AttrsToPydanticAdaptor):
                        schema["default"] = _default.json()
                return super().default_schema(schema=schema)

        kwargs["schema_generator"] = CustomGenerateJsonSchema
        schema = super().model_json_schema(*args, **kwargs)
        definitions = schema.get("$defs", {})
        definitions_out = {}
        for class_name, metadata in definitions.items():
            if class_name in datamodels.ATTRS_MODELS:
                class_object: AttrsToPydanticAdaptor = getattr(datamodels, class_name)
                metadata = class_object.get_json_schema()
            definitions_out[class_name] = metadata
        if definitions_out:
            schema["$defs"] = definitions_out

        log_level = schema["properties"].pop("log_level")
        shared_variable_fields = schema["properties"].pop("shared_variable_fields")
        schema["properties"]["shared_variable_fields"] = shared_variable_fields
        schema["properties"]["log_level"] = log_level
        return schema

    @classmethod
    def check_if_variables_are_unique(cls, names):
        """Check if a given iterable of AgentVariables have a
        unique name."""
        if len(names) != len(set(names)):
            for name in set(names.copy()):
                names.remove(name)
            raise ValueError(
                f"{cls.__name__} contains variables with the same name. The "
                f"following appear at least twice: {' ,'.join(names)}"
            )

    @field_validator("shared_variable_fields")
    @classmethod
    def check_valid_fields(cls, shared_variables_fields):
        """
        Check if the shared_variables_fields are valid
        fields.
        """
        wrong_public_fields = set(shared_variables_fields).difference(
            cls.model_fields.keys()
        )
        if wrong_public_fields:
            raise ConfigurationError(
                f"Public fields {wrong_public_fields} do not exist. Maybe you "
                f"misspelled them?"
            )
        return shared_variables_fields

    @field_validator("log_level")
    @classmethod
    def check_valid_level(cls, log_level: str):
        """
        Check if the given log_level is valid
        """
        if log_level is None:
            return log_level
        log_level = log_level.upper()
        if not isinstance(logging.getLevelName(log_level), int):
            raise ValueError(
                f"Given log level '{log_level}' is not "
                f"supported by logging library."
            )
        return log_level

    @classmethod
    def merge_variables(
        cls,
        pre_validated_instance: BaseModuleConfig,
        user_config: dict,
        agent_id: str,
        shared_variable_fields: List[str],
    ):
        """
        Merge, rigorously check and validate the input of
        all AgentVariables into the module.
        This function:

        - Collects all variables
        - Checks if duplicate names (will cause errors in the get() function.
        """
        _vars = []
        # Extract all variables from fields
        for field_name, field in cls.model_fields.items():
            # If field is missing in values, validation of field was not
            # successful. Continue and pydantic will later raise the ValidationError
            if field_name not in pre_validated_instance.model_fields:
                continue

            pre_merged_attr = pre_validated_instance.__getattribute__(field_name)
            # we need the type if plugins subclass the AgentVariable

            if isinstance(pre_merged_attr, AgentVariable):
                update_var_with = user_config.get(field_name, {})

                make_shared = field_name in shared_variable_fields

                var = update_default_agent_variable(
                    default_var=field.default,
                    user_data=update_var_with,
                    make_shared=make_shared,
                    agent_id=agent_id,
                    field_name=field_name,
                )
                _vars.append(var)
                pre_validated_instance.__setattr__(field_name, var)

            elif is_list_of_agent_variables(pre_merged_attr):
                user_config_var_dicts = user_config.get(field_name, [])
                type_ = pre_merged_attr[0].__class__
                update_vars_with = [
                    conf
                    for conf in user_config_var_dicts
                    if is_valid_agent_var_config(conf, field_name, type_)
                ]

                make_shared = field_name in shared_variable_fields
                variables = include_defaults_in_root(
                    variables=update_vars_with,
                    field=field,
                    type_=type_,  # subtype of AgentVariable
                    make_shared=make_shared,
                    agent_id=agent_id,
                    field_name=field_name,
                )

                _vars.extend(variables)
                pre_validated_instance.__setattr__(field_name, variables)

        # Extract names
        variable_names = [var.name for var in _vars]

        # First check if names exists more than once
        cls.check_if_variables_are_unique(names=variable_names)

        for _var in _vars:
            # case the agent id is a different agent
            if (_var.source.agent_id != agent_id) and (
                _var.source.module_id is not None
            ):
                logger.warning(
                    "Setting given module_id '%s' in variable '%s' to None. "
                    "You can not specify module_ids of other agents.",
                    _var.source.module_id,
                    _var.name,
                )
                _var.source = Source(agent_id=_var.source.agent_id)

        return _vars

    @classmethod
    def default(cls, field: str):
        return cls.model_fields[field].get_default()

    def __init__(self, _agent_id, *args, **kwargs):
        _user_config = kwargs.copy()
        try:
            super().__init__(*args, **kwargs)
        except pydantic.ValidationError as e:
            better_error = self._improve_extra_field_error_messages(e)
            raise better_error
        # Enable mutation
        self.model_config["frozen"] = False
        self._variables = self.__class__.merge_variables(
            pre_validated_instance=self,
            user_config=_user_config,
            agent_id=_agent_id,
            shared_variable_fields=self.shared_variable_fields,
        )
        self._user_config = _user_config
        # Disable mutation
        self.model_config["frozen"] = True

    @classmethod
    def _improve_extra_field_error_messages(
        cls, e: pydantic.ValidationError
    ) -> pydantic.ValidationError:
        """Checks the validation errors for invalid fields and adds suggestions for
        correct field names to the error message."""
        if not RAPIDFUZZ_IS_INSTALLED:
            return e

        error_list = e.errors()
        for error in error_list:
            if not error["type"] == "extra_forbidden":
                continue

            # change error type to literal because it allows for context
            error["type"] = "literal_error"
            # pydantic automatically prints the __dict__ of an error, so it is
            # sufficient to just assign the suggestions to an arbitrary attribute of
            # the error
            suggestions = fuzzy_match(
                target=error["loc"][0], choices=cls.model_fields.keys()
            )
            if suggestions:
                error["ctx"] = {
                    "expected": f"a valid Field name. Field '{error['loc'][0]}' does "
                    f"not exist. Did you mean any of {suggestions}?"
                }

        return pydantic.ValidationError.from_exception_data(
            title=e.title, line_errors=error_list
        )


BaseModuleConfigClass = TypeVar("BaseModuleConfigClass", bound=BaseModuleConfig)


class BaseModule(abc.ABC):
    """
    Basic module used by any agent.
    Besides a common configuration, where ids
    and variables are defined, this class manages
    the setting and getting of variables and relevant
    attributes.
    """

    # pylint: disable=too-many-public-methods

    def __init__(self, *, config: dict, agent: Agent):
        self._agent = agent
        self.logger = agentlib_logging.create_logger(
            env=self.env, name=f"{self.agent.id}/{config['module_id']}"
        )
        self.config = config  # evokes the config setter
        # Add process to environment
        self.env.process(self.process())
        self.register_callbacks()

    ############################################################################
    # Methods to inherit by subclasses
    ############################################################################

    @classmethod
    def get_config_type(cls) -> Type[BaseModuleConfigClass]:
        return get_type_hints(cls).get("config")

    @abc.abstractmethod
    def register_callbacks(self):
        raise NotImplementedError("Needs to be implemented by derived modules")

    @abc.abstractmethod
    def process(self):
        """This abstract method must be implemented in order to sync the module
        with the other processes of the agent and the whole MAS."""
        raise NotImplementedError("Needs to be implemented by derived modules")

    def terminate(self):
        """
        Terminate all relevant processes of the module.
        This is necessary to correctly terminate an agent
        at runtime. Not all modules may need this, hence it is
        not an abstract method.
        """
        self.logger.info(
            "Successfully terminated module %s in agent %s", self.id, self.agent.id
        )

    ############################################################################
    # Properties
    ############################################################################

    @property
    def agent(self) -> Agent:
        """Get the agent this module is located in."""
        return self._agent

    @property
    def config(self) -> BaseModuleConfigClass:
        """
        The module config.

        Returns:
            BaseModuleConfigClass: Config of type self.config_type
        """
        return self._config

    @config.setter
    def config(self, config: Union[BaseModuleConfig, dict, str]):
        """Set a new config"""
        if self.get_config_type() is None:
            raise ConfigurationError(
                "The module has no valid config. Please make sure you "
                "specify the class attribute 'config' when writing your module."
            )
        if isinstance(config, str):
            config = json.loads(config)
        self._config = self.get_config_type()(_agent_id=self.agent.id, **config)

        # Update variables:
        self._variables_dict: Dict[str, AgentVariable] = self._copy_list_to_dict(
            self.config.get_variables()
        )
        # Now de-and re-register all callbacks:
        self._register_variable_callbacks()

        # Set log-level
        if self.config.log_level is not None:
            if not logging.getLogger().hasHandlers():
                _root_lvl_int = logging.getLogger().level
                _log_lvl_int = logging.getLevelName(self.config.log_level)
                if _log_lvl_int < _root_lvl_int:
                    self.logger.error(
                        "Log level '%s' is below root loggers level '%s'. "
                        "Without calling logging.basicConfig, "
                        "logs will not be printed.",
                        self.config.log_level,
                        logging.getLevelName(_root_lvl_int),
                    )
            self.logger.setLevel(self.config.log_level)

        # Call the after config update:
        self._after_config_update()

    def _after_config_update(self):
        """
        This function is called after the config of
        the module is updated.

        Overwrite this function to enable custom behaviour
        after your config is updated.
        For instance, a simulator may re-initialize it's model,
        or a coordinator in an ADMM-MAS send new settings to
        the participants.

        Returns nothing, the config is immutable
        """

    def _register_variable_callbacks(self):
        """
        This functions de-registers and then re-registers
        callbacks for all variables of the module to
        update their specific values.
        """
        # Keep everything in THAT order!!
        for name, var in self._variables_dict.items():
            self.agent.data_broker.deregister_callback(
                alias=var.alias,
                source=var.source,
                callback=self._callback_config_vars,
                name=name,
            )
        for name, var in self._variables_dict.items():
            self.agent.data_broker.register_callback(
                alias=var.alias,
                source=var.source,
                callback=self._callback_config_vars,
                name=name,
                _unsafe_no_copy=True,
            )

    @property
    def env(self) -> CustomSimpyEnvironment:
        """Get the environment of the agent."""
        return self.agent.env

    @property
    def id(self) -> str:
        """Get the module's id"""
        return self.config.module_id

    @property
    def source(self) -> Source:
        """Get the source of the module,
        containing the agent and module id"""
        return Source(agent_id=self.agent.id, module_id=self.id)

    @property
    def variables(self) -> List[AgentVariable]:
        """Return all values as a list."""
        return [v.copy() for v in self._variables_dict.values()]

    ############################################################################
    # Get, set and updaters
    ############################################################################
    def get(self, name: str) -> AgentVariable:
        """
        Get any variable matching the given name:

        Args:
            name (str): The item to get by name of Variable.
                        Hence, item=AgentVariable.name
        Returns:
            var (AgentVariable): The matching variable
        Raises:
            KeyError: If the item was not found in the variables of the
                      module.
        """
        try:
            return self._variables_dict[name].copy()
        except KeyError as err:
            raise KeyError(
                f"'{self.__class__.__name__}' has "
                f"no AgentVariable with the name '{name}' "
                f"in the configs variables."
            ) from err

    def get_value(self, name: str) -> Any:
        """
        Get the value of the variable matching the given name:

        Args:
            name (str): The item to get by name of Variable.
                        Hence, item=AgentVariable.name
        Returns:
            var (Any): The matching value
        Raises:
            KeyError: If the item was not found in the variables of the
                      module.
        """
        try:
            return deepcopy(self._variables_dict[name].value)
        except KeyError as err:
            raise KeyError(
                f"'{self.__class__.__name__}' has "
                f"no AgentVariable with the name '{name}' "
                f"in the configs variables."
            ) from err

    def set(self, name: str, value: Any, timestamp: float = None):
        """
        Set any variable by using the name:

        Args:
            name (str): The item to get by name of Variable.
                        Hence, item=AgentVariable.name
            value (Any): Any value to set to the Variable
            timestamp (float): The timestamp associated with the variable.
                If None, current environment time is used.

        Raises:
            AttributeError: If the item was not found in the variables of the
                            module.
        """
        # var = self.get(name)
        var = self._variables_dict[name]
        var = self._update_relevant_values(
            variable=var, value=value, timestamp=timestamp
        )
        self.agent.data_broker.send_variable(
            variable=var.copy(update={"source": self.source}),
            copy=False,
        )

    def update_variables(self, variables: List[AgentVariable], timestamp: float = None):
        """
        Updates the given list of variables in the current data_broker.
        If a given Variable is not in the config of the module, an
        error is raised.
        TODO: check if this is needed, we currently don't use it anywhere

        Args:
            variables: List with agent_variables.
            timestamp: The timestamp associated with the variable.
                If None, current environment time is used.
        """
        if timestamp is None:
            timestamp = self.env.time

        for v in variables:
            if v.name not in self._variables_dict:
                raise ValueError(
                    f"'{self.__class__.__name__}' has "
                    f"no AgentVariable with the name '{v.name}' "
                    f"in the config."
                )
            self.set(name=v.name, value=v.value, timestamp=timestamp)

    ############################################################################
    # Private and or static class methods
    ############################################################################

    def _update_relevant_values(
        self, variable: AgentVariable, value: Any, timestamp: float = None
    ):
        """
        Update the given variables fields
        with the given value (and possibly timestamp)
        Args:
            variable (AgentVariable): The variable to be updated.
            value (Any): Any value to set to the Variable
            timestamp (float): The timestamp associated with the variable.
                If None, current environment time is used.

        Returns:
            AgentVariable: The updated variable
        """
        # Update value
        variable.value = value
        # Update timestamp
        if timestamp is None:
            timestamp = self.env.time
        variable.timestamp = timestamp
        # Return updated variable
        return variable

    def _callback_config_vars(self, variable: AgentVariable, name: str):
        """
        Callback to update the AgentVariables of the module defined in the
        config.

        Args:
            variable: Variable sent by data broker
            name: Name of the variable in own config
        """
        own_var = self._variables_dict[name]
        value = deepcopy(variable.value)
        own_var.set_value(value=value, validate=self.config.validate_incoming_values)
        own_var.timestamp = variable.timestamp

    @staticmethod
    def _copy_list_to_dict(ls: List[AgentVariable]):
        # pylint: disable=invalid-name
        return {var.name: var for var in ls.copy()}

    def get_results(self):
        """
        Returns results of this modules run.

        Override this method, if your module creates data that you would like to obtain
         after the run.

        Returns:
            Some form of results data, often in the form of a pandas DataFrame.
        """

    def cleanup_results(self):
        """
        Deletes all files this module created.

        Override this method, if your module creates e.g. results files etc.
        """

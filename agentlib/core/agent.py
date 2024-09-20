"""
Module containing only the Agent class.
"""

import json
import threading
from typing import Union, List, Dict, TypeVar, Optional

from pathlib import Path
from pydantic import field_validator, BaseModel, FilePath, Field

import agentlib
import agentlib.core.logging_ as agentlib_logging
from agentlib.core import (
    Environment,
    LocalDataBroker,
    RTDataBroker,
    BaseModule,
    DataBroker,
)
from agentlib.core.environment import CustomSimpyEnvironment
from agentlib.utils import custom_injection
from agentlib.utils.load_config import load_config

BaseModuleClass = TypeVar("BaseModuleClass", bound=BaseModule)


class AgentConfig(BaseModel):
    """
    Class containing settings / config for an Agent.

    Contains just two fields, id and modules.
    """

    id: Union[str, int] = Field(
        title="id",
        description="The ID of the Agent, should be unique in "
        "the multi-agent-system the agent is living in.",
    )
    modules: List[Union[Dict, FilePath]] = None
    check_alive_interval: float = Field(
        title="check_alive_interval",
        default=1,
        ge=0,
        description="Check every other check_alive_interval second "
        "if the threads of the agent are still alive."
        "If that's not the case, exit the main thread of the "
        "agent. Updating this value at runtime will "
        "not work as all processes have already been started.",
    )
    max_queue_size: Optional[int] = Field(
        default=1000,
        ge=-1,
        description="Maximal number of waiting items in data-broker queues. "
        "Set to -1 for infinity",
    )

    @field_validator("modules")
    @classmethod
    def check_modules(cls, modules: List):
        """Validator to ensure all modules are in dict-format."""
        modules_loaded = []
        for module in modules:
            if isinstance(module, (str, Path)):
                if Path(module).exists():
                    with open(module, "r") as f:
                        module = json.load(f)
                else:
                    module = json.loads(module)
            modules_loaded.append(module)
        return modules_loaded


class Agent:
    """
    The base class for all reactive agent implementations.

    Args:
        config (Union[AgentConfig, FilePath, str, dict]):
            A config object to initialize the agents config
        env (Environment): The environment the agent is running in
    """

    def __init__(self, *, config, env: Environment):
        """
        Create instance of Agent
        """
        self._modules = {}
        self._threads: Dict[str, threading.Thread] = {}
        self.env = env
        self.is_alive = True
        config: AgentConfig = load_config(config, config_type=AgentConfig)
        data_broker_logger = agentlib_logging.create_logger(
            env=self.env, name=f"{config.id}/DataBroker"
        )
        if env.config.rt:
            self._data_broker = RTDataBroker(
                env=env, logger=data_broker_logger, max_queue_size=config.max_queue_size
            )
            self.register_thread(thread=self._data_broker.thread)
        else:
            self._data_broker = LocalDataBroker(
                env=env, logger=data_broker_logger, max_queue_size=config.max_queue_size
            )
        # Update modules
        self.config = config
        # Setup logger
        self.logger = agentlib_logging.create_logger(env=self.env, name=self.id)

        # Register the thread monitoring if configured
        if env.config.rt:
            self.env.process(self._monitor_threads())

    @property
    def id(self) -> str:
        """
        Getter for current agent's id

        Returns:
            str: current id of agent
        """
        return self.config.id

    def __repr__(self):
        return f"Agent {self.id}"

    @property
    def config(self) -> AgentConfig:
        """
        Get the config (AgentConfig) of the agent

        Returns:
            AgentConfig: An instance of AgentConfig
        """
        return self._config

    @config.setter
    def config(self, config: Union[AgentConfig, FilePath, str, dict]):
        """
        Set the config of the agent.
        As relevant info may be updated, all modules
        are re-registered.

        Args:
            config (Union[AgentConfig, FilePath, str, dict]):
                Essentially any object which can be parsed by pydantic
        """
        # Set the config

        self._config = load_config(config, config_type=AgentConfig)
        self._register_modules()

    @property
    def data_broker(self) -> DataBroker:
        """
        Get the data_broker of the agent

        Returns:
            DataBroker: An instance of the DataBroker class
        """
        return self._data_broker

    @property
    def env(self) -> CustomSimpyEnvironment:
        """
        Get the environment the agent is in

        Returns:
            Environment: The environment instance
        """
        return self._env

    @env.setter
    def env(self, env: Environment):
        """
        Set the environment of the agent

        Args:
            env (Environment): The environment instance
        """
        self._env = env

    @property
    def modules(self) -> List[BaseModuleClass]:
        """
        Get all modules of agent

        Returns:
            List[BaseModule]: List of all modules
        """
        return list(self._modules.values())

    def get_module(self, module_id: str) -> BaseModuleClass:
        """
        Get the module by given module_id.
        If no such module exists, None is returned
        Args:
            module_id (str): Id of the module to return
        Returns:
            BaseModule: Module with the given name
        """
        return self._modules.get(module_id, None)

    def register_thread(self, thread: threading.Thread):
        """
        Registers the given thread to the dictionary of threads
        which need to run in order for the agent
        to work.

        Args:
            thread threading.Thread:
                The thread object
        """
        name = thread.name
        if name in self._threads:
            raise KeyError(
                f"Given thread with name '{name}' is already a registered thread"
            )
        if not thread.daemon:
            self.logger.warning(
                "'%s' is not a daemon thread. "
                "If the agent raises an error, the thread will keep running.",
                name,
            )
        self._threads[name] = thread

    def _monitor_threads(self):
        """Process loop to monitor the threads of the agent."""
        while True:
            for name, thread in self._threads.items():
                if not thread.is_alive():
                    msg = (
                        f"The thread {name} is not alive anymore. Exiting agent. "
                        f"Check errors above for possible reasons"
                    )
                    self.logger.critical(msg)
                    self.is_alive = False
                    raise RuntimeError(msg)
            yield self.env.timeout(self.config.check_alive_interval)

    def _register_modules(self):
        """
        Function to register all modules from the
        current config.
        The module_ids need to be unique inside the
        agents config.
        The agent object (self) is passed to the modules.
        This is the reason the function is not inside the
        validator.
        """
        updated_modules = []
        for module_config in self.config.modules:
            module_cls = get_module_class(module_config=module_config)
            _module_id = module_config.get("module_id", module_cls.__name__)

            # Insert default module id if it did not exist:
            module_config.update({"module_id": _module_id})

            if _module_id in updated_modules:
                raise KeyError(
                    f"Module with module_id '{_module_id}' "
                    f"exists multiple times inside agent "
                    f"{self.id}. Use unique names only."
                )

            updated_modules.append(_module_id)

            if _module_id in self._modules:
                # Update the config:
                self.get_module(_module_id).config = module_config
            else:
                # Add the modules to the list of modules
                self._modules.update(
                    {_module_id: module_cls(agent=self, config=module_config)}
                )

    def get_results(self, cleanup=True):
        """
        Gets the results of this agent.
        Args:
            cleanup: If true, created files are deleted.
        """
        results = {}
        for module in self.modules:
            result = module.get_results()
            if result is not None:
                results[module.id] = result
        if cleanup:
            self.clean_results()
        return results

    def clean_results(self):
        """
        Calls the cleanup_results function of all modules, removing files that
        were created by them.
        """
        for module in self.modules:
            try:
                module.cleanup_results()
            except BaseException as e:
                self.logger.error(
                    f"Could not cleanup results for the following module: {module.id}. "
                    f"The reason is the following exception: {e}"
                )

    def terminate(self):
        """Calls the terminate function of all modules."""
        for module in self.modules:
            module.terminate()


def get_module_class(module_config):
    """
    Return the Module-Class object for the given config.

    Args:
        module_config (dict): Config of the module to return
    Returns:
        BaseModule: Module-Class object
    """
    _type = module_config.get("type")

    if isinstance(_type, str):
        # Get the module-class from the agentlib
        module_cls = agentlib.modules.get_module_type(module_type=_type.casefold())
    elif isinstance(_type, dict):
        # Load module class
        module_cls = custom_injection(config=_type)
    else:
        raise TypeError(
            f"Given module type is of type '{type(_type)}' "
            f"but should be str or dict."
        )

    return module_cls

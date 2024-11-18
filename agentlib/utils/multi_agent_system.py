"""
Module containing a local agency to test any LocalMASAgency system
without the need of cloneMAP.
"""

import abc
import json
import logging
import multiprocessing
import threading
from pathlib import Path
from typing import List, Dict, Union, Any

import pandas as pd
from pydantic import (
    field_validator,
    ConfigDict,
    BaseModel,
    PrivateAttr,
    Field,
    FilePath,
)

from agentlib.core import Agent, Environment
from agentlib.core.agent import AgentConfig
from agentlib.utils.load_config import load_config

logger = logging.getLogger(__name__)


class MAS(BaseModel):
    """Parent class for all MAS"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_configs: List[Union[dict, FilePath, str]]
    env: Union[Environment, dict, FilePath] = Field(
        default_factory=Environment,
        title="env",
        description="The environment for the agents.",
    )
    variable_logging: bool = Field(
        default=False,
        title="variable_logging",
        description="Enable variable logging in all agents with sampling rate of environment.",
    )
    _agent_configs: Dict[str, AgentConfig] = PrivateAttr(default={})

    def __init__(self, **data: Any) -> None:
        """Add all agents as Agent object"""
        super().__init__(**data)
        for agent_config in self.agent_configs:
            self.add_agent(config=agent_config)

    @field_validator("agent_configs")
    @classmethod
    def setup_agents(cls, agent_configs):
        """Load agent configs and add them."""
        cfgs = []
        for cfg in agent_configs:
            cfgs.append(load_config(cfg, config_type=AgentConfig))
        return cfgs

    def add_agent(self, config: AgentConfig):
        """
        Add an agent to the local agency with the
        given agent config.

        Args:
            config Dict: agent config
        """

        if self.variable_logging:
            if isinstance(self.env, dict):
                config = self.add_agent_logger(
                    config=config, sampling=self.env.get("t_sample", 60)
                )
            else:
                config = self.add_agent_logger(
                    config=config, sampling=self.env.config.t_sample
                )
        self._agent_configs[config.id] = config.model_copy()
        logger.info("Registered agent %s in agency", config.id)

    @staticmethod
    def add_agent_logger(config: AgentConfig, sampling=60) -> AgentConfig:
        """Adds the AgentLogger to the list of configs.

        Args:
            config dict: The config to be updated
            sampling=
        """
        # Add Logger config
        filename = f"variable_logs//Agent_{config.id}_Logger.log"
        cfg = {
            "module_id": "AgentLogger",
            "type": "AgentLogger",
            "t_sample": sampling,
            "values_only": True,
            "filename": filename,
            "overwrite_log": True,
            "clean_up": False,
        }
        config.modules.append(cfg)
        return config

    @abc.abstractmethod
    def run(self, until):
        """
        Run the MAS.
        Args:
            until: The time until which the simulation should run.

        Returns:

        """
        raise NotImplementedError("'run' is not implemented by the parent class MAS.")


class LocalMASAgency(MAS):
    """
    Local LocalMASAgency agency class which holds the agents in a common environment,
    executes and terminates them.
    """

    _agents: Dict[str, Agent] = PrivateAttr(default={})

    @field_validator("env")
    @classmethod
    def setup_env(cls, env):
        """Setup the env if a config is given."""
        if isinstance(env, Environment):
            return env
        if isinstance(env, (Path, str)):
            if Path(env).exists():
                with open(env, "r") as f:
                    env = json.load(f)
        return Environment(config=env)

    def add_agent(self, config: AgentConfig):
        """Also setup the agent directly"""
        super().add_agent(config=config)
        self.setup_agent(id=config.id)

    def stop_agency(self):
        """Stop all threads"""
        logger.info("Stopping agency")
        self.terminate_agents()

    def run(self, until):
        """Execute the LocalMASAgency and terminate it after run is finished"""
        self.env.run(until=until)
        self.stop_agency()

    def __enter__(self):
        """Enable 'with' statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exit in 'with' statement, stop the agency"""
        self.stop_agency()

    def terminate_agents(self):
        """Terminate all agents modules."""
        logger.info("Terminating all agent modules")
        for agent in self._agents.values():
            agent.terminate()

    def setup_agent(self, id: str) -> Agent:
        """Setup the agent matching the given id"""
        # pylint: disable=redefined-builtin
        agent = Agent(env=self.env, config=self._agent_configs[id])
        self._agents[agent.id] = agent
        return agent

    def get_agent(self, id: str) -> Agent:
        """Get the agent matching the given id"""
        # pylint: disable=redefined-builtin, inconsistent-return-statements
        try:
            return self._agents[id]
        except KeyError:
            KeyError(f"Given id '{id}' is not in the set of agents.")

    def get_results(self, cleanup: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get all results of the agentLogger
        Args:
            cleanup: If true, read files are deleted.

        Returns:
            Dict[str, pd.DataFrame]: key is the agent_id, value the dataframe
        """
        results = {}
        for agent in self._agents.values():
            new_res = agent.get_results(cleanup=cleanup)
            results[agent.id] = new_res
        return results


class LocalCloneMAPAgency(LocalMASAgency):
    """
    Local LocalMASAgency agency class which tries to mimic cloneMAP
    behaviour for the local execution.
    """

    # todo-fwu delete or add to clonemap example. But I dont think we need the threads, since we have simpy

    def run(self, until=None):
        pass  # Already running

    def terminate_agents(self):
        """Terminate all agents modules."""
        logger.info("Can't terminate agents yet in this MAS")

    def setup_agent(self, id: str):
        """Setup the agent matching the given id"""

        # pylint: disable=redefined-builtin
        def _get_ag(env, ag_config):
            ag = Agent(env=Environment(config=env), config=ag_config)
            ag.env.run()
            return ag

        thread = threading.Thread(
            target=_get_ag,
            kwargs={
                "env": self.env.config.model_copy(),
                "ag_config": self._agent_configs[id].copy(),
            },
        )
        thread.start()
        self._agents[id] = thread


def agent_process(
    agent_config: Union[dict, FilePath],
    until: float,
    env: Union[dict, FilePath],
    results_dict: dict,
    cleanup=True,
    log_level=logging.ERROR,
):
    """
    Function to initialize and start an agent in its own process.
    Collects results from the agent and stores them
    in the passed results_dict.
    Args:
        cleanup:
        agent_config: Config for an agent.
        until: Simulation runtime
        env: config for an environment
        results_dict: dict from process manager
        log_level: the log level for this process

    Returns:

    """
    logging.basicConfig(level=log_level)
    env = Environment(config=env)
    agent = Agent(config=agent_config, env=env)
    agent.env.run(until=until)
    results = agent.get_results(cleanup)
    for mod in agent.modules:
        mod.terminate()
    results_dict[agent.id] = results


class MultiProcessingMAS(MAS):
    """
    Helper class to conveniently run multi-agent-systems in separate processes.
    """

    env: Union[dict, FilePath] = Field(
        default_factory=lambda: Environment(config={"rt": True}),
        title="env",
        description="The environment for the agents.",
    )
    cleanup: bool = Field(
        default=False,
        description="Whether agents should clean the results files after " "running.",
    )
    log_level: int = Field(
        default=logging.ERROR, description="Loglevel to set for the processes."
    )

    _processes: List[multiprocessing.Process] = PrivateAttr(default=[])
    _results_dict: Dict[str, pd.DataFrame] = PrivateAttr(default={})

    @field_validator("env")
    @classmethod
    def setup_env(cls, env):
        """Setup the env if a config is given."""
        if isinstance(env, Environment):
            env = env.config.model_dump()
        elif isinstance(env, (Path, str)):
            if Path(env).exists():
                with open(env, "r") as f:
                    env = json.load(f)
        assert env.setdefault("rt", True), (
            "Synchronization between processes relies on time, RealTimeEnvironment "
            "is required."
        )
        return env

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        manager = multiprocessing.Manager()
        self._results_dict = manager.dict()

    def run(self, until):
        """Execute the multi-agent-system in parallel and terminate it after
        run is finished"""
        for agent in self._agent_configs.values():
            kwargs = {
                "agent_config": agent,
                "until": until,
                "env": self.env,
                "results_dict": self._results_dict,
                "cleanup": self.cleanup,
                "log_level": self.log_level,
            }
            process = multiprocessing.Process(
                target=agent_process, name=agent.id, kwargs=kwargs
            )
            self._processes.append(process)
            process.start()
        for process in self._processes:
            process.join()

    def get_results(self) -> Dict[str, pd.DataFrame]:
        """
        Get all results of the agentLogger
        Returns:
            Dict[str, pd.DataFrame]: key is the agent_id, value the dataframe
        """
        return dict(self._results_dict)

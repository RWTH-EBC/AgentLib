"""
This module implements a clonemap compatible communicator
"""

import json
import logging
import os
from functools import cached_property
from typing import Union, List

from pydantic import Field, field_validator

from agentlib.core import Agent, Environment
from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import OptionalDependencyError
from agentlib.modules.communicator.communicator import (
    Communicator,
    SubscriptionCommunicatorConfig,
)
from agentlib.utils.validators import convert_to_list

try:
    import clonemapy.agent as clonemapyagent
    import clonemapy.agency as clonemapyagency
except ImportError as err:
    raise OptionalDependencyError(
        dependency_install="git+https://github.com/sogno-platform/clonemapy",
        used_object="Module type 'clonemap'",
    ) from err


def set_and_get_cmap_config(agent_config: dict, cagent: clonemapyagent.Agent):
    """
    Manipulate the given agent_config to pass the cagent into the config.
    Further, get the settings of log level and env_factor to
    start the agent correctly.

    Args:
        agent_config dict: Agent configuation
        cagent clonemapyagent.Agent: Clonemappy Agent

    Returns:
        agent_config: dict
            Manipulated config
        env_factor: float
            Environment factor in config
        log_level: str
            Log-level of config
    """
    env_factor = 1
    _default_lvl = os.environ.get("CLONEMAP_LOG_LEVEL", "ERROR")
    log_level = _default_lvl
    found_clonemap_module = False
    module_types = []
    for module in agent_config["modules"]:
        _type = module["type"]
        if isinstance(_type, dict):
            module_types.append(_type["class_name"])
            continue
        module_types.append(_type)
        if module["type"] == "clonemap":
            module.update(
                {
                    "cagent": cagent,
                }
            )
            env_factor = module.get("env_factor", 1)
            log_level = module.get("log_level", _default_lvl)
            found_clonemap_module = True
    if not found_clonemap_module:
        from agentlib.core.errors import ConfigurationError

        raise ConfigurationError(
            "Each agents needs a clonemap communicator "
            "module to be executed on clonemap. You passed the modules:"
            f"{' ,'.join(module_types)}"
        )
    return agent_config, env_factor, log_level.upper()


class CloneMAPClientConfig(SubscriptionCommunicatorConfig):
    """
    clonemap communicator settings
    """

    cagent: clonemapyagent.Agent = Field(
        default=None, description="Agent object of CloneMAP"
    )
    subtopics: Union[List[str], str] = Field(
        default=[], description="Topics to that the agent " "subscribes"
    )
    prefix: str = Field(default="/agentlib", description="Prefix for MQTT-Topic")
    env_factor: float = Field(
        default=1, description="Specify Environment Variable Factor"
    )

    # Add validator
    check_subtopics = field_validator("subtopics")(convert_to_list)


class CloneMAPClient(Communicator):
    """
    This communicator implements the communication between agents via clonemap.
    """

    config: CloneMAPClientConfig

    def __init__(self, config: dict, agent: Agent):
        Communicator.__init__(self=self, config=config, agent=agent)
        self._subscribe()
        behavior = self.config.cagent.new_mqtt_default_behavior(self._message_callback)
        behavior.start()
        behavior = self.config.cagent.new_custom_update_behavior(
            self._config_update_callback
        )
        behavior.start()

    @cached_property
    def pubtopic(self):
        """Generate the publication topic"""
        return self.generate_topic(agent_id=self.agent.id, subscription=False)

    def generate_topic(self, agent_id: str, subscription: bool = True):
        """
        Generate the topic with the given agent_id and
        configs prefix
        """
        if subscription:
            topic = "/".join([self.prefix, agent_id, "#"])
        else:
            topic = "/".join([self.prefix, agent_id])
        topic.replace("//", "/")
        return topic

    @property
    def prefix(self):
        """Custom prefix for clonemap.
        For MAS with id 0 and default config it's:
        /mas_0/agentlib
        """
        return "/".join(
            ["", f"mas_{self.config.cagent.masid}", self.config.prefix.strip("/")]
        )

    # The callback for when the client receives a CONNACK response from the server.
    def _subscribe(self):
        topics = set()
        for subscription in self.config.subscriptions:
            topics.add(self.generate_topic(agent_id=subscription))
        topics.update(set(self.config.subtopics))
        for topic in topics:
            self.logger.debug("Subscribing to topic %s", topic)
            self.config.cagent.mqtt.subscribe(topic=topic)

    def _message_callback(self, msg):
        variable = AgentVariable.from_json(msg.payload)
        self.logger.debug(
            "Received variable %s from %s", variable.alias, variable.source
        )
        self.agent.data_broker.send_variable(variable)

    def _send(self, payload: AgentVariable):
        self.logger.debug(
            "Publishing variable %s over mqtt to %s", payload["alias"], self.pubtopic
        )
        self.config.cagent.mqtt.publish(
            topic=self.pubtopic, payload=self.to_json(payload)
        )

    def _config_update_callback(self, new_config: str):
        """Set the new agent config and thus update all modules -
        including this module"""
        self.logger.info("Updating agent config. Payload: %s", new_config)
        new_config = json.loads(new_config)
        new_config, _, _ = set_and_get_cmap_config(
            agent_config=new_config, cagent=self.config.cagent
        )
        self.agent.config = new_config


class CustomLogger(logging.Handler):
    """
    custom logger to route all logs to the cloneMAP logger module
    """

    def __init__(self, cagent: clonemapyagent.Agent):
        logging.Handler.__init__(self)
        self._cagent = cagent

    def emit(self, record):
        msg = record.name + " | " + self.format(record)
        if record.levelname == "ERROR":
            self._cagent.logger.new_log("error", msg, "")
        elif record.levelname == "CRITICAL":
            self._cagent.logger.new_log("error", msg, "")
        elif record.levelname == "WARNING":
            msg = record.name + " | WARNING: " + self.format(record)
            self._cagent.logger.new_log("error", msg, "")
        elif record.levelname == "DEBUG":
            self._cagent.logger.new_log("debug", msg, "")
        else:
            self._cagent.logger.new_log("app", msg, "")


class CloneMAPAgent(clonemapyagent.Agent):
    """
    cloneMAP Agent
    """

    def task(self):
        """
        Method task is executed by the agency for each agent in a separate process
        """
        # get agent config and inject self object
        agent_config = json.loads(self.custom)
        agent_config, env_factor, log_level = set_and_get_cmap_config(
            agent_config=agent_config, cagent=self
        )

        cl = CustomLogger(self)
        logger = logging.getLogger()
        cl.setLevel(logging.DEBUG)
        logger.addHandler(cl)

        env = Environment(config={"rt": True, "factor": env_factor})
        agent = Agent(env=env, config=agent_config)
        env.run()
        self.loop_forever()


if __name__ == "__main__":
    ag = clonemapyagency.Agency(CloneMAPAgent)

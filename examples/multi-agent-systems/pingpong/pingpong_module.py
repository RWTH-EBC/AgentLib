import logging
import time
import sys
from pydantic import Field
from agentlib.core import BaseModule, BaseModuleConfig
from agentlib.core.datamodels import AgentVariable

logger = logging.getLogger(__name__)


class PingPongConfig(BaseModuleConfig):
    start: bool = Field(
        default=False, description="Indicates if the agent should start communication"
    )
    initial_wait: float = Field(
        default=0, description="Wait the given amount of seconds before starting."
    )


class PingPong(BaseModule):
    config: PingPongConfig

    def process(self):
        if self.config.start:
            self.logger.debug("Waiting %s s before starting", self.config.initial_wait)
            yield self.env.timeout(self.config.initial_wait)
            self.logger.debug("Sending first message: %s", self.id)
            self.agent.data_broker.send_variable(
                AgentVariable(
                    name=self.id, value=self.id, source=self.source, shared=True
                )
            )
        yield self.env.event()

    def register_callbacks(self):
        if self.id == "Ping":
            alias = "Pong"
        else:
            alias = "Ping"
        self.agent.data_broker.register_callback(
            alias=alias, source=None, callback=self._callback
        )

    def _callback(self, variable: AgentVariable):
        self.logger.info("Received: %s==%s", variable.name, variable.value)
        sys.stdout.flush()
        time.sleep(1)
        self.agent.data_broker.send_variable(
            AgentVariable(name=self.id, value=self.id, source=self.source, shared=True)
        )

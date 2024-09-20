from agentlib.core.datamodels import AgentVariable
from agentlib.modules.communicator.communicator import (
    CommunicationDict,
    LocalCommunicator,
    LocalCommunicatorConfig,
    SubscriptionCommunicatorConfig,
)
from agentlib.utils import LocalBroker


class LocalSubscriptionCommunicatorConfig(
    LocalCommunicatorConfig, SubscriptionCommunicatorConfig
): ...


class LocalClient(LocalCommunicator):
    """
    This communicator implements the communication between agents via a
    central process broker.
    Note: The broker is implemented as singleton this means that all agents must
    be in the same process!

    """

    config: LocalSubscriptionCommunicatorConfig
    broker: LocalBroker

    def setup_broker(self):
        """Use the LocalBroker"""
        return LocalBroker()

    @property
    def subscriptions(self):
        return self.config.subscriptions

    def _send(self, payload: CommunicationDict):
        if self.config.parse_json:
            self.broker.send(agent_id=payload["source"], message=self.to_json(payload))
        else:
            # we have to create the AgentVariable new here, because we modified the
            # source, and don't want to modify the original variable
            payload["name"] = payload["alias"]
            self.broker.send(
                agent_id=payload["source"], message=AgentVariable(**payload)
            )

from agentlib.core.datamodels import AgentVariable
from agentlib.modules.communicator.communicator import (
    CommunicationDict,
    LocalCommunicatorConfig,
    LocalCommunicator,
)
from agentlib.utils import LocalBroadcastBroker


class LocalBroadcastClient(LocalCommunicator):
    """
    This communicator implements the communication between agents via a
    broadcast broker central process.
    Note: The broker is implemented as singleton. This means that all agents must
    be in the same process!
    """

    broker: LocalBroadcastBroker
    config: LocalCommunicatorConfig

    def setup_broker(self):
        """Use the LocalBroadCastBroker"""
        return LocalBroadcastBroker()

    def _send(self, payload: CommunicationDict):
        if self.config.parse_json:
            self.broker.broadcast(payload["source"], self.to_json(payload))
        else:
            payload["name"] = payload["alias"]
            self.broker.broadcast(payload["source"], AgentVariable(**payload))

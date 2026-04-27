"""
Module containing a LocalBroadcastBroker that
enables local broadcast communication.
"""

from typing import Union

from agentlib.utils.broker import Broker
from agentlib.core import AgentVariable


class DataSpaceBroker(Broker):
    """Broker class which broadcasts messages via Data Space"""

    def broadcast(self, agent_id: str, msg_id: str):
        """Broadcast message object to all agents but itself"""

        # lock is required so the clients loop does not change size during
        # iteration if clients are added or removed
        with self.lock:
            for client in list(self._clients):
                if client.source.agent_id != agent_id:
                    client.receive(msg_id)

"""
Module containing a LocalBroker that
enables local communication with subscriptions.
"""

from typing import Union

from .broker import Broker
from agentlib.core import datamodels


class LocalBroker(Broker):
    """Local broker class which sends messages"""

    def send(self, agent_id: str, message: Union[bytes, datamodels.AgentVariable]):
        """
        Send the given message to all clients if the source
        matches.
        Args:
            agent_id: Source to match
            message: The message to send

        Returns:

        """
        for client in self._clients:
            for sub in client.subscriptions:
                if sub == agent_id:
                    client.receive(message)

"""
Module containing a DataSpaceBroker that
enables broadcast communication via Data Space.
"""

from agentlib.utils.broker import Broker


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

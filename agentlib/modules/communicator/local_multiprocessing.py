import multiprocessing
import threading
import time
from abc import abstractmethod
from ipaddress import IPv4Address

from pydantic import Field, BaseModel

from agentlib.core import Agent
from agentlib.core.datamodels import AgentVariable
from agentlib.modules.communicator.communicator import (
    Communicator,
    CommunicationDict,
    CommunicatorConfig,
    SubscriptionCommunicatorConfig,
)
from agentlib.utils import multi_processing_broker


class MultiProcessingConfigMixin(BaseModel):
    """Mixin containing common multiprocessing configuration fields"""

    ipv4: IPv4Address = Field(
        default="127.0.0.1",
        description="IP Address for the communication server. Defaults to localhost.",
    )
    port: int = Field(
        default=50_000,
        description="Port for setting up the connection with the broker.",
    )
    authkey: bytes = Field(
        default=b"useTheAgentlib",
        description="Authorization key for the connection with the broker.",
    )


class MultiProcessingBroadcastClientConfig(
    CommunicatorConfig, MultiProcessingConfigMixin
):
    """Config for the Multiprocessing Broadcast Communicator"""

    pass


class MultiProcessingSubscriptionClientConfig(
    SubscriptionCommunicatorConfig, MultiProcessingConfigMixin
):
    """Config for the Multiprocessing Subscription Communicator"""

    pass


class MultiProcessingCommunicatorBase(Communicator):
    """
    Base class for multiprocessing communicators.
    Implements the common communication logic between agents via a central process broker.
    """

    config: MultiProcessingConfigMixin

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        manager = multi_processing_broker.BrokerManager(
            address=(self.config.ipv4, self.config.port), authkey=self.config.authkey
        )
        started_wait = time.time()

        while True:
            try:
                manager.connect()
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
            if time.time() - started_wait > 10:
                raise RuntimeError("Could not connect to the server.")

        signup_queue = manager.get_queue()
        client_read, broker_write = multiprocessing.Pipe(duplex=False)
        broker_read, client_write = multiprocessing.Pipe(duplex=False)

        # Create the appropriate client (to be implemented by subclasses)
        signup = self._create_client(broker_read, broker_write)
        signup_queue.put(signup)

        self._client_write = client_write
        self._broker_write = broker_write
        self._client_read = client_read
        self._broker_read = broker_read

        # ensure the broker has set up the connection and sends its ack
        self._client_read.recv()

    @abstractmethod
    def _create_client(self, broker_read, broker_write):
        """
        Create the appropriate client type (broadcast or subscription).
        To be implemented by subclasses.
        """
        pass

    def process(self):
        """Only start the loop once the env is running."""
        _thread = threading.Thread(
            target=self._message_handler, name=str(self.source), daemon=True
        )
        _thread.start()
        self.agent.register_thread(thread=_thread)
        yield self.env.event()

    def _message_handler(self):
        """Reads messages that were put in the message queue."""
        while True:
            try:
                msg = self._client_read.recv()
            except EOFError:
                break
            variable = AgentVariable.from_json(msg)
            self.logger.debug(f"Received variable {variable.alias}.")
            self._handle_received_variable(variable)

    def terminate(self):
        """Closes all of the connections."""
        self._client_write.close()
        self._client_read.close()
        self._broker_write.close()
        self._broker_read.close()
        super().terminate()

    def _send(self, payload: CommunicationDict):
        """Sends a variable to the Broker."""
        agent_id = payload["source"]
        msg = multi_processing_broker.Message(
            agent_id=agent_id, payload=self.to_json(payload)
        )
        self._client_write.send(msg)


class MultiProcessingBroadcastClient(MultiProcessingCommunicatorBase):
    """
    This communicator implements broadcast communication between agents via a
    central process broker.
    """

    config: MultiProcessingBroadcastClientConfig

    def _create_client(self, broker_read, broker_write):
        """Creates a broadcast client."""
        return multi_processing_broker.MPClient(
            agent_id=self.agent.id, read=broker_read, write=broker_write
        )


class MultiProcessingSubscriptionClient(MultiProcessingCommunicatorBase):
    """
    This communicator implements subscription-based communication between agents
    via a central process broker.
    """

    config: MultiProcessingSubscriptionClientConfig

    def _create_client(self, broker_read, broker_write):
        """Creates a subscription client with subscription information."""
        return multi_processing_broker.MPSubscriptionClient(
            agent_id=self.agent.id,
            read=broker_read,
            write=broker_write,
            subscriptions=tuple(self.config.subscriptions),
        )

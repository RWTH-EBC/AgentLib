import time
import multiprocessing
import threading

from pydantic import Field
from ipaddress import IPv4Address

from agentlib.core import Agent
from agentlib.core.datamodels import AgentVariable
from agentlib.modules.communicator.communicator import (
    Communicator,
    CommunicationDict,
    CommunicatorConfig,
)
from agentlib.utils import multi_processing_broker


class MultiProcessingBroadcastClientConfig(CommunicatorConfig):
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


class MultiProcessingBroadcastClient(Communicator):
    """
    This communicator implements the communication between agents via a
    central process broker.
    """

    config: MultiProcessingBroadcastClientConfig

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
        signup = multi_processing_broker.MPClient(
            agent_id=self.agent.id, read=broker_read, write=broker_write
        )

        signup_queue.put(signup)

        self._client_write = client_write
        self._broker_write = broker_write
        self._client_read = client_read
        self._broker_read = broker_read

        # ensure the broker has set up the connection and sends its ack
        self._client_read.recv()

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
            self.logger.debug(f"Received variable {variable.name}.")
            self.agent.data_broker.send_variable(variable)

    def terminate(self):
        """Closes all of the connections."""
        # Terminating is important when running multiple
        # simulations/environments, otherwise the broker will keep spamming all
        # agents from the previous simulation, potentially filling their queues.
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

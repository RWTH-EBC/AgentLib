"""
Module containing a MultiProcessingBroker that
enables communication across different processes.
"""

import json
import logging
import multiprocessing
import threading
import time
from collections import namedtuple
from ipaddress import IPv4Address
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field, FilePath

from .broker import Broker

logger = logging.getLogger(__name__)


MPClient = namedtuple("MPClient", ["agent_id", "read", "write"])
Message = namedtuple("Message", ["agent_id", "payload"])
MPSubscriptionClient = namedtuple(
    "MPSubscriptionClient", ["agent_id", "read", "write", "subscriptions"]
)


class BrokerManager(SyncManager):
    pass


BrokerManager.register("get_queue")


class MultiProcessingBrokerConfig(BaseModel):
    """Class describing the configuration options for the MultiProcessingBroker."""

    ipv4: IPv4Address = Field(
        default="127.0.0.1",
        description="IP Address for the communication server. Defaults to localhost.",
    )
    port: int = Field(
        default=50000, description="Port for setting up the connection with the server."
    )
    authkey: bytes = Field(
        default=b"useTheAgentlib",
        description="Authorization key for the connection with the broker.",
    )


ConfigTypes = Union[MultiProcessingBrokerConfig, dict, str, FilePath]


class MultiProcessingBroker(Broker):
    """
    Singleton which acts as a broker for distributed simulations among multiple
    local processes. Establishes a connection to a multiprocessing.Manager object,
    which defines a queue. This queue is used to receive connection requests from
    local clients. The clients send a Conn object (from multiprocessing.Pipe())
    object through which the connection is established.
    For each connected client, a thread waits for incoming objects.
    """

    def __init__(self, config: ConfigTypes = None):
        super().__init__()
        if config is None:
            self.config = MultiProcessingBrokerConfig()
        else:
            self.config = config

        logger.info(
            f"Starting Multiprocessing Broker on {(self.config.ipv4, self.config.port)}"
        )
        server = multiprocessing.Process(
            target=self._server, name="Broker_Server", args=(self.config,), daemon=True
        )
        server.start()

        signup_handler = threading.Thread(
            target=self._signup_handler, daemon=True, name="Broker_SignUp"
        )
        signup_handler.start()

    @property
    def config(self) -> MultiProcessingBrokerConfig:
        """Return the config of the environment"""
        return self._config

    @config.setter
    def config(self, config: ConfigTypes):
        """Set the config/settings of the environment"""
        if isinstance(config, MultiProcessingBrokerConfig):
            self._config = config
            return
        elif isinstance(config, (str, Path)):
            if Path(config).exists():
                with open(config, "r") as f:
                    config = json.load(f)
        self._config = MultiProcessingBrokerConfig.model_validate(config)

    @staticmethod
    def _server(config: MultiProcessingBrokerConfig):
        """Creates the Manager object which owns the queue and lets it serve forever."""
        from multiprocessing.managers import BaseManager
        from queue import Queue

        queue = Queue()

        class QueueManager(BaseManager):
            pass

        QueueManager.register("get_queue", callable=lambda: queue)
        m = QueueManager(address=(config.ipv4, config.port), authkey=config.authkey)

        s = m.get_server()
        s.serve_forever()

    def _signup_handler(self):
        """Connects to the manager queue and processes the signup requests. Starts a
        child thread listening to messages from each client."""
        from multiprocessing.managers import BaseManager

        class QueueManager(BaseManager):
            pass

        QueueManager.register("get_queue")
        m = QueueManager(
            address=(self.config.ipv4, self.config.port), authkey=self.config.authkey
        )
        started_wait = time.time()
        while True:
            try:
                m.connect()
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
            if time.time() - started_wait > 10:
                raise RuntimeError("Could not connect to server.")

        signup_queue = m.get_queue()

        while True:
            try:
                client = signup_queue.get()
            except ConnectionResetError:
                logger.info("Multiprocessing Broker disconnected.")
                break

            with self.lock:
                self._clients.add(client)

            # send the client an ack its messages are now being received
            client.write.send(1)
            threading.Thread(
                target=self._client_loop,
                args=(client,),
                name=f"MPBroker_{client.agent_id}",
                daemon=True,
            ).start()

    def _client_loop(self, client: MPClient):
        """Receives messages from a client and redistributes them."""
        while True:
            try:
                msg: Message = client.read.recv()
            except EOFError:
                with self.lock:
                    self._clients.remove(client)
                break
            self.send(message=msg.payload, source=msg.agent_id)

    def send(self, source, message):
        """
        Send the given message to all clients if the source
        matches.
        Args:
            source: Source to match
            message: The message to send

        Returns:

        """
        # lock is required so the clients loop does not change size during
        # iteration if clients are added or removed
        with self.lock:
            for client in list(self._clients):
                if client.agent_id != source:
                    try:
                        client.write.send(message)
                    except BrokenPipeError:
                        pass


class MultiProcessingSubscriptionBroker(MultiProcessingBroker):
    """
    Subscription-based multiprocessing broker that only sends messages to
    clients subscribed to the sender's agent_id.
    """

    def _signup_handler(self):
        """Connects to the manager queue and processes the signup requests. Starts a
        child thread listening to messages from each client."""
        from multiprocessing.managers import BaseManager

        class QueueManager(BaseManager):
            pass

        QueueManager.register("get_queue")
        m = QueueManager(
            address=(self.config.ipv4, self.config.port), authkey=self.config.authkey
        )
        started_wait = time.time()
        while True:
            try:
                m.connect()
                break
            except ConnectionRefusedError:
                time.sleep(0.01)
            if time.time() - started_wait > 10:
                raise RuntimeError("Could not connect to server.")

        signup_queue = m.get_queue()

        while True:
            try:
                client = signup_queue.get()
            except ConnectionResetError:
                logger.info("Multiprocessing Subscription Broker disconnected.")
                break

            with self.lock:
                self._clients.add(client)

            # send the client an ack its messages are now being received
            client.write.send(1)
            threading.Thread(
                target=self._client_loop,
                args=(client,),
                name=f"MPSubBroker_{client.agent_id}",
                daemon=True,
            ).start()

    def send(self, source, message):
        """
        Send the given message to subscribed clients only.
        Args:
            source: Source agent_id to match against subscriptions
            message: The message to send
        """
        # lock is required so the clients loop does not change size during
        # iteration if clients are added or removed
        with self.lock:
            for client in list(self._clients):
                # Check if this client is subscribed to the source agent
                if source in client.subscriptions:
                    try:
                        client.write.send(message)
                    except BrokenPipeError:
                        pass

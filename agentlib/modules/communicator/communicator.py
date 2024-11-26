"""
Module contains basics communicator modules
"""

import abc
import json
import queue
import threading
from typing import Union, List, TypedDict, Any

import pandas as pd
from pydantic import Field, field_validator

from agentlib.core import Agent, BaseModule, BaseModuleConfig
from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import OptionalDependencyError
from agentlib.utils.broker import Broker
from agentlib.utils.validators import convert_to_list


class CommunicationDict(TypedDict):
    alias: str
    value: Any
    timestamp: float
    type: str
    source: str


class CommunicatorConfig(BaseModuleConfig):
    use_orjson: bool = Field(
        title="Use orjson",
        default=False,
        description="If true, the faster orjson library will be used for serialization "
        "deserialization. Requires the optional dependency.",
    )


class SubscriptionCommunicatorConfig(CommunicatorConfig):
    subscriptions: Union[List[str], str] = Field(
        title="Subscriptions",
        default=[],
        description="List of agent-id strings to subscribe to",
    )
    check_subscriptions = field_validator("subscriptions")(convert_to_list)


class Communicator(BaseModule):
    """
    Base class for all communicators
    """

    config: CommunicatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)

        if self.config.use_orjson:
            try:
                import orjson
            except ImportError:
                raise OptionalDependencyError(
                    dependency_name="orjson",
                    dependency_install="orjson",
                    used_object="Communicator with 'use_orjson=True'",
                )

            def _to_orjson(payload: CommunicationDict) -> bytes:
                return orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

            self.to_json = _to_orjson
        else:

            def _to_json_builtin(payload: CommunicationDict) -> str:
                return json.dumps(payload)

            self.to_json = _to_json_builtin

    def register_callbacks(self):
        """Register all outputs to the callback function"""
        self.agent.data_broker.register_callback(
            callback=self._send_only_shared_variables, _unsafe_no_copy=True
        )

    def process(self):
        yield self.env.event()

    def _send_only_shared_variables(self, variable: AgentVariable):
        """Send only variables with field ``shared=True``"""
        if not self._variable_can_be_send(variable):
            return

        payload = self.short_dict(variable)
        self.logger.debug("Sending variable %s=%s", variable.alias, variable.value)
        self._send(payload=payload)

    def _variable_can_be_send(self, variable):
        return variable.shared and (
            (variable.source.agent_id is None)
            or (variable.source.agent_id == self.agent.id)
        )

    @abc.abstractmethod
    def _send(self, payload: CommunicationDict):
        raise NotImplementedError(
            "This method needs to be implemented " "individually for each communicator"
        )

    def short_dict(self, variable: AgentVariable) -> CommunicationDict:
        """Creates a short dict serialization of the Variable.

        Only contains attributes of the AgentVariable, that are relevant for other
        modules or agents. For performance and privacy reasons, this function should
        be called for communicators."""
        if isinstance(variable.value, pd.Series):
            value = variable.value.to_json()
        else:
            value = variable.value
        return CommunicationDict(
            alias=variable.alias,
            value=value,
            timestamp=variable.timestamp,
            type=variable.type,
            source=self.agent.id,
        )

    def to_json(self, payload: CommunicationDict) -> Union[bytes, str]:
        """Transforms the payload into json serialized form. Dynamically uses orjson
        if it is installed, and the builtin json otherwise.

        Returns bytes or str depending on the library used, but this has not mattered
        with the communicators as of now.
        """
        # implemented on init
        pass


class LocalCommunicatorConfig(CommunicatorConfig):
    parse_json: bool = Field(
        title="Indicate whether variables are converted to json before sending. "
        "Increasing computing time but makes MAS more close to later stages"
        "which use MQTT or similar.",
        default=False,
    )


class LocalCommunicator(Communicator):
    """
    Base class for local communicators.
    """

    config: LocalCommunicatorConfig

    def __init__(self, config: dict, agent: Agent):
        # assign methods to receive messages either in realtime or in the
        # simpy process. Has to be done before calling super().__init__()
        # because that already calls the process method
        if agent.env.config.rt:
            self.process = self._process_realtime
            self.receive = self._receive_realtime
            self._loop = None
        else:
            self._received_variable = agent.env.event()
            self.process = self._process
            self.receive = self._receive

        super().__init__(config=config, agent=agent)
        self.broker = self.setup_broker()
        self._msg_q_in = queue.Queue(100)
        self.broker.register_client(client=self)

    @property
    def broker(self) -> Broker:
        """Broker used by LocalCommunicator"""
        return self._broker

    @broker.setter
    def broker(self, broker):
        """Set the broker of the LocalCommunicator"""
        self._broker = broker
        self.logger.info("%s uses broker %s", self.__class__.__name__, self.broker)

    @abc.abstractmethod
    def setup_broker(self):
        """Function to set up the broker object.
        Needs to return a valid broker option."""
        raise NotImplementedError(
            "This method needs to be implemented " "individually for each communicator"
        )

    def _process(self):
        """Waits for new messages, sends them to the broker."""
        yield self.env.event()

    def _process_realtime(self):
        """Only start the loop once the env is running."""
        self._loop = threading.Thread(
            target=self._message_handler, name=str(self.source)
        )
        self._loop.daemon = True  # Necessary to enable terminations of scripts
        self._loop.start()
        self.agent.register_thread(thread=self._loop)
        yield self.env.event()

    def _send_simpy(self, ignored):
        """Sends new messages to the broker when receiving them, adhering to the
        simpy event queue. To be appended to a simpy event callback."""
        variable = self._msg_q_in.get_nowait()
        self.agent.data_broker.send_variable(variable)

    def _receive(self, msg_obj):
        """Receive a given message and put it in the queue and set the
        corresponding simpy event."""
        if self.config.parse_json:
            variable = AgentVariable.from_json(msg_obj)
        else:
            variable = msg_obj
        self._msg_q_in.put(variable, block=False)
        self._received_variable.callbacks.append(self._send_simpy)
        self._received_variable.succeed()
        self._received_variable = self.env.event()

    def _receive_realtime(self, msg_obj):
        """Receive a given message and put it in the queue. No event setting
        is required for realtime."""
        if self.config.parse_json:
            variable = AgentVariable.from_json(msg_obj)
        else:
            variable = msg_obj
        self._msg_q_in.put(variable)

    def _message_handler(self):
        """Reads messages that were put in the message queue."""
        while True:
            variable = self._msg_q_in.get()
            self.agent.data_broker.send_variable(variable)

    def terminate(self):
        # Terminating is important when running multiple
        # simulations/environments, otherwise the broker will keep spamming all
        # agents from the previous simulation, potentially filling their queues.
        self.broker.delete_client(self)
        super().terminate()

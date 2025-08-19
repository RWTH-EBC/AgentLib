"""
Module contains basics communicator modules
"""

import abc
import json
import logging
import queue
import threading
from typing import Union, List, TypedDict, Any, Optional, Literal, Tuple

import pandas as pd
from pydantic import Field, field_validator

from agentlib.core import Agent, BaseModule, BaseModuleConfig
from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import OptionalDependencyError
from agentlib.utils.broker import Broker
from agentlib.utils.validators import convert_to_list
from agentlib.modules.communicator.communication_logging_handling import (
    CommunicationLogger,
)

logger = logging.getLogger(__name__)  # Added logger initialization


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
    communication_log_level: Literal["none", "basic", "detail"] = Field(
        default="none",
        title="Communication Log Level",
        description="Level of communication logging: 'none', 'basic', 'detail'.",
    )
    communication_log_file: Optional[str] = Field(
        default=None,
        title="Communication Log File",
        description="Filename for 'detail' logging. Defaults to 'communicator_logs/{agent_id}_{module_id}.jsonl'.",
    )
    communication_log_overwrite: bool = Field(
        default=True,
        title="Overwrite Communication Log",
        description="If true, existing log file will be overwritten at the start.",
    )
    communication_log_t_sample: Union[float, int] = Field(
        default=300,
        title="Communication Log Sampling Time",
        description="Interval in seconds for batch writing 'detail' logs to file. Only for non-realtime.",
        ge=0,
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
    parse_json = True

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)

        # Initialize communication logger
        self._communication_logger = CommunicationLogger(
            config=self.config,
            agent_id=self.agent.id,
            module_id=self.id,
            env=self.env,
        )

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

        # Delegate logging to communication logger
        self._communication_logger.log_sent_message(payload["alias"])

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
        if isinstance(variable.value, pd.Series) and self.parse_json:
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

    def _handle_received_variable(
        self, variable: AgentVariable, remote_agent_id: Optional[str] = None
    ):
        """
        Centralized handler for received variables that manages logging and forwarding.
        """
        source_agent_id = remote_agent_id or variable.source.agent_id

        # Delegate logging to communication logger
        self._communication_logger.log_received_message(variable.alias, source_agent_id)

        # Forward to data broker
        self.agent.data_broker.send_variable(variable)

        self.logger.debug(
            "Received and processed variable %s=%s from source %s",
            variable.alias,
            variable.value,
            source_agent_id,
        )

    def to_json(self, payload: CommunicationDict) -> Union[bytes, str]:
        """Transforms the payload into json serialized form. Dynamically uses orjson
        if it is installed, and the builtin json otherwise.

        Returns bytes or str depending on the library used, but this has not mattered
        with the communicators as of now.
        """
        # implemented on init
        pass

    def terminate(self):
        self._communication_logger.terminate()
        super().terminate()

    def get_results(self) -> Optional[Union[dict, pd.DataFrame]]:
        """Returns logged communication data based on the log level."""
        return self._communication_logger.get_results()

    def get_results_incremental(
        self, update_token: Optional[Any] = None
    ) -> Tuple[Optional[Union[dict, pd.DataFrame]], Optional[Any]]:
        """Returns logged communication data incrementally."""
        return self._communication_logger.get_results_incremental(update_token)

    def cleanup_results(self):
        """Deletes the communication log file if 'detail' logging was active."""
        self._communication_logger.cleanup_results()

    @classmethod
    def visualize_results(
        cls,
        results_data: Optional[Union[dict, pd.DataFrame]],
        module_id: str,
        agent_id: str,
    ) -> "Optional[html.Div]":
        return CommunicationLogger.visualize_results(results_data, module_id, agent_id)


class LocalCommunicatorConfig(CommunicatorConfig):
    parse_json: bool = Field(
        title="Indicate whether variables are converted to json before sending. "
        "Increasing computing time but makes MAS more close to later stages"
        "which use MQTT or similar.",
        default=False,
    )
    queue_size: int = Field(title="Size of the queue", default=10000)


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
        self.parse_json = self.config.parse_json
        self._msg_q_in = queue.Queue(self.config.queue_size)
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
            "This method needs to be implemented individually for each communicator"
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
        self._handle_received_variable(variable)

    def _receive(self, msg_obj):  # msg_obj is raw from broker
        """Receive a given message and put it in the queue and set the
        corresponding simpy event."""
        variable_to_queue = (
            AgentVariable.from_json(msg_obj) if self.config.parse_json else msg_obj
        )
        self._msg_q_in.put(variable_to_queue, block=False)
        self._received_variable.callbacks.append(self._send_simpy)
        self._received_variable.succeed()
        self._received_variable = self.env.event()

    def _receive_realtime(self, msg_obj):  # msg_obj is raw from broker
        """Receive a given message and put it in the queue. No event setting
        is required for realtime."""
        variable_to_queue = (
            AgentVariable.from_json(msg_obj) if self.config.parse_json else msg_obj
        )
        self._msg_q_in.put(variable_to_queue)

    def _message_handler(self):  # Realtime message handler
        """Reads messages that were put in the message queue."""
        while True:
            variable = self._msg_q_in.get()
            self._handle_received_variable(variable)

    def terminate(self):
        self.broker.delete_client(self)
        super().terminate()

"""
Module contains basics communicator modules
"""

import abc
import collections
import json
import os
import queue
import threading
from pathlib import Path
from typing import Union, List, TypedDict, Any, Optional, Literal

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

        # Initialize logging attributes based on the validated config
        self._sent_alias_counts: Optional[collections.Counter] = None
        self._received_source_alias_counts: Optional[collections.Counter] = None
        self._communication_log_filename: Optional[str] = None
        self._communication_log_batch: Optional[List[dict]] = None

        if self.config.communication_log_level == "basic":
            self._sent_alias_counts = collections.Counter()
            self._received_source_alias_counts = collections.Counter()
        elif self.config.communication_log_level == "detail":
            self._init_detail_logging()

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

    def _init_detail_logging(self):
        """Helper to initialize attributes for 'detail' logging."""
        self._communication_log_batch = []
        if self.config.communication_log_file is None:
            logs_dir = Path("communicator_logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._communication_log_filename = str(
                logs_dir / f"{self.agent.id}_{self.id}.jsonl"
            )
        else:
            self._communication_log_filename = self.config.communication_log_file
            # Ensure parent directory exists if a custom path is given
            Path(self._communication_log_filename).parent.mkdir(
                parents=True, exist_ok=True
            )

        if (
            Path(self._communication_log_filename).exists()
            and self.config.communication_log_overwrite
        ):
            try:
                os.remove(self._communication_log_filename)
                self.logger.info(
                    f"Overwriting existing communication log: {self._communication_log_filename}"
                )
            except OSError as e:
                self.logger.error(
                    f"Could not remove communication log file {self._communication_log_filename} for overwrite: {e}"
                )

        # For non-realtime, schedule periodic flushing
        if not self.env.config.rt and self.config.communication_log_t_sample > 0:
            self.env.process(self._log_detail_process())
        elif self.config.communication_log_t_sample <= 0 and not self.env.config.rt:
            self.logger.info(
                "communication_log_t_sample <= 0, detail logs will only be written on terminate."
            )

    def _log_detail_process(self):
        """SimPy process to periodically flush detail logs."""
        while True:
            yield self.env.timeout(self.config.communication_log_t_sample)
            self._flush_detail_log()

    def _flush_detail_log(self):
        """Writes the current batch of detail logs to file."""
        if (
            not self._communication_log_batch
            or self._communication_log_filename is None
        ):
            return

        try:
            with open(self._communication_log_filename, "a", encoding="utf-8") as f:
                for entry in self._communication_log_batch:
                    json.dump(entry, f)
                    f.write("\n")
            self.logger.debug(
                f"Flushed {len(self._communication_log_batch)} entries to {self._communication_log_filename}"
            )
            self._communication_log_batch = []  # Clear batch
        except IOError as e:
            self.logger.error(
                f"Error writing communication detail log to {self._communication_log_filename}: {e}"
            )

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

        # Logging for 'sent' messages
        if self.config.communication_log_level == "basic":
            self._sent_alias_counts[payload["alias"]] += 1
        elif self.config.communication_log_level == "detail":
            log_entry = {
                "timestamp": self.env.time,
                "direction": "sent",
                "own_agent_id": self.agent.id,
                "remote_agent_id": None,  # Cannot be known for sent messages generally
                "alias": payload["alias"],
            }
            self._communication_log_batch.append(log_entry)

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

        Args:
            variable: The received AgentVariable
            remote_agent_id: ID of the sending agent (if known and different from variable.source.agent_id)
        """
        # Use remote_agent_id if provided, otherwise fall back to variable's source
        source_agent_id = remote_agent_id or variable.source.agent_id

        # Handle logging for received messages
        if (
            self.config.communication_log_level == "basic"
            and self._received_source_alias_counts is not None
        ):
            self._received_source_alias_counts[(source_agent_id, variable.alias)] += 1
        elif (
            self.config.communication_log_level == "detail"
            and self._communication_log_batch is not None
        ):
            log_entry = {
                "timestamp": self.env.time,
                "direction": "received",
                "own_agent_id": self.agent.id,
                "remote_agent_id": source_agent_id,
                "alias": variable.alias,
            }
            self._communication_log_batch.append(log_entry)

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
        if self.config.communication_log_level == "detail":
            self.logger.debug(
                f"Terminating communicator {self.id}, flushing any remaining detail logs."
            )
            self._flush_detail_log()
        super().terminate()

    def get_results(self) -> Optional[Union[dict, pd.DataFrame]]:
        """Returns logged communication data based on the log level."""
        log_level = self.config.communication_log_level
        if log_level == "none":
            return None
        elif log_level == "basic":
            return {
                "sent_counts": dict(self._sent_alias_counts or {}),
                "received_counts": {
                    str(k): v
                    for k, v in (self._received_source_alias_counts or {}).items()
                },
            }
        elif log_level == "detail":
            self._flush_detail_log()  # Ensure all data is written before reading
            if (
                self._communication_log_filename
                and Path(self._communication_log_filename).exists()
            ):
                try:
                    log_entries = []
                    with open(
                        self._communication_log_filename, "r", encoding="utf-8"
                    ) as f:
                        for line in f:
                            log_entries.append(json.loads(line))
                    df = pd.DataFrame(log_entries)
                    return df
                except Exception as e:
                    self.logger.error(
                        f"Error loading communication detail log from {self._communication_log_filename}: {e}"
                    )
                    return pd.DataFrame()  # Return empty DataFrame on error
            return pd.DataFrame()  # Return empty DataFrame if file doesn't exist
        return None

    def cleanup_results(self):
        """Deletes the communication log file if 'detail' logging was active and cleanup is configured."""
        try:
            os.remove(self._communication_log_filename)
            self.logger.info(
                f"Cleaned up communication log file: {self._communication_log_filename}"
            )
        except OSError as e:
            self.logger.error(
                f"Error cleaning up communication log file {self._communication_log_filename}: {e}"
            )

    @classmethod
    def visualize_results(
        cls,
        results_data: Optional[Union[dict, pd.DataFrame]],
        module_id: str,
        agent_id: str,
    ) -> "Optional[html.Div]":
        try:
            from dash import dcc, html
            import plotly.graph_objects as go
            import dash_bootstrap_components as dbc
        except ImportError:
            raise OptionalDependencyError(
                used_object="Dashboard",
                dependency_install="dash",
                dependency_name="interactive",
            )
        if results_data is None:
            return html.Div(
                f"No communication data to visualize for {module_id} (Agent: {agent_id}). Log level might be 'none'."
            )

        main_container_children = [
            html.H4(
                f"Communicator Activity: {module_id} (Agent: {agent_id})",
                className="mb-3",
            )
        ]

        if isinstance(results_data, dict):  # Basic logging
            log_type_info = html.P("Log Level: basic (Counters)")
            sent_counts = results_data.get("sent_counts", {})
            received_counts_str_keys = results_data.get("received_counts", {})

            received_counts_display = {
                k: v for k, v in received_counts_str_keys.items()
            }

            if sent_counts:
                sent_fig = go.Figure(
                    data=[
                        go.Bar(x=list(sent_counts.keys()), y=list(sent_counts.values()))
                    ]
                )
                sent_fig.update_layout(
                    title_text="Sent Message Counts by Alias",
                    xaxis_title="Alias",
                    yaxis_title="Count",
                    height=300,
                    margin=dict(l=40, r=20, t=40, b=30),
                )
                main_container_children.append(dcc.Graph(figure=sent_fig))
            else:
                main_container_children.append(
                    html.P(
                        "No messages sent or 'basic' logging did not capture sent messages."
                    )
                )

            if received_counts_display:
                received_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=list(received_counts_display.keys()),
                            y=list(received_counts_display.values()),
                        )
                    ]
                )
                received_fig.update_layout(
                    title_text="Received Message Counts by (Source Agent, Alias)",
                    xaxis_title="(Source Agent, Alias)",
                    yaxis_title="Count",
                    height=300,
                    margin=dict(l=40, r=20, t=40, b=30),
                )
                main_container_children.append(dcc.Graph(figure=received_fig))
            else:
                main_container_children.append(
                    html.P(
                        "No messages received or 'basic' logging did not capture received messages."
                    )
                )

        elif isinstance(results_data, pd.DataFrame):
            log_type_info = html.P("Log Level: detail (Timeline & Aggregates)")
            if results_data.empty:
                main_container_children.append(html.P("Detail log is empty."))
            else:
                # Sent Messages Summary (from DataFrame)
                sent_df = results_data[results_data["direction"] == "sent"]
                if not sent_df.empty:
                    sent_counts_df = (
                        sent_df.groupby("alias").size().reset_index(name="count")
                    )
                    sent_fig_df = go.Figure(
                        data=[
                            go.Bar(x=sent_counts_df["alias"], y=sent_counts_df["count"])
                        ]
                    )
                    sent_fig_df.update_layout(
                        title_text="Sent Message Counts by Alias (from Detail Log)",
                        xaxis_title="Alias",
                        yaxis_title="Count",
                        height=300,
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    main_container_children.append(dcc.Graph(figure=sent_fig_df))
                else:
                    main_container_children.append(
                        html.P("No 'sent' messages in detail log.")
                    )

                # Received Messages Summary (from DataFrame)
                received_df = results_data[results_data["direction"] == "received"]
                if not received_df.empty:
                    # Create a combined key for grouping, e.g., "agent_id | alias"
                    # Ensure remote_agent_id is not None before concatenating
                    received_df_copy = (
                        received_df.copy()
                    )  # Avoid SettingWithCopyWarning
                    received_df_copy["source_alias_key"] = (
                        received_df_copy["remote_agent_id"].fillna("Unknown")
                        + " | "
                        + received_df_copy["alias"]
                    )
                    received_counts_df = (
                        received_df_copy.groupby("source_alias_key")
                        .size()
                        .reset_index(name="count")
                    )
                    received_counts_df = received_counts_df.sort_values(
                        by="source_alias_key"
                    )

                    received_fig_df = go.Figure(
                        data=[
                            go.Bar(
                                x=received_counts_df["source_alias_key"],
                                y=received_counts_df["count"],
                            )
                        ]
                    )
                    received_fig_df.update_layout(
                        title_text="Received Message Counts by (Source Agent | Alias) (from Detail Log)",
                        xaxis_title="(Source Agent | Alias)",
                        yaxis_title="Count",
                        height=300,
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    main_container_children.append(dcc.Graph(figure=received_fig_df))
                else:
                    main_container_children.append(
                        html.P("No 'received' messages in detail log.")
                    )

                # Timelines (at the bottom)
                timeline_children = []
                if not sent_df.empty:
                    sent_timeline_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=sent_df["timestamp"],
                                y=sent_df["alias"],
                                mode="markers",
                                name="Sent",
                            )
                        ]
                    )
                    sent_timeline_fig.update_layout(
                        title_text="Sent Messages Timeline",
                        xaxis_title="Time",
                        yaxis_title="Alias",
                        height=350,
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    timeline_children.append(
                        dbc.Col(dcc.Graph(figure=sent_timeline_fig), md=6)
                    )

                if not received_df.empty:
                    received_timeline_fig = go.Figure()
                    # Group by remote_agent_id, handling potential None values
                    for r_agent_id, group in received_df.fillna(
                        {"remote_agent_id": "Unknown"}
                    ).groupby("remote_agent_id"):
                        received_timeline_fig.add_trace(
                            go.Scatter(
                                x=group["timestamp"],
                                y=group["alias"],
                                mode="markers",
                                name=f"From {r_agent_id}",
                                hovertext=[
                                    f"From: {ra} Alias: {al}"
                                    for ra, al in zip(
                                        group["remote_agent_id"], group["alias"]
                                    )
                                ],
                            )
                        )
                    received_timeline_fig.update_layout(
                        title_text="Received Messages Timeline",
                        xaxis_title="Time",
                        yaxis_title="Alias",
                        height=350,
                        margin=dict(l=40, r=20, t=40, b=30),
                    )
                    timeline_children.append(
                        dbc.Col(dcc.Graph(figure=received_timeline_fig), md=6)
                    )

                if timeline_children:
                    main_container_children.append(html.Hr())
                    main_container_children.append(
                        html.H5("Message Timelines (Detail Log)", className="mt-4 mb-3")
                    )
                    main_container_children.append(dbc.Row(timeline_children))

        else:
            return html.Div(
                f"Unknown results data type for {module_id} (Agent: {agent_id}): {type(results_data)}"
            )

        main_container_children.insert(
            1, log_type_info
        )  # Insert log type info after H4

        return html.Div(main_container_children, style={"padding": "10px"})


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

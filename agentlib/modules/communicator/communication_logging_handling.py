"""
Communication logging and result handling functionality
"""

import collections
import json
import logging
import os
from pathlib import Path
from typing import Union, List, Any, Optional, Tuple, Callable

import pandas as pd

from agentlib.core.errors import OptionalDependencyError

logger = logging.getLogger(__name__)


class CommunicationLogger:
    """Handles all communication logging and result processing"""

    def __init__(self, config, agent_id: str, module_id: str, env):
        self.config = config
        self.agent_id = agent_id
        self.module_id = module_id
        self.env = env
        self.logger = logging.getLogger(f"{__name__}.{agent_id}.{module_id}")

        # Initialize attributes that might be used
        self._sent_alias_counts: Optional[collections.Counter] = None
        self._received_source_alias_counts: Optional[collections.Counter] = None
        self._communication_log_filename: Optional[str] = None
        self._communication_log_batch: Optional[List[dict]] = None

        # Set up logging methods based on level
        self._setup_logging_strategy()

    def _setup_logging_strategy(self):
        """Set up the appropriate logging methods based on config level"""
        log_level = self.config.communication_log_level

        if log_level == "none":
            self.log_sent_message = self._log_sent_none
            self.log_received_message = self._log_received_none
        elif log_level == "basic":
            self._init_basic_logging()
            self.log_sent_message = self._log_sent_basic
            self.log_received_message = self._log_received_basic
        elif log_level == "detail":
            self._init_detail_logging()
            self.log_sent_message = self._log_sent_detail
            self.log_received_message = self._log_received_detail

    def _init_basic_logging(self):
        """Initialize basic logging with counters"""
        self._sent_alias_counts = collections.Counter()
        self._received_source_alias_counts = collections.Counter()

    def _init_detail_logging(self):
        """Initialize detail logging with file handling"""
        self._communication_log_batch = []

        if self.config.communication_log_file is None:
            logs_dir = Path("communicator_logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._communication_log_filename = str(
                logs_dir / f"{self.agent_id}_{self.module_id}.jsonl"
            )
        else:
            self._communication_log_filename = self.config.communication_log_file
            # Ensure parent directory exists if a custom path is given
            Path(self._communication_log_filename).parent.mkdir(
                parents=True, exist_ok=True
            )

        if (
            self._communication_log_filename is not None
            and Path(self._communication_log_filename).exists()
            and self.config.communication_log_overwrite
        ):
            os.remove(self._communication_log_filename)
            self.logger.info(
                f"Overwriting existing communication log: {self._communication_log_filename}"
            )

        # For non-realtime, schedule periodic flushing
        if not self.env.config.rt and self.config.communication_log_t_sample > 0:
            self.env.process(self._log_detail_process())
        elif self.config.communication_log_t_sample <= 0 and not self.env.config.rt:
            self.logger.info(
                "communication_log_t_sample <= 0, detail logs will only be written on terminate."
            )

    # No-op logging methods for "none" level
    def _log_sent_none(self, alias: str):
        """No-op logging for sent messages"""
        pass

    def _log_received_none(self, alias: str, source_agent_id: Optional[str]):
        """No-op logging for received messages"""
        pass

    # Basic logging methods
    def _log_sent_basic(self, alias: str):
        """Log sent message for basic level"""
        self._sent_alias_counts[alias] += 1

    def _log_received_basic(self, alias: str, source_agent_id: Optional[str]):
        """Log received message for basic level"""
        self._received_source_alias_counts[(source_agent_id, alias)] += 1

    # Detail logging methods
    def _log_sent_detail(self, alias: str):
        """Log sent message for detail level"""
        log_entry = {
            "timestamp": self.env.time,
            "direction": "sent",
            "own_agent_id": self.agent_id,
            "remote_agent_id": None,  # Cannot be known for sent messages generally
            "alias": alias,
        }
        self._communication_log_batch.append(log_entry)

    def _log_received_detail(self, alias: str, source_agent_id: Optional[str]):
        """Log received message for detail level"""
        log_entry = {
            "timestamp": self.env.time,
            "direction": "received",
            "own_agent_id": self.agent_id,
            "remote_agent_id": source_agent_id,
            "alias": alias,
        }
        self._communication_log_batch.append(log_entry)

    # These methods will be assigned during init
    def log_sent_message(self, alias: str):
        """Log a sent message - implementation assigned during init"""
        pass

    def log_received_message(self, alias: str, source_agent_id: Optional[str]):
        """Log a received message - implementation assigned during init"""
        pass

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

        with open(self._communication_log_filename, "a", encoding="utf-8") as f:
            for entry in self._communication_log_batch:
                json.dump(entry, f)
                f.write("\n")
        self.logger.debug(
            f"Flushed {len(self._communication_log_batch)} entries to {self._communication_log_filename}"
        )
        self._communication_log_batch = []  # Clear batch

    def terminate(self):
        """Handle termination cleanup"""
        if self.config.communication_log_level == "detail":
            self.logger.debug(
                f"Terminating communicator {self.module_id}, flushing any remaining detail logs."
            )
            self._flush_detail_log()

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
            log_entries = []
            with open(self._communication_log_filename, "r", encoding="utf-8") as f:
                for line in f:
                    log_entries.append(json.loads(line))
            if not log_entries:  # Handle empty log file
                return pd.DataFrame()
            df = pd.DataFrame(log_entries)
            return df
        return None

    def get_results_incremental(
        self, update_token: Optional[Any] = None
    ) -> Tuple[Optional[Union[dict, pd.DataFrame]], Optional[Any]]:
        """Returns logged communication data incrementally."""
        log_level = self.config.communication_log_level

        if log_level == "none":
            return None, None
        elif log_level == "basic":
            current_counts = {
                "sent_counts": dict(self._sent_alias_counts or {}),
                "received_counts": {
                    str(k): v
                    for k, v in (self._received_source_alias_counts or {}).items()
                },
            }
            if update_token is None or update_token != current_counts:
                return current_counts, current_counts
            return None, update_token
        elif log_level == "detail":
            self._flush_detail_log()

            if update_token is None:  # Initial call
                df = self.get_results()
                if df is None or df.empty:
                    return pd.DataFrame(), 0
                with open(self._communication_log_filename, "r", encoding="utf-8") as f:
                    lines_in_file = sum(1 for _ in f)
                return df, lines_in_file
            else:  # Incremental call
                df_chunk, lines_read = self._load_detail_log_incremental(
                    filename=self._communication_log_filename,
                    start_line_index=update_token,
                )
                if df_chunk is None or df_chunk.empty:
                    return None, update_token
                return df_chunk, update_token + lines_read

    @classmethod
    def _load_detail_log_incremental(
        cls, filename: str, start_line_index: int
    ) -> Tuple[Optional[pd.DataFrame], int]:
        """Loads detail log file from a specific line index."""
        log_entries = []
        lines_read_count = 0

        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_line_index:
                    continue
                log_entries.append(json.loads(line))
                lines_read_count += 1

        if not log_entries:
            return None, 0

        df = pd.DataFrame(log_entries)
        return df, lines_read_count

    def cleanup_results(self):
        """Deletes the communication log file if 'detail' logging was active."""
        if (
            self.config.communication_log_level == "detail"
            and self._communication_log_filename
        ):
            try:
                if Path(self._communication_log_filename).exists():
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

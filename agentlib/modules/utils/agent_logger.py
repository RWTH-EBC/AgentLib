"""This module contains a custom Module to log
all variables inside an agent's data_broker."""

import collections
import json
import logging
import os
from ast import literal_eval
from pathlib import Path
from typing import Union, Optional, Tuple, TYPE_CHECKING

import pandas as pd
from pydantic import Field

from agentlib import AgentVariable
from agentlib.core import BaseModule, Agent, BaseModuleConfig
from agentlib.core.errors import OptionalDependencyError

if TYPE_CHECKING:
    from dash import html


logger = logging.getLogger(__name__)


class AgentLoggerConfig(BaseModuleConfig):
    """Define parameters for the AgentLogger"""

    t_sample: Union[float, int] = Field(
        title="t_sample",
        default=300,
        description="The log is saved every other t_sample seconds.",
    )
    values_only: bool = Field(
        title="values_only",
        default=True,
        description="If True, only the values are logged. Else, all"
        "fields in the AgentVariable are logged.",
    )
    clean_up: bool = Field(
        title="clean_up",
        default=True,
        description="If True, file is deleted once load_log is called.",
    )
    overwrite_log: bool = Field(
        title="Overwrite file",
        default=False,
        description="If true, old logs are auto deleted when a new log should be written with that name.",
    )
    filename: Optional[str] = Field(
        title="filename",
        default=None,
        description="The filename where the log is stored. If None, will use 'agent_logs/{agent_id}_log.jsonl'",
    )
    merge_sources: bool = Field(
        title="Merge Sources",
        default=True,
        description="When loading the results file, automatically merges variables by sources, leaving only alias in the columns.",
    )


class AgentLogger(BaseModule):
    """
    A custom logger for Agents to write variables
    which are updated in data_broker into a file.
    """

    config: AgentLoggerConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)

        # If filename is None, create a custom one using the agent ID
        self._filename = self._setup_log_file()

        self._variables_to_log = {}
        if not self.env.config.rt and self.config.t_sample < 60:
            self.logger.warning(
                "Sampling time of agent_logger %s is very low %s. This can hinder "
                "performance.",
                self.id,
                self.config.t_sample,
            )

    def _setup_log_file(self) -> str:
        """Centralized file setup logic"""
        # Determine the target filename
        if self.config.filename is None:
            logs_dir = Path("agent_logs")
            target_file = logs_dir / f"{self.agent.id}.jsonl"
        else:
            target_file = Path(self.config.filename)

        # Handle file existence and overwrite logic
        return self._handle_file_existence(target_file)

    def _handle_file_existence(self, file_path: Path) -> str:
        """Consistent file existence and overwrite handling"""
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle existing file
        if file_path.exists():
            if self.config.overwrite_log:
                file_path.unlink()
                self.logger.info(f"Overwrote existing log file: {file_path}")
            else:
                raise FileExistsError(
                    f"Log file '{file_path}' already exists. "
                    f"Set 'overwrite_log=True' to enable automatic overwrite."
                )

        return str(file_path)

    @property
    def filename(self):
        """Return the filename where to log."""
        return self._filename

    def process(self):
        """Calls the logger every other t_sample
        is used."""
        while True:
            self._log()
            yield self.env.timeout(self.config.t_sample)

    def register_callbacks(self):
        """Callbacks trigger the log_cache function"""
        callback = (
            self._callback_values if self.config.values_only else self._callback_full
        )
        self.agent.data_broker.register_callback(
            alias=None, source=None, callback=callback
        )

    def _callback_values(self, variable: AgentVariable):
        """Save variable values to log later."""
        if not isinstance(variable.value, (float, int, str)):
            return
        current_time = self._variables_to_log.setdefault(str(self.env.time), {})
        # we merge alias and source tuple into a string so we can .json it
        current_time[str((variable.alias, str(variable.source)))] = variable.value

    def _callback_full(self, variable: AgentVariable):
        """Save full variable to log later."""
        current_time = self._variables_to_log.setdefault(str(self.env.time), {})
        current_time[str((variable.alias, str(variable.source)))] = variable.dict()

    def _log(self):
        """Writes the currently in memory saved values to file"""
        _variables_to_log = self._variables_to_log
        self._variables_to_log = {}
        with open(self.filename, "a") as file:
            json.dump(_variables_to_log, file)
            file.write("\n")

    @classmethod
    def load_from_file(
        cls, filename: str, values_only: bool = True, merge_sources: bool = True
    ) -> pd.DataFrame:
        """Loads the log file and consolidates it as a pandas DataFrame.

        Args:
            filename: The file to load
            values_only: If true, loads a file that only has values saved (default True)
            merge_sources: When there are variables with the same alias from multiple
                sources, they are saved in different columns. For backwards
                compatibility, they are merged into a single column. However, if you
                specify False for this parameter, you can view them separately,
                resulting in a multi-indexed return column index

        """
        chunks = []
        with open(filename, "r") as file:
            for data_line in file.readlines():
                chunks.append(json.loads(data_line))
        if not any(chunks):
            return pd.DataFrame()
        full_dict = collections.ChainMap(*chunks)
        df = pd.DataFrame.from_dict(full_dict, orient="index")
        df.index = df.index.astype(float)
        columns = (literal_eval(column) for column in df.columns)
        df.columns = pd.MultiIndex.from_tuples(columns)

        if not values_only:

            def _load_agent_variable(var):
                try:
                    return AgentVariable.validate_data(var)
                except TypeError:
                    pass

            df = df.applymap(_load_agent_variable)

        if merge_sources:
            df = df.droplevel(1, axis=1)
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        return df.sort_index()

    def get_results(self) -> pd.DataFrame:
        """Load the own filename"""
        # Ensure current in-memory logs are flushed before reading for static results
        self._log()
        return self.load_from_file(
            filename=self.filename,
            values_only=self.config.values_only,
            merge_sources=self.config.merge_sources,
        )

    def cleanup_results(self):
        """Deletes the log if wanted."""
        if self.config.clean_up:
            try:
                os.remove(self.filename)
            except OSError:
                self.logger.error(
                    "Could not delete filename %s. Please delete it yourself.",
                    self.filename,
                )

    def terminate(self):
        # when terminating, we log one last time, since otherwise the data since the
        # last log interval is lost
        self._log()

    def get_results_incremental(
        self, update_token: Optional[int] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
        """Fetches results incrementally for live dashboard."""
        self._log()  # Ensure current logs are written

        if update_token is None:  # Initial call
            df = self.load_from_file(
                filename=self.filename,
                values_only=self.config.values_only,
                merge_sources=self.config.merge_sources,
            )
            try:
                with open(self.filename, "r") as f:
                    lines_in_file = sum(1 for _ in f)
            except FileNotFoundError:
                lines_in_file = 0
            return df, lines_in_file
        else:  # Incremental call
            df_chunk, lines_read = self.load_from_file_incremental(
                filename=self.filename,
                start_line_index=update_token,
                values_only=self.config.values_only,
                merge_sources=self.config.merge_sources,
            )
            if df_chunk is None or df_chunk.empty:
                return None, update_token
            return df_chunk, update_token + lines_read

    @classmethod
    def load_from_file_incremental(
        cls,
        filename: str,
        start_line_index: int,
        values_only: bool = True,
        merge_sources: bool = True,
    ) -> Tuple[Optional[pd.DataFrame], int]:
        """Loads log file from a specific line index."""
        chunks = []
        lines_read_count = 0

        try:
            with open(filename, "r") as file:
                for i, data_line in enumerate(file):
                    if i < start_line_index:
                        continue
                    chunks.append(json.loads(data_line))
                    lines_read_count += 1
        except FileNotFoundError:
            return None, 0

        if not any(chunks):
            return None, 0

        full_dict = collections.ChainMap(*chunks)
        df = pd.DataFrame.from_dict(full_dict, orient="index")
        df.index = df.index.astype(float)
        columns = (literal_eval(column) for column in df.columns)
        df.columns = pd.MultiIndex.from_tuples(columns)

        if not values_only:

            def _load_agent_variable(var):
                try:
                    return AgentVariable.validate_data(var)
                except TypeError:
                    pass

            df = df.applymap(_load_agent_variable)

        if merge_sources:
            df = df.droplevel(1, axis=1)
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        return df.sort_index(), lines_read_count

    @classmethod
    def visualize_results(
        cls, results_data: pd.DataFrame, module_id: str, agent_id: str
    ) -> "Optional[html.Div]":
        try:
            from dash import dcc, html
            import plotly.graph_objs as go
            import dash_bootstrap_components as dbc  # Added for responsive layout
        except ImportError:
            raise OptionalDependencyError(
                used_object=f"{cls.__name__}.visualize_results",
                dependency_install="agentlib[interactive]",
                dependency_name="interactive",
            )

        if results_data is None or results_data.empty:
            raise ValueError(
                f"No results data for AgentLogger '{module_id}' in agent '{agent_id}'."
            )
            return None

        rows = []
        current_row_children = []
        for i, col_name in enumerate(results_data.columns):
            series = results_data[col_name].dropna()
            if series.empty:
                continue

            try:
                numeric_series = pd.to_numeric(series, errors="coerce")
                if numeric_series.isnull().all() and not series.isnull().all():
                    is_numeric = False
                else:
                    is_numeric = True
                    series_to_plot = numeric_series
            except (ValueError, TypeError):
                is_numeric = False

            if not is_numeric:
                series_to_plot = series

            fig = go.Figure()
            y_axis_title = "Value"

            if is_numeric:
                fig.add_trace(
                    go.Scatter(
                        x=series_to_plot.index,
                        y=series_to_plot,
                        mode="lines+markers",
                        name=str(col_name),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=series_to_plot.index,
                        y=[1] * len(series_to_plot),
                        mode="markers",
                        name=str(col_name),
                        hovertext=[str(v) for v in series_to_plot.values],
                        hoverinfo="x+text",
                    )
                )
                y_axis_title = "Occurrence"

            title_str = str(col_name)
            if isinstance(col_name, tuple) and len(col_name) > 0:
                title_str = f"{col_name[0]}"
                if len(col_name) > 1 and col_name[1]:
                    title_str += f" (Source: {col_name[1]})"

            fig.update_layout(
                title=title_str,
                xaxis_title="Time",
                yaxis_title=y_axis_title,
                margin=dict(l=40, r=20, t=40, b=30),
                height=250,
            )
            current_row_children.append(dbc.Col(dcc.Graph(figure=fig), md=6))

            if len(current_row_children) == 2 or i == len(results_data.columns) - 1:
                rows.append(dbc.Row(current_row_children, className="mb-3"))
                current_row_children = []

        if not rows:
            raise ValueError(
                f"No plottable data generated for AgentLogger '{module_id}'."
            )

        return html.Div(
            children=[
                html.H4(
                    f"AgentLogger Results: {module_id} (Agent: {agent_id})",
                    className="mb-3",
                )
            ]
            + rows,
            style={"padding": "10px"},
        )

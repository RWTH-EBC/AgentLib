"""This module contains a custom Module to log
all variables inside an agent's data_broker."""

import collections
import json
import logging
import os
from ast import literal_eval
from typing import Union

import pandas as pd
from pydantic import field_validator, Field
from pydantic_core.core_schema import FieldValidationInfo

from agentlib import AgentVariable
from agentlib.core import BaseModule, Agent, BaseModuleConfig

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
        description="If true, old logs are auto deleted when a new log should be written with that name."
    )
    filename: str = Field(
        title="filename",
        description="The filename where the log is stored.",
    )

    @field_validator("filename")
    @classmethod
    def check_existence_of_file(cls, filename, info: FieldValidationInfo):
        """Checks whether the file already exists."""
        # pylint: disable=no-self-argument,no-self-use
        if os.path.isfile(filename):
            # remove result file, so a new one can be created
            if info.data["overwrite_log"]:
                os.remove(filename)
                return filename
            raise FileExistsError(
                f"Given filename at {filename} "
                f"already exists. We won't overwrite it automatically. "
                f"You can use the key word 'overwrite_log' to "
                f"activate automatic overwrite."
            )
        # Create path in case it does not exist
        fpath = os.path.dirname(filename)
        if fpath:
            os.makedirs(fpath, exist_ok=True)
        return filename


class AgentLogger(BaseModule):
    """
    A custom logger for Agents to write variables
    which are updated in data_broker into a file.
    """

    config: AgentLoggerConfig

    def __init__(self, *, config: dict, agent: Agent):
        """Overwrite init to enable a custom default filename
        which uses the agent_id."""
        super().__init__(config=config, agent=agent)
        self._filename = self.config.filename
        self._variables_to_log = {}
        if not self.env.config.rt and self.config.t_sample < 60:
            self.logger.warning(
                "Sampling time of agent_logger %s is very low %s. This can hinder "
                "performance.",
                self.id,
                self.config.t_sample,
            )

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
        return self.load_from_file(
            filename=self.filename, values_only=self.config.values_only
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

from pathlib import Path
from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, FilePath, model_validator

from agentlib import Agent
from agentlib.core import BaseModule, BaseModuleConfig, AgentVariable


class CSVDataSourceConfig(BaseModuleConfig):
    data: Union[pd.DataFrame, FilePath] = Field(
        title="Data",
        default=pd.DataFrame(),
        description="Data that should be communicated during execution. Index should "
        "be either numeric or Datetime, numeric values are interpreted as"
        " seconds.",
        validate_default=True,
    )
    outputs: List[AgentVariable] = Field(
        title="Outputs",
        default=[],
        description="Optional list of columns of data frame that should be sent. If "
        "ommited, all datapoint in frame are sent.",
    )
    t_sample: Union[float, int] = Field(
        title="t_sample",
        default=1,
        ge=0,
        description="Sampling time. Data source sends an interpolated value from the "
        "data every <t_sample> seconds. Default is 1 s.",
    )
    data_offset: Optional[Union[pd.Timedelta, float]] = Field(
        title="data_offset",
        default=0,
        description="Offset will be subtracted from index, allowing you to start at "
        "any point in your data. I.e. if your environment starts at 0, "
        "and you want your data-source to start at 1000 seconds, "
        "you should set this to 1000.",
    )
    extrapolation: Literal["constant", "repeat", "backwards"] = Field(
        title="Extrapolation",
        default="constant",
        description="Determines what to do, when the data source runs out. 'constant' "
        "returns the last value, 'repeat' repeats the data from the "
        "start, and 'backwards' goes through the data backwards, bouncing "
        "indefinitely.",
    )
    shared_variable_fields: List[str] = ["outputs"]

    @field_validator("data")
    @classmethod
    def check_data(cls, data):
        """Makes sure data is a data frame, and loads it if required."""
        if isinstance(data, (str, Path)) and Path(data).is_file():
            data = pd.read_csv(data, engine="python", index_col=0)
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"Data {data} is not a valid DataFrame or the path is not found."
            )
        if data.empty:
            raise ValueError("Provided data is empty.")
        if len(data) < 2:
            raise ValueError(
                "The dataframe must contain at least two rows for interpolation."
            )
        return data

    def transform_to_numeric_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handles the index and ensures it is numeric, with correct offset"""
        # Convert offset to seconds if it's a Timedelta
        offset = self.data_offset
        if isinstance(offset, pd.Timedelta):
            offset = offset.total_seconds()
        # Handle different index types
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = (data.index - data.index[0]).total_seconds()
        else:
            # Try to convert to numeric if it's a string
            try:
                data.index = pd.to_numeric(data.index)
                data.index = data.index - data.index[0]
            except ValueError:
                # If conversion to numeric fails, try to convert to datetime
                try:
                    data.index = pd.to_datetime(data.index)
                    data.index = (data.index - data.index[0]).total_seconds()
                except ValueError:
                    raise ValueError("Unable to convert index to numeric format")

        data.index = data.index.astype(float) - offset

    @model_validator(mode="after")
    def validate_data(self):
        """Checks if outputs and data columns match, and ensures a numeric index."""
        if self.outputs:
            columns = set(self.data.columns)
            output_names = set(o.name for o in self.outputs)

            missing_columns = output_names - columns
            if missing_columns:
                raise ValueError(
                    f"The following output columns are not present in the dataframe: "
                    f"{', '.join(missing_columns)}"
                )
        self.transform_to_numeric_index(self.data)
        return self


class CSVDataSource(BaseModule):

    config: CSVDataSourceConfig

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)

        data = self.config.data

        # Interpolate the dataframe
        start_time = data.index[0]
        end_time = data.index[-1]
        new_index = np.arange(
            start_time, end_time + self.config.t_sample, self.config.t_sample
        )
        interpolated_data = (
            data.reindex(data.index.union(new_index))
            .interpolate(method="index")
            .loc[new_index]
        )

        # Transform to list of tuples
        self.data_tuples = list(interpolated_data.itertuples(index=False, name=None))
        self.data_iterator = self.create_iterator()

    def _get_next_data(self):
        """Yield the next data point"""
        data = next(self.data_iterator)
        return data

    def create_iterator(self):
        """Create a custom iterator based on the extrapolation method"""
        while True:
            for item in self.data_tuples:
                yield item

            if self.config.extrapolation == "constant":
                self.logger.warning(
                    "Data source has been exhausted. Returning last value indefinitely."
                )
                while True:
                    yield self.data_tuples[-1]
            elif self.config.extrapolation == "repeat":
                self.logger.warning(
                    "Data source has been exhausted. Repeating data from the start."
                )
                continue  # This will restart the outer loop
            elif self.config.extrapolation == "backwards":
                self.logger.warning(
                    "Data source has been exhausted. Going through data backwards."
                )
                yield from self.backwards_iterator()

    def backwards_iterator(self):
        """Iterator for backwards extrapolation"""
        while True:
            for item in reversed(
                self.data_tuples[:-1]
            ):  # Exclude the last item to avoid repetition
                yield item
            for item in self.data_tuples[
                1:
            ]:  # Exclude the first item to avoid repetition
                yield item

    def process(self):
        """Write the current data values into data_broker every t_sample"""
        while True:
            current_data = self._get_next_data()
            for output, value in zip(self.config.outputs, current_data):
                self.logger.debug(
                    f"At {self.env.time}: Sending variable {output.name} with value {value} to data broker."
                )
                self.set(output.name, value)
            yield self.env.timeout(self.config.t_sample)

    def register_callbacks(self):
        """Don't do anything as this module is not event-triggered"""

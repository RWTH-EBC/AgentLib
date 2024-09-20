"""This module contains a SensorModule which reads a .dat file
from the Deutsche Wetterdienst (DWD)."""

import io
from typing import Union, List
from pydantic import FilePath, Field, validator
import numpy as np
import pandas as pd
from agentlib.core import (
    BaseModule,
    Agent,
    AgentVariable,
    BaseModuleConfig,
    AgentVariables,
)


class TRYSensorConfig(BaseModuleConfig):
    """Define parameters for the TRYSensor"""

    outputs: AgentVariables = [
        AgentVariable(
            name="T_oda", unit="K", description="Air temperature 2m over ground [K]"
        ),
        AgentVariable(
            name="pressure",
            unit="hPa",
            description="Air pressure in standard height [hPa]",
        ),
        AgentVariable(
            name="wind_direction",
            unit="°",
            description="Wind direction 10 m above gorund " "[Grad] {0..360;999}",
        ),
        AgentVariable(
            name="wind_speed",
            unit="m/s",
            description="Wind speed 10 m above ground [m/s]",
        ),
        AgentVariable(name="coverage", unit="eighth", description="[eighth]  {0..8;9}"),
        AgentVariable(name="absolute_humidity", unit="g/kg", description="[g/kg]"),
        AgentVariable(
            name="relative_humidity",
            unit="%",
            description="Relative humidity 2 m above ground " "[%] {1..100}",
        ),
        AgentVariable(
            name="beam_direct",
            unit="W/m^2",
            description="Direct beam of sun (hor. plane) "
            "[W/m^2] downwards: positive",
        ),
        AgentVariable(
            name="beam_diffuse",
            unit="/m^2",
            description="Diffuse beam of sun (hor. plane) "
            "[W/m^2] downwards: positive",
        ),
        AgentVariable(
            name="beam_atm",
            unit="/m^2",
            description="Beam of atmospheric heat (hor. plane) "
            "[W/m^2] downwards: positive",
        ),
        AgentVariable(
            name="beam_terr",
            unit="/m^2",
            description="Beam of terrestrial heat " "[W/m^2] upwards: negative",
        ),
    ]

    filename: FilePath = Field(
        title="filename", description="The filepath to the data."
    )
    t_sample: Union[float, int] = Field(
        title="t_sample",
        default=1,
        description="Sample time of sensor. As TRY are hourly, " "default is 1 h",
    )
    shared_variable_fields: List[str] = ["outputs"]


class TRYSensor(BaseModule):
    """A module that emulates a sensor using the TRY
    data of the DWD.
    The module casts the following AgentVariables into the data_broker:

    - T_oda:               Air temperature 2m over ground [K]
    - pressure:            Air pressure in standard height [hPa]
    - wind_direction:      Wind direction 10 m above gorund [Grad] {0..360;999}
    - wind_speed:          Wind speed 10 m above ground [m/s]
    - coverage:            [eighth]  {0..8;9}
    - absolute_humidity:   [g/kg]
    - relative_humidity:   Relative humidity 2 m above ground [%] {1..100}
    - beam_direct:         Direct beam of sun (hor. plane) [W/m^2]   downwards: positive
    - beam_diffuse:        Diffuse beam of sun (hor. plane) [W/m^2] downwards: positive
    - beam_atm:            Beam of atmospheric heat (hor. plane)    [W/m^2] downwards: positive
    - beam_terr:           Beam of terrestrial heat [W/m^2] upwards: negative
    """

    config: TRYSensorConfig

    def __init__(self, *, config: dict, agent: Agent):
        """Overwrite init to enable a custom default filename
        which uses the agent_id."""
        super().__init__(config=config, agent=agent)
        self.one_year = 86400 * 365
        self._data = read_dat_file(self.filename)
        # Resample and interpolate once according to t_sample:
        self._data = self._data.reindex(np.arange(0, self.one_year, self.t_sample))
        self._data = self._data.interpolate()
        # Check if outputs match the _data:
        _names = [v.name for v in self.variables]
        if set(self._data.columns).difference(_names):
            raise KeyError(
                "The internal variables differ from the "
                "supported data in a TRY dataset."
            )

    @property
    def filename(self):
        """Return the filename."""
        return self.config.filename

    @property
    def t_sample(self):
        """Return the sample rate."""
        return self.config.t_sample

    def process(self):
        """Write the current TRY values into data_broker every other t_sample"""
        while True:
            data = self.get_data_now()
            for key, val in data.items():
                self.set(name=key, value=val)
            yield self.env.timeout(self.t_sample)

    def get_data_now(self):
        """Get the data at the current env time.
        Reiterate the year if necessary.
        """
        now = self.env.now % self.one_year
        if now in self._data.index:
            return self._data.loc[now]
        # Interpolate:
        df = self._data.copy()
        df.loc[now] = np.nan
        return df.sort_index().interpolate().loc[now]

    def register_callbacks(self):
        """Dont do anything as this BaseModule is not event-triggered"""


def read_dat_file(dat_file):
    """
    Read a .dat file from the DWD's TRY data from the given .dat-file

    Args:
        dat_file (str): The file to read
    Returns:

    """
    # pylint: disable=raise-missing-from
    _sep = r"\s+"

    with open(dat_file, "r") as file:
        # First try for 2012 dataset:
        try:
            data_lines = file.readlines()[32:]
            data_lines.remove("*** \n")  # Remove line "*** "
        except ValueError:
            # Now for 2007 dataset:
            try:
                data_lines = file.readlines()[36:]
                data_lines.remove("***\n")  # Remove line "***"
            except ValueError:
                raise TypeError("Given .dat file could not be loaded.")

        output = io.StringIO()
        output.writelines(data_lines)
        output.seek(0)
        df = pd.read_csv(output, sep=_sep, header=0)
        # Convert the ours to seconds
        df.index *= 3600
    _key_map = {
        "t": "T_oda",
        "p": "pressure",
        "WR": "wind_direction",
        "WG": "wind_speed",
        "N": "coverage",
        "x": "absolute_humidity",
        "RF": "relative_humidity",
        "B": "beam_direct",
        "D": "beam_diffuse",
        "A:": "beam_atm",
        "E": "beam_terr",
    }
    # Rename df for easier later usage.
    df.rename(columns=_key_map, inplace=True)
    df = df[df.columns.intersection(_key_map.values())]
    df["T_oda"] += 273.15  # To Kelvin
    return df

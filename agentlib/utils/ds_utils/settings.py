from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

"""
    Defining global settings for the Data Space connectors. Agents on MPC and Simulation side need different settings from separate files.
    A OnenetClient will set up a ConnectorSettings instance for itself depending on the config of the Agent it's attached to
"""
class ConnectorConfigError(RuntimeError):
    pass

class MPCConnectorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env_mpc"), 
        env_prefix='connector_'
    )
    url: HttpUrl = "http://localhost"
    username: str
    password: SecretStr

class SimConnectorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env_sim"), 
        env_prefix='connector_'
    )
    url: HttpUrl = "http://localhost"
    username: str
    password: SecretStr
     
def GetConnectorSettings(kind: str):
    if kind == "sim":
        settings = SimConnectorSettings()
    elif kind == "mpc": 
        settings = MPCConnectorSettings()
    else:
        return ConnectorConfigError()
    return settings
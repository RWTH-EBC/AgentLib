import json
from pathlib import Path
from typing import Union, TypeVar, Type

from pydantic import FilePath, BaseModel

from agentlib.core.errors import ConfigurationError

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(
    config: Union[ConfigT, FilePath, str, dict], config_type: Type[ConfigT]
) -> ConfigT:
    """Generic config loader, either accepting a path to a json file, a json string, a
    dict or passing through a valid config object."""

    if isinstance(config, (str, Path)):
        # if we have a str / path, we need to check whether it is a file or a json string
        if Path(config).is_file():
            # if we have a valid file pointer, we load it
            with open(config, "r") as f:
                config = json.load(f)
        else:
            # since the str is not a file path, we assume it is json and try to load it
            try:
                config = json.loads(config)
            except json.JSONDecodeError as e:
                # if we failed, we raise an error notifying the user of possibilities
                raise ConfigurationError(
                    f"The config '{config:.100}' is neither an existing file path, nor a "
                    f"valid json document."
                ) from e
    return config_type.model_validate(config)

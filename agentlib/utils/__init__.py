"""
Module containing all util
functions for the agentlib.

Most notably, the custom injection enabling
dynamic loading of custom models and modules.
"""

import importlib.util
import os
import sys
from pathlib import Path

from .local_broadcast_broker import LocalBroadcastBroker
from .local_broker import LocalBroker
from .multi_processing_broker import MultiProcessingBroker


def custom_injection(config: dict, module_name: str = None):
    """
    Function to dynamically load new python files into
    the agentlib. Using this, users may use custom modules
    oder custom models together with the existing agentlib objects.

    Args:
        config (dict): Config dict containing the following items:
            file (str): Filepath to a python file (.py)
            class_name (str): Name of the class to be imported
        module_name (str, optional): Name of the imported module
            in the sys.modules list. Carefully check if duplicate
            module keys raise unexpected behaviour. If so,
            use randomly generated strings or similar in classes
            calling this function. Default is None.
            In that case, the path is converted to a matching string.

    Returns:
        class (object): The class object specified by class_name
    """
    assert "file" in config, (
        "For custom module injection, the config type dict has to "
        "contain a 'file'-key with an existing python file as value"
    )
    assert "class_name" in config, (
        "For custom module injection, the config type dict has to "
        "contain a 'class_name'-key with a string as value "
        "specifying the class to inject"
    )
    file = config.get("file")
    class_name = config.get("class_name")
    if not isinstance(file, (str, Path)):
        raise TypeError(f"Given file is not a string but {type(file)}")
    # Convert to Path object
    file = Path(file)
    # Check if file is a valid filepath
    if not os.path.isfile(file):
        raise FileNotFoundError(
            f"Given file '{str(file)}' was not found on your device."
        )

    # Build module_name if not given:
    if module_name is None:
        # Build a unique module_name to be imported based on the path
        module_name = ".".join([p.name for p in file.parents][:-1] + [file.stem])

    # Custom file import
    try:
        # Check if the module_name is already present
        if module_name in sys.modules:
            custom_module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, file)
            custom_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = custom_module
            spec.loader.exec_module(custom_module)
    except ImportError as err:
        raise ImportError(
            f"Could not inject given module '{class_name}' at '{file}' due to import "
            "error. Carefully check for circular imports and partially "
            "imported objects based on the following error message: "
            f"{err}"
        ) from err
    try:
        return custom_module.__dict__[class_name]
    except KeyError:
        raise ImportError(
            f"Given module '{custom_module}' does not "
            f"contain the specified class {class_name}"
        )

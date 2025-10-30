"""
Package containing all modules used by agents.
Use the helper functions get_module_type
to load module classes from this package.
"""

from typing import Union, Dict, Iterable, List

from agentlib.utils import plugin_import
from agentlib.utils.fuzzy_matching import fuzzy_match

# Global modules:
_MODULE_TYPES = plugin_import.SaveUpdateDict()
_LOADED_CORE_MODULES = False


def _import_modules():
    """
    Import the module information,
    but only if it has not been loaded yet.
    """
    global _MODULE_TYPES
    if not _LOADED_CORE_MODULES:
        _load_core_modules()

    # Only return copy, never allow changes in the private object
    return _MODULE_TYPES.copy()


def _load_core_modules() -> None:
    global _LOADED_CORE_MODULES, _MODULE_TYPES
    # Communicator
    import agentlib.modules.communicator as communicator

    _MODULE_TYPES.update(communicator.MODULE_TYPES)
    # Controller
    import agentlib.modules.controller as controller

    _MODULE_TYPES.update(controller.MODULE_TYPES)
    # Utils
    import agentlib.modules.utils as utils

    _MODULE_TYPES.update(utils.MODULE_TYPES)
    # Simulator
    import agentlib.modules.simulation as simulation

    _MODULE_TYPES.update(simulation.MODULE_TYPES)
    _LOADED_CORE_MODULES = True


def get_module_type(module_type) -> Union[Dict, Iterable]:
    """
    Return and load the given module type

    Args:
        module_type str:
            The string identifier to load the module.

    Returns:
        module BaseModuleType:
            The module specified by the given module_type
    """
    # Check if it's a plugin
    if "." in module_type:
        plugin_import.load_plugin(
            name=module_type.split(".")[0],
            loaded_classes=_MODULE_TYPES,
            plugin_types_name="MODULE_TYPES",
        )
    # Load the core and plugin modules
    module_types = _import_modules()
    if module_type in module_types:
        return module_types[module_type].import_class()
    # Raise error if still here

    matches = fuzzy_match(target=module_type, choices=module_types.keys())
    msg = (
        f"Given module_type '{module_type}' is neither in the AgentLib nor in "
        f"installed plugins. "
    )
    if matches:
        msg += f"Did you mean one of these? {', '.join(matches)}"

    # Extract names only
    raise ModuleNotFoundError(msg)


def get_all_module_types(plugins: List[str] = None):
    """
    Returns all available module types

    Args:
        plugins List[str]:
            A list of strings being the
            plugins to consider in the search.

    Returns:
        dict: Module types, with the key as the types name
        and the value being the ModuleImport instance
    """
    for plugin in plugins:
        plugin_import.load_plugin(
            name=plugin, loaded_classes=_MODULE_TYPES, plugin_types_name="MODULE_TYPES"
        )
    module_types = _import_modules()
    return module_types

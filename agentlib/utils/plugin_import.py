"""
Module containing all function to import new plugins
"""

import importlib
from pydantic import BaseModel


class ModuleImport(BaseModel):
    """
    Data-Class to import a given python file
    from ``import_path`` and load the given
    ``class_name``
    """

    import_path: str
    class_name: str

    def import_class(self):
        """Import the Module with class_name from the import path"""
        module = importlib.import_module(self.import_path)
        return getattr(module, self.class_name)


class SaveUpdateDict(dict):
    """
    Custom object to safely update the dictionary.
    Duplicate entries will raise an error.
    """

    def update(self, new_dict, **kwargs) -> None:
        """Check if all modules have distinct identifier strings

        Args:
            new_dict:
            **kwargs:
        """
        duplicates = set(self.keys()).intersection(new_dict.keys())
        if duplicates:
            raise KeyError(
                "One or multiple MODULE_TYPES contain "
                "the same string identifier (type). "
                f"The type has to be unique! {', '.join(duplicates)}"
            )
        super().update(new_dict)


def load_plugin(name: str, loaded_classes: SaveUpdateDict, plugin_types_name: str):
    """
    Loads the plugin based on the given name.

    Args:
        name str:
            Name of the plugin
        loaded_classes SaveUpdateDict:
            SaveUpdateDict instance with already
            loaded classes (modules or models)
        plugin_types_name str:
            Name of the dictionary in the plugin.
            Typical values are "MODULE_TYPES" or "MODEL_TYPES".
    """
    for key in loaded_classes:
        if key.startswith(f"{name}."):
            return  # Already loaded
    try:
        plugin = importlib.import_module(name)
    except ImportError as err:
        raise ImportError(f"Plugin '{name}' is not installed.") from err
    try:
        plugin_dict = getattr(plugin, plugin_types_name)
    except AttributeError:
        raise ImportError(
            f"Plugin '{name}' has no dictionary called "
            f"'{plugin_types_name}' to import plugin types."
        )
    if not isinstance(plugin_dict, dict):
        raise TypeError(
            f"Loaded object '{plugin_types_name}' is not a dictionary "
            f"but a {type(plugin_dict)}"
        )
    for key, value in plugin_dict.items():
        loaded_classes.update({f"{name}.{key}": value})

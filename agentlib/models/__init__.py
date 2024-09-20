"""Package with available models for the agentlib"""

from agentlib.utils import plugin_import
from agentlib.utils.fuzzy_matching import fuzzy_match
from agentlib.utils.plugin_import import ModuleImport

UNINSTALLED_MODEL_TYPES = {}

MODEL_TYPES = plugin_import.SaveUpdateDict(
    **{
        "statespace": ModuleImport(
            import_path="agentlib.models.scipy_model", class_name="ScipyStateSpaceModel"
        ),
        "fmu": ModuleImport(
            import_path="agentlib.models.fmu_model", class_name="FmuModel"
        ),
    }
)


def get_model_type(model_type) -> type:
    """
    Return and load the given module type

    Args:
        model_type str:
            The string identifier to load the module.

    Returns:
        module BaseModelType:
    """
    # Check if it's a plugin
    if "." in model_type:
        plugin_import.load_plugin(
            name=model_type.split(".")[0],
            loaded_classes=MODEL_TYPES,
            plugin_types_name="MODEL_TYPES",
        )
    # Load the core and plugin modules
    if model_type in MODEL_TYPES:
        return MODEL_TYPES[model_type].import_class()

    matches = fuzzy_match(target=model_type, choices=MODEL_TYPES.keys())
    msg = (
        f"Given model_type '{model_type}' is neither in the AgentLib nor in "
        f"installed plugins. "
    )
    if matches:
        msg += f"Did you mean one of these? {', '.join(matches)}"
    raise ModuleNotFoundError(msg)

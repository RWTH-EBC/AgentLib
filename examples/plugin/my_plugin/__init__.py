"""Top level file for an example plugin. Has to have a register function."""

from agentlib.utils.plugin_import import ModuleImport

from . import another_new_module
from . import new_module

# This property MODULE_TYPES has to exist with exactly this name, so that the plugin
# is usable in the agentlib
MODULE_TYPES = {
    "new_module": ModuleImport(
        import_path="my_plugin.new_module", class_name=new_module.NewModule.__name__
    ),
    "new_module2": ModuleImport(
        import_path="my_plugin.another_new_module",
        class_name=another_new_module.NewModule2.__name__,
    ),
}

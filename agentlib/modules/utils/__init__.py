"""Package containing utility modules."""

from agentlib.utils.plugin_import import ModuleImport


MODULE_TYPES = {
    "agentlogger": ModuleImport(
        import_path="agentlib.modules.utils.agent_logger", class_name="AgentLogger"
    ),
    "trysensor": ModuleImport(
        import_path="agentlib.modules.utils.try_sensor", class_name="TRYSensor"
    ),
}

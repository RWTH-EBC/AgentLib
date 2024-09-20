"""
Package contains all modules to communicate messages with
"""

from agentlib.utils.plugin_import import ModuleImport


MODULE_TYPES = {
    "clonemap": ModuleImport(
        import_path="agentlib.modules.communicator.clonemap",
        class_name="CloneMAPClient",
    ),
    "local": ModuleImport(
        import_path="agentlib.modules.communicator.local", class_name="LocalClient"
    ),
    "local_broadcast": ModuleImport(
        import_path="agentlib.modules.communicator.local_broadcast",
        class_name="LocalBroadcastClient",
    ),
    "mqtt": ModuleImport(
        import_path="agentlib.modules.communicator.mqtt", class_name="MqttClient"
    ),
    "multiprocessing_broadcast": ModuleImport(
        import_path="agentlib.modules.communicator.local_multiprocessing",
        class_name="MultiProcessingBroadcastClient",
    ),
}

from agentlib.utils.plugin_import import ModuleImport

MODULE_TYPES = {
    "pid": ModuleImport(
        import_path="agentlib.modules.controller.pid", class_name="PID"
    ),
    "bangbang": ModuleImport(
        import_path="agentlib.modules.controller.bangbang", class_name="BangBang"
    ),
}

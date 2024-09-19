from agentlib.utils import plugin_import

MODULE_TYPES = {
    "simulator": plugin_import.ModuleImport(
        import_path="agentlib.modules.simulation.simulator", class_name="Simulator"
    ),
    "csv_data_source": plugin_import.ModuleImport(
        import_path="agentlib.modules.simulation.csv_data_source",
        class_name="CSVDataSource",
    ),
}

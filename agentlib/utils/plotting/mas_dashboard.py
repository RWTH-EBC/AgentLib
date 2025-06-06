"""
Dashboard for visualizing results from a Multi-Agent System (MAS).
Allows selecting an agent and then a module within that agent to display
its specific results visualization.
Runs in a separate process to avoid blocking the main thread.
"""

import importlib
import logging
import multiprocessing
import socket
import webbrowser

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    import dash_bootstrap_components as dbc
except ImportError:
    raise ImportError(
        "Dash is not installed. Please install it to use the MAS dashboard: "
        "pip install agentlib[interactive]"
    )

from agentlib.core.errors import OptionalDependencyError
from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)


# Moved create_dash_app to top level to ensure picklability for multiprocessing
def create_dash_app(processed_agent_module_info, processed_mas_results):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "MAS Results Dashboard"

    agent_ids = list(processed_mas_results.keys())

    app.layout = dbc.Container(
        [
            html.H1("Multi-Agent System Results Dashboard", className="my-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Agent:"),
                            dcc.Dropdown(
                                id="agent-selector",
                                options=[
                                    {"label": agent_id, "value": agent_id}
                                    for agent_id in agent_ids
                                ],
                                value=agent_ids[0] if agent_ids else None,
                                clearable=False,
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Select Module:"),
                            dcc.Dropdown(
                                id="module-selector",
                                options=[],  # Populated by callback
                                clearable=False,
                            ),
                        ],
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(id="module-visualization-content"),
                )
            ),
        ],
        fluid=True,
    )

    @app.callback(
        [Output("module-selector", "options"), Output("module-selector", "value")],
        [Input("agent-selector", "value")],
    )
    def update_module_dropdown(selected_agent_id):
        if not selected_agent_id or not processed_mas_results:
            return [], None

        # Use preprocessed agent_module_info
        modules_for_agent = processed_agent_module_info.get(selected_agent_id, [])
        agent_modules_data = processed_mas_results.get(
            selected_agent_id, {}
        )  # Results data

        module_options = []
        first_module_id = None

        for module_info in modules_for_agent:
            module_id = module_info["id"]
            # Check if results exist for this module to decide if it's selectable
            if module_id in agent_modules_data:
                module_options.append({"label": module_id, "value": module_id})
                if first_module_id is None:
                    first_module_id = module_id

        return module_options, first_module_id

    @app.callback(
        Output("module-visualization-content", "children"),
        [Input("agent-selector", "value"), Input("module-selector", "value")],
    )
    def display_module_visualization(selected_agent_id, selected_module_id):
        if not selected_agent_id or not selected_module_id or not processed_mas_results:
            return html.P("Select an agent and a module to view visualization.")

        # Get module class_path from preprocessed info
        agent_modules = processed_agent_module_info.get(selected_agent_id, [])
        module_info_dict = next(
            (m for m in agent_modules if m["id"] == selected_module_id), None
        )

        if not module_info_dict:
            return html.P(
                f"Module '{selected_module_id}' (metadata) not found in agent '{selected_agent_id}'."
            )

        results_data = processed_mas_results.get(selected_agent_id, {}).get(
            selected_module_id, "__no_key__"
        )

        if isinstance(results_data, str) and results_data == "__no_key__":
            return html.P(
                f"No results data key found for module '{selected_module_id}' in agent '{selected_agent_id}'."
            )

        try:
            # Dynamically import the ModuleType
            class_path = module_info_dict["class_path"]
            module_path_str, class_name_str = class_path.rsplit(".", 1)
            imported_module_obj = importlib.import_module(module_path_str)
            ModuleType = getattr(imported_module_obj, class_name_str)

            visualization_layout = ModuleType.visualize_results(
                results_data=results_data,
                module_id=selected_module_id,
                agent_id=selected_agent_id,
            )
            if visualization_layout is None:
                return html.P(
                    f"Visualization not implemented or not available for module "
                    f"'{selected_module_id}' (type: {ModuleType.__name__}) in agent '{selected_agent_id}'."
                )
            return visualization_layout
        except OptionalDependencyError as e:
            # Use logger from the current process (Dash app process)
            logging.getLogger(__name__).error(
                f"Optional dependency error for {class_path}: {e}"
            )
            return html.Div(
                [
                    html.H4("Visualization Error"),
                    html.P(f"Could not render visualization for {class_path}."),
                    html.P(str(e)),
                    html.P("Please ensure all necessary libraries are installed."),
                ]
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Error generating visualization for module '{selected_module_id}' "
                f"(path: {class_path}) in agent '{selected_agent_id}': {e}",
                exc_info=True,
            )
            return html.P(
                f"An error occurred while generating the visualization for "
                f"module '{selected_module_id}': {str(e)}"
            )

    return app


def _run_dash_server(app_factory_func, app_factory_args, port, host):
    """
    Helper function to create and run the Dash app in a separate process.
    app_factory_func is now the actual create_dash_app function.
    """
    # Ensure that the current process is not a daemon if it needs to spawn children
    # (though Dash app itself usually doesn't spawn more processes)
    # multiprocessing.current_process().daemon = False
    app = app_factory_func(*app_factory_args)
    logger.info(f"Dash app server starting on http://{host}:{port}")
    try:
        app.run_server(debug=False, port=port, host=host)
    except Exception as e:
        logger.error(f"Error running Dash server: {e}", exc_info=True)


def get_free_port():
    """Get an available port by trying to bind to a socket."""
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                port += 1


def launch_mas_dashboard(
    mas: LocalMASAgency, mas_results: dict, block_main: bool
) -> multiprocessing.Process | None:
    """
    Launches the MAS results dashboard in a separate process.

    Args:
        mas: The LocalMASAgency instance.
        mas_results: A dictionary of results, typically from mas.get_results().
                     Expected structure: {agent_id: {module_id: results_data}}
    Returns:
        multiprocessing.Process: The process running the Dash dashboard.
    """
    # Preprocess MAS structure for pickling
    agent_module_info_for_app = {}
    if mas and hasattr(mas, "_agents"):
        for agent_id, agent_instance in mas._agents.items():
            agent_module_info_for_app[agent_id] = []
            if agent_instance and hasattr(agent_instance, "modules"):
                for module_instance in agent_instance.modules:
                    module_class = module_instance.__class__
                    class_path = f"{module_class.__module__}.{module_class.__name__}"
                    agent_module_info_for_app[agent_id].append(
                        {"id": module_instance.id, "class_path": class_path}
                    )

    port = get_free_port()
    host = "0.0.0.0"  # Standard host for Dash to be accessible

    # Arguments for the app factory function
    app_factory_args = (agent_module_info_for_app, mas_results)

    # open tab
    webbrowser.open_new_tab(f"http://localhost:{port}")

    if not block_main:
        # Create and start the process
        # Note: On Windows, make sure the script calling this is guarded by if __name__ == '__main__'
        # if it also creates processes.
        ctx = multiprocessing.get_context(
            "spawn"
        )  # Using 'spawn' context for better cross-platform compatibility
        dashboard_process = ctx.Process(
            target=_run_dash_server,  # Target is the helper
            args=(
                create_dash_app,
                app_factory_args,
                port,
                host,
            ),  # Pass the top-level create_dash_app and its args
        )
        dashboard_process.daemon = (
            False  # Main program will wait for this process unless terminated
        )
        dashboard_process.start()
        logger.info(
            f"MAS Dashboard is launching in a separate process on http://localhost:{port}"
        )
        return dashboard_process
    else:
        # run in main process
        _run_dash_server(create_dash_app, app_factory_args, port, host)


if __name__ == "__main__":
    # This example won't run directly as it needs a LocalMASAgency instance and results.
    # It's intended to be called from another script.
    print(
        "To run this dashboard, import and call launch_mas_dashboard "
        "from your script with a LocalMASAgency instance and its results. \n"
        "Example: \n"
        "import multiprocessing \n"
        "if __name__ == '__main__': \n"
        "    multiprocessing.freeze_support() # For Windows executable freezing \n"
        "    # ... your MAS setup ... \n"
        "    dashboard_proc = launch_mas_dashboard(my_mas, my_results) \n"
        "    # ... your main script continues ... \n"
        "    print('Dashboard process started. Press Ctrl+C in the console running the script, or implement other logic to stop it.') \n"
        "    try: \n"
        "        dashboard_proc.join() # Wait for dashboard process to finish (e.g., if user closes browser tab and server stops) \n"
        "    except KeyboardInterrupt: \n"
        "        print('Terminating dashboard process...') \n"
        "        dashboard_proc.terminate() \n"
        "        dashboard_proc.join() \n"
        "    print('Dashboard process finished.')"
    )
    pass

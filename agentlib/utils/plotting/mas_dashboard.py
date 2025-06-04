"""
Dashboard for visualizing results from a Multi-Agent System (MAS).
Allows selecting an agent and then a module within that agent to display
its specific results visualization.
"""

import logging
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


def launch_mas_dashboard(mas: LocalMASAgency, mas_results: dict):
    """
    Launches the MAS results dashboard.

    Args:
        mas: The LocalMASAgency instance.
        mas_results: A dictionary of results, typically from mas.get_results().
                     Expected structure: {agent_id: {module_id: results_data}}
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "MAS Results Dashboard"

    agent_ids = list(mas_results.keys())

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
        if not selected_agent_id or not mas_results:
            return [], None

        agent_modules_data = mas_results.get(selected_agent_id, {})
        module_options = []
        first_module_id = None

        agent_instance = mas._agents.get(selected_agent_id)
        if not agent_instance:
            return [], None

        for module_instance in agent_instance.modules:
            module_id = module_instance.id
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
        if not selected_agent_id or not selected_module_id or not mas_results:
            return html.P("Select an agent and a module to view visualization.")

        agent_instance = mas._agents.get(selected_agent_id)
        if not agent_instance:
            return html.P(f"Agent '{selected_agent_id}' not found in MAS instance.")

        module_instance = next(
            (m for m in agent_instance.modules if m.id == selected_module_id), None
        )
        if not module_instance:
            return html.P(
                f"Module '{selected_module_id}' not found in agent '{selected_agent_id}'."
            )

        results_data = mas_results.get(selected_agent_id, {}).get(
            selected_module_id, "__no_key__"
        )

        if isinstance(results_data, str) and results_data == "__no_key__":
            return html.P(
                f"No results data key found for module '{selected_module_id}' in agent '{selected_agent_id}'."
            )

        ModuleType = type(module_instance)
        try:
            # results_data here can be the actual data (DataFrame, None, etc.)
            visualization_layout = ModuleType.visualize_results(
                results_data=results_data,  # This will be passed to the module's visualize_results
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
            logger.error(f"Optional dependency error for {ModuleType.__name__}: {e}")
            return html.Div(
                [
                    html.H4("Visualization Error"),
                    html.P(
                        f"Could not render visualization for {ModuleType.__name__}."
                    ),
                    html.P(str(e)),
                    html.P("Please ensure all necessary libraries are installed."),
                ]
            )
        except Exception as e:
            logger.error(
                f"Error generating visualization for module '{selected_module_id}' "
                f"(type: {ModuleType.__name__}) in agent '{selected_agent_id}': {e}",
                exc_info=True,
            )
            return html.P(
                f"An error occurred while generating the visualization for "
                f"module '{selected_module_id}': {str(e)}"
            )

    port = get_free_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    logger.info(f"MAS Dashboard is running on http://localhost:{port}")
    app.run_server(debug=False, port=port, host="0.0.0.0")


if __name__ == "__main__":
    print(
        "To run this dashboard, import and call launch_mas_dashboard "
        "with a LocalMASAgency instance and its results."
    )
    pass

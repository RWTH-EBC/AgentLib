"""
Dashboard for visualizing results from a Multi-Agent System (MAS).
Allows selecting an agent and then a module within that agent to display
its specific results visualization.
Supports both static (post-run) and live (during run) modes.
Runs in a separate process to avoid blocking the main thread.
"""

import importlib
import logging
import multiprocessing
import socket
import webbrowser
import threading
import queue  # For queue.Empty
from typing import Dict, Optional, Tuple, Any, List

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc
except ImportError:
    raise ImportError(
        "Dash is not installed. Please install it to use the MAS dashboard: "
        "pip install agentlib[interactive]"
    )

from agentlib.core.errors import OptionalDependencyError
from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)

# Sentinel value to signal the IPC listener thread to stop
_STOP_IPC_LISTENER = object()


def _ipc_listener_loop(
    mas_instance: LocalMASAgency,
    request_q: multiprocessing.Queue,
    response_q: multiprocessing.Queue,
    stop_event: threading.Event,
):
    """Listens for data requests from the Dash app and calls the appropriate module."""
    logger.info("IPC listener thread started for MAS dashboard.")
    while not stop_event.is_set():
        try:
            # Timeout to allow checking stop_event periodically
            agent_id, module_id, update_token = request_q.get(timeout=0.1)
            if agent_id is _STOP_IPC_LISTENER:
                logger.info("IPC listener received stop signal.")
                break

            logger.debug(
                f"IPC listener: Request for {agent_id}/{module_id}, token: {update_token}"
            )
            data_chunk, next_token = None, None
            try:
                agent = mas_instance.get_agent(agent_id)
                if agent:
                    module = agent.get_module(module_id)
                    if module:
                        if hasattr(module, "get_results_incremental"):
                            data_chunk, next_token = module.get_results_incremental(
                                update_token=update_token
                            )
                        else:
                            logger.warning(
                                f"Module {module_id} in agent {agent_id} does not have "
                                f"'get_results_incremental' method. Falling back to 'get_results'."
                            )
                            if update_token is None: # Only call get_results for initial load
                                data_chunk = module.get_results()
                            # next_token remains None, implying no further incremental updates from this fallback
                    else:
                        logger.error(f"IPC listener: Module {module_id} not found in agent {agent_id}")
                else:
                    logger.error(f"IPC listener: Agent {agent_id} not found in MAS")
            except Exception as e:
                logger.error(
                    f"IPC listener: Error calling get_results_incremental for {agent_id}/{module_id}: {e}",
                    exc_info=True,
                )
            response_q.put((data_chunk, next_token))
        except queue.Empty:
            continue  # Timeout, check stop_event again
        except Exception as e:
            logger.error(f"IPC listener: Unexpected error in loop: {e}", exc_info=True)
            import time
            time.sleep(0.1) # Avoid busy-looping
    logger.info("IPC listener thread stopped for MAS dashboard.")


def create_dash_app(
    agent_module_info: Dict,
    live_update: bool,
    static_results: Optional[Dict] = None,
    ipc_queues: Optional[Tuple[multiprocessing.Queue, multiprocessing.Queue]] = None,
    update_interval_sec: Optional[float] = None,
):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "MAS Results Dashboard"

    agent_ids = list(agent_module_info.keys())

    # In-memory cache for DataFrames within the Dash app process
    # This dict is accessible to callbacks defined within create_dash_app
    app_data_cache = {} 

    layout_components = [
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
        dbc.Row(dbc.Col(html.Div(id="module-visualization-content"))),
    ]

    if live_update:
        layout_components.extend([
            # dcc.Store(id='data-cache', data={}), # Removed: DataFrames not JSON serializable for Store
            dcc.Store(id='token-cache', data={}), # Stores { "agent/module": next_token }
            dcc.Store(id='data-update-trigger', data=0), # Dummy store to trigger visualization updates
            dcc.Interval(
                id='live-update-interval',
                interval=int(update_interval_sec * 1000) if update_interval_sec else 5000,
                n_intervals=0
            )
        ])

    app.layout = dbc.Container(layout_components, fluid=True)

    @app.callback(
        [Output("module-selector", "options"), Output("module-selector", "value")],
        [Input("agent-selector", "value")],
    )
    def update_module_dropdown(selected_agent_id):
        if not selected_agent_id:
            return [], None

        modules_for_agent = agent_module_info.get(selected_agent_id, [])
        module_options = []
        first_module_id = None

        for module_info_dict in modules_for_agent:
            module_id = module_info_dict["id"]
            if live_update:
                module_options.append({"label": module_id, "value": module_id})
                if first_module_id is None:
                    first_module_id = module_id
            elif static_results:
                agent_modules_data = static_results.get(selected_agent_id, {})
                if module_id in agent_modules_data: # Check if results exist for this module
                    module_options.append({"label": module_id, "value": module_id})
                    if first_module_id is None:
                        first_module_id = module_id
        return module_options, first_module_id

    # Callback for fetching data in live mode
    if live_update:
        @app.callback(
            [Output('token-cache', 'data'), Output('data-update-trigger', 'data')],
            [Input('live-update-interval', 'n_intervals'),
             Input("agent-selector", "value"),
             Input("module-selector", "value")],
            [State('token-cache', 'data'), State('data-update-trigger', 'data')]
        )
        def fetch_live_data(n_intervals, selected_agent_id, selected_module_id,
                            current_token_cache, current_trigger_val):
            ctx = dash.callback_context
            triggered_prop_id = ctx.triggered[0]['prop_id'] if ctx.triggered else "initial_load"

            if not selected_agent_id or not selected_module_id or not ipc_queues:
                return current_token_cache, dash.no_update # No change to trigger if no selection

            request_q, response_q = ipc_queues
            module_key = f"{selected_agent_id}/{selected_module_id}"
            
            update_token_to_send = None
            if 'live-update-interval.n_intervals' in triggered_prop_id:
                update_token_to_send = current_token_cache.get(module_key)
            
            logger.debug(f"Dash app: Requesting data for {module_key} with token: {update_token_to_send}")
            try:
                request_q.put((selected_agent_id, selected_module_id, update_token_to_send))
                data_chunk, next_token = response_q.get(timeout=(update_interval_sec * 0.9) if update_interval_sec else 4.5)
                logger.debug(f"Dash app: Received data for {module_key}. Next token: {next_token}")

                # Update in-memory Python dictionary for DataFrames
                app_data_cache[module_key] = data_chunk 
                
                new_token_cache = current_token_cache.copy()
                new_token_cache[module_key] = next_token
                
                # Increment trigger to signal data update for visualization callback
                new_trigger_val = (current_trigger_val + 1) if current_trigger_val is not None else 0
                return new_token_cache, new_trigger_val

            except queue.Empty:
                logger.warning(f"Dash app: Timeout waiting for data from IPC for {module_key}")
                return current_token_cache, dash.no_update 
            except Exception as e:
                logger.error(f"Dash app: Error in fetch_live_data for {module_key}: {e}", exc_info=True)
                return current_token_cache, dash.no_update

    # Callback for displaying module visualization
    @app.callback(
        Output("module-visualization-content", "children"),
        [Input("agent-selector", "value"), 
         Input("module-selector", "value"),
         Input('data-update-trigger', 'data') if live_update else Input("module-selector", "value")],
        # The dummy Input for static mode (using module-selector again) is just to make Dash happy
        # with the conditional Input list. The value of this dummy input isn't used in static mode for this logic.
    )
    def display_module_visualization(selected_agent_id, selected_module_id, data_update_trigger_val_or_dummy):
        if not selected_agent_id or not selected_module_id:
            return html.P("Select an agent and a module to view visualization.")

        current_results_data = None
        if live_update:
            module_key = f"{selected_agent_id}/{selected_module_id}"
            # Read from the in-memory Python dictionary
            current_results_data = app_data_cache.get(module_key)
            
            if current_results_data is None:
                ctx = dash.callback_context
                # Check if triggered by agent/module selection, if so, data is being fetched.
                if ctx.triggered and any(trigger['prop_id'] in ['agent-selector.value', 'module-selector.value'] for trigger in ctx.triggered):
                    return html.P(f"Fetching live data for {selected_module_id}...")
                return html.P(f"No live data currently available for module '{selected_module_id}'.")
        else: # Static mode
            if not static_results:
                 return html.P("Static results not available.")
            current_results_data = static_results.get(selected_agent_id, {}).get(
                selected_module_id, "__no_key__"
            )
            if isinstance(current_results_data, str) and current_results_data == "__no_key__":
                return html.P(
                    f"No results data key found for module '{selected_module_id}' in agent '{selected_agent_id}' (static mode)."
                )
        
        agent_modules_meta = agent_module_info.get(selected_agent_id, [])
        module_info_dict = next(
            (m for m in agent_modules_meta if m["id"] == selected_module_id), None
        )

        if not module_info_dict:
            return html.P(
                f"Module metadata for '{selected_module_id}' not found in agent '{selected_agent_id}'."
            )
        
        try:
            class_path = module_info_dict["class_path"]
            module_path_str, class_name_str = class_path.rsplit(".", 1)
            imported_module_obj = importlib.import_module(module_path_str)
            ModuleType = getattr(imported_module_obj, class_name_str)

            visualization_layout = ModuleType.visualize_results(
                results_data=current_results_data,
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


def _run_dash_server(
    app_factory_func, # This is create_dash_app
    # Args for create_dash_app:
    agent_module_info_arg: Dict,
    live_update_arg: bool,
    static_results_arg: Optional[Dict],
    ipc_queues_arg: Optional[Tuple[multiprocessing.Queue, multiprocessing.Queue]],
    update_interval_sec_arg: Optional[float],
    # Args for app.run_server:
    port: int,
    host: str,
):
    """
    Helper function to create and run the Dash app in a separate process.
    """
    app = app_factory_func(
        agent_module_info=agent_module_info_arg,
        live_update=live_update_arg,
        static_results=static_results_arg,
        ipc_queues=ipc_queues_arg,
        update_interval_sec=update_interval_sec_arg,
    )
    logger.info(f"Dash app server starting on http://{host}:{port}")
    try:
        # Set use_reloader=False to prevent issues with multiprocessing and threads
        app.run_server(debug=False, port=port, host=host, use_reloader=False)
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
        if port > 9000: # Avoid infinite loop in rare cases
            raise OSError("Could not find a free port between 8050 and 9000.")


def launch_mas_dashboard(
    mas: LocalMASAgency,
    mas_results: Optional[dict] = None,
    block_main: bool = False,
    live_update: bool = False,
    update_interval_sec: float = 5.0,
) -> multiprocessing.Process | None:
    """
    Launches the MAS results dashboard.

    Args:
        mas: The LocalMASAgency instance. Required for agent_module_info,
             and for live_update mode.
        mas_results: A dictionary of results for static mode.
                     Expected: {agent_id: {module_id: results_data}}.
                     Ignored if live_update is True.
        block_main: If True, runs Dash server in the main process, blocking it.
                    If False, runs in a separate process.
        live_update: If True, dashboard attempts to fetch live data from modules.
        update_interval_sec: Interval for polling data in live_update mode.
    Returns:
        multiprocessing.Process: The process running the Dash dashboard (if not block_main).
                                 None otherwise or if an error occurs.
    """
    if mas is None:
        logger.error("LocalMASAgency instance ('mas') must be provided.")
        return None
    if not live_update and mas_results is None:
        logger.error("mas_results must be provided if not in live_update mode.")
        return None

    agent_module_info_for_app = {}
    if hasattr(mas, "_agents"):
        for agent_id, agent_instance in mas._agents.items():
            agent_module_info_for_app[agent_id] = []
            if hasattr(agent_instance, "modules"):
                for module_instance in agent_instance.modules:
                    module_class = module_instance.__class__
                    class_path = f"{module_class.__module__}.{module_class.__name__}"
                    agent_module_info_for_app[agent_id].append(
                        {"id": module_instance.id, "class_path": class_path}
                    )
    if not agent_module_info_for_app:
        logger.warning("No agents or modules found in the MAS instance. Dashboard might be empty.")

    port = get_free_port()
    host = "0.0.0.0"

    factory_args_for_create_dash_app = {
        "agent_module_info": agent_module_info_for_app,
        "live_update": live_update,
        "static_results": None if live_update else mas_results,
        "ipc_queues": None,
        "update_interval_sec": update_interval_sec if live_update else None,
    }

    ipc_listener_thread_obj = None # To store the thread object
    stop_ipc_event_obj = None # To store the event object

    if live_update:
        request_q = multiprocessing.Queue()
        response_q = multiprocessing.Queue()
        factory_args_for_create_dash_app["ipc_queues"] = (request_q, response_q)
        
        stop_ipc_event_obj = threading.Event()
        ipc_listener_thread_obj = threading.Thread(
            target=_ipc_listener_loop,
            args=(mas, request_q, response_q, stop_ipc_event_obj),
            daemon=True
        )
        ipc_listener_thread_obj.start()
        
    webbrowser.open_new_tab(f"http://localhost:{port}")

    process_args_for_run_server = (
        create_dash_app, # app_factory_func
        factory_args_for_create_dash_app["agent_module_info"],
        factory_args_for_create_dash_app["live_update"],
        factory_args_for_create_dash_app["static_results"],
        factory_args_for_create_dash_app["ipc_queues"],
        factory_args_for_create_dash_app["update_interval_sec"],
        port,
        host,
    )

    dashboard_process = None
    if not block_main:
        ctx = multiprocessing.get_context("spawn")
        dashboard_process = ctx.Process(
            target=_run_dash_server,
            args=process_args_for_run_server
        )
        dashboard_process.daemon = False # So main program can wait for it if needed
        dashboard_process.start()
        logger.info(
            f"MAS Dashboard is launching in a separate process on http://localhost:{port}"
        )
        # The caller might need to manage stop_ipc_event_obj and ipc_listener_thread_obj
        # if they want to explicitly stop the IPC listener when the dashboard_process is terminated.
        # However, as a daemon thread, it will exit with the main process.
        return dashboard_process
    else:
        # Run in main process
        _run_dash_server(*process_args_for_run_server)
        if live_update and stop_ipc_event_obj and ipc_listener_thread_obj:
            logger.info("Main process dashboard finished. Attempting to stop IPC listener.")
            stop_ipc_event_obj.set()
            # Send a stop signal through the queue as well to break out of q.get()
            if factory_args_for_create_dash_app["ipc_queues"]:
                try:
                    factory_args_for_create_dash_app["ipc_queues"][0].put((_STOP_IPC_LISTENER, None, None), timeout=0.1)
                except Exception: # pragma: no cover
                    pass 
            ipc_listener_thread_obj.join(timeout=1.0)
            if ipc_listener_thread_obj.is_alive():
                logger.warning("IPC listener thread did not stop in time.")
        return None


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
        "    # For static dashboard: \n"
        "    # results = my_mas.get_results() \n"
        "    # dashboard_proc = launch_mas_dashboard(my_mas, mas_results=results) \n"
        "    # For live dashboard: \n"
        "    # dashboard_proc = launch_mas_dashboard(my_mas, live_update=True) \n"
        "    # ... your main script continues ... \n"
        "    print('Dashboard process started. Press Ctrl+C in the console running the script, or implement other logic to stop it.') \n"
        "    try: \n"
        "        if dashboard_proc: dashboard_proc.join() \n"
        "    except KeyboardInterrupt: \n"
        "        print('Terminating dashboard process...') \n"
        "        if dashboard_proc: dashboard_proc.terminate(); dashboard_proc.join() \n"
        "    print('Dashboard process finished.')"
    )
    pass

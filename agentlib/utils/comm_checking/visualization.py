import json
import socket
import threading
import webbrowser
from contextlib import closing
from pathlib import Path
from typing import (
    List,
    Union,
)

from agentlib.core.errors import OptionalDependencyError
from agentlib.utils.comm_checking.app import create_dash_app
from agentlib.utils.comm_checking.create_comm_graph import create_comm_graph

try:
    import dash
    import networkx as nx
    import plotly.graph_objects as go
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
except ImportError:
    raise OptionalDependencyError(
        used_object=f"Communication checker",
        dependency_install="gui",
    )


def load_json_to_dict(
    input_data: Union[str, dict, List[Union[str, dict]]]
) -> List[dict]:
    """
    Loads JSON data from various input formats and returns a list of dictionaries.
    """
    if not isinstance(input_data, list):
        input_data = [input_data]

    results = []
    for item in input_data:
        if isinstance(item, str):
            if Path(item).is_file():
                with open(item, "r") as file:
                    data = json.load(file)
            else:
                data = json.loads(item)
        elif isinstance(item, dict):
            data = item
        else:
            raise ValueError(
                "Input data must be a string (file path or JSON), or a dictionary."
            )

        results.append(data)

    return results


def process_config(config: Union[str, dict]) -> dict:
    """
    Process a single config, loading JSON files for modules if necessary.
    """
    if isinstance(config, str):
        config = load_json_to_dict(config)[0]

    processed_modules = []
    for module in config.get("modules", []):
        if isinstance(module, str):
            processed_modules.extend(load_json_to_dict(module))
        else:
            processed_modules.append(module)

    config["modules"] = processed_modules
    return config


def find_free_port(start_port=8050, max_port=8200):
    """Find a free port in the given range."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        for port in range(start_port, max_port):
            try:
                s.bind(("", port))
                return port
            except socket.error:
                continue
    return None


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def visualize_agents(
    configs: List[Union[str, dict]],
    port: int = 8050,
    background: bool = False,
    force_new: bool = False,
):
    # Process configs
    processed_configs = [process_config(config) for config in configs]

    G, vars_by_module = create_comm_graph(processed_configs)
    app = create_dash_app(G, vars_by_module)

    if is_port_in_use(port):
        if force_new:
            raise RuntimeError(
                f"Port {port} is in use. Please terminate the application that uses it."
            )
        else:
            new_port = find_free_port(port)
            if new_port:
                print(f"Port {port} is already in use. Using port {new_port} instead.")
                port = new_port
            else:
                print(
                    "No available ports found. Please close some running servers and try again."
                )
                return

    def run_dash():
        print(f"Starting server on port {port}")
        app.run_server(debug=False, port=port)

    if background:
        # Start the Dash app in a daemon thread
        dash_thread = threading.Thread(target=run_dash, daemon=True)
        dash_thread.start()

        # Open the web browser
        webbrowser.open(f"http://localhost:{port}")

        return dash_thread
    else:
        # Run in the main thread (blocking)
        webbrowser.open(f"http://localhost:{port}")
        run_dash()


if __name__ == "__main__":
    directory_path = Path(
        r"D:\repos\AgentLib\examples\multi-agent-systems\room_mas\configs"
    )
    configs = load_json_to_dict([str(file) for file in directory_path.glob("*")])

    # Run the visualization
    visualize_agents(configs)

    # Keep the main thread alive
    while True:
        try:
            input("Press Ctrl+C to exit...")
        except KeyboardInterrupt:
            print("Exiting...")
            break


# TODO next steps for this comm check:
#  1. on hover over a node, add a subgraph that shows communication between modules
#  2. check the shared attribute for inter-agent-communication
#  3. add multi-processing broadcast

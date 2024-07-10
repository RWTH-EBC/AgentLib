import itertools
import json
import socket
import threading
import typing
import webbrowser
from contextlib import closing
from pathlib import Path
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
    get_type_hints,
    get_origin,
    get_args,
    Union,
)

from agentlib.core.errors import OptionalDependencyError

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
import numpy as np


from agentlib import AgentVariable, Source
from agentlib.core.agent import get_module_class

AG_ID = str
MOD_ID = str
Alias = str


def get_config_class(type_):
    module_class = get_module_class(type_)
    config_class = get_type_hints(module_class)["config"]
    config_fields = {}
    agent_variables = []

    for field_name, field_info in config_class.__fields__.items():
        if field_name.startswith("_"):
            continue

        field_type = field_info.annotation
        default_value = field_info.default

        if default_value is None or default_value == Ellipsis:
            continue

        # If our annotation is List[AgentVariable], we have to work around the
        # generic type hints
        origin = get_origin(field_type)
        generic_args = get_args(field_type)
        if origin in {List, list} and len(generic_args) == 1:
            field_type = generic_args[0]

        if not isinstance(field_type, type):
            continue

        if not issubclass(field_type, AgentVariable):
            continue

        config_fields[field_name] = default_value

        if origin in {List, list}:
            agent_variables.extend(default_value)
        else:
            agent_variables.append(default_value)

    return config_class, config_fields, agent_variables


def create_configs(configs: list[dict]) -> List[Dict]:
    agent_configs = []
    for config in configs:
        agent_config = {"id": config["id"], "modules": []}
        for module in config["modules"]:
            module_config = module.copy()
            _conf_class, _fields, _variables = get_config_class(module)
            module_config["_config_class"] = _conf_class
            module_config["_config_fields"] = _fields
            module_config["_agent_variables"] = _variables
            agent_config["modules"].append(module_config)
        agent_configs.append(agent_config)
    return agent_configs


def collect_vars(configs_: List[Dict]) -> Dict[AG_ID, Dict[MOD_ID, List[Dict]]]:
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]] = {}
    for config in configs_:
        ag_id = config["id"]
        vars_by_module[ag_id] = {}
        for module in config["modules"]:
            mod_id = module.get("id", module["_config_class"].__name__)
            vars_by_module[ag_id][mod_id] = []
            config_fields = module["_config_fields"]
            for key, value in module.items():
                if key in config_fields:
                    if isinstance(value, list):
                        vars_by_module[ag_id][mod_id].extend(value)
                    else:
                        vars_by_module[ag_id][mod_id].append(value)
            vars_by_module[ag_id][mod_id].extend(
                [var.dict() for var in module["_agent_variables"]]
            )
            # vars_by_module[ag_id][mod_id] = [
            #     dict(t)
            #     for t in {tuple(d.items()) for d in vars_by_module[ag_id][mod_id]}
            # ]
    return vars_by_module


class VarTuple(typing.NamedTuple):
    ag_id: str
    mod_id: str
    var_name: str
    shared: bool
    unknown: Optional[Dict]


def order_vars_by_alias(
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]]
) -> Dict[Alias, List[VarTuple]]:
    vars_by_alias: Dict[Alias, List[VarTuple]] = {}
    for ag_id, modules in vars_by_module.items():
        for mod_id, ag_vars in modules.items():
            for var in ag_vars:
                alias = var.get("alias", var["name"])
                if alias not in vars_by_alias:
                    vars_by_alias[alias] = []
                vars_by_alias[alias].append(
                    VarTuple(
                        ag_id=ag_id,
                        mod_id=mod_id,
                        var_name=var["name"],
                        shared=var.get("shared", False),
                        unknown=None,
                    )
                )
    return vars_by_alias


def check_comm_conditions(
    sender_agent: str, receiver_agent: str, configs: List[Dict], vars_by_module: Dict
) -> List[str]:
    def get_comm_module(agent_config):
        return next(
            (
                module
                for module in agent_config["modules"]
                if module["type"] in ["mqtt", "local", "local_broadcast"]
            ),
            None,
        )

    sender_config = next(config for config in configs if config["id"] == sender_agent)
    receiver_config = next(
        config for config in configs if config["id"] == receiver_agent
    )

    sender_comm = get_comm_module(sender_config)
    receiver_comm = get_comm_module(receiver_config)

    if not sender_comm or not receiver_comm:
        return []

    comm_type_match = sender_comm["type"] == receiver_comm["type"]
    subscription_valid = sender_agent in receiver_comm.get("subscriptions", [])
    is_broadcast = sender_comm["type"] in {
        "local_broadcast",
        "multiprocessing_broadcast",
    }

    if not (comm_type_match and (subscription_valid or is_broadcast)):
        return []

    communicable_vars = []
    for module in vars_by_module[receiver_agent].values():
        for var in module:
            sender_source = Source.create(var.get("source", None))
            receiver_source = Source.create(var.get("source", None))
            if receiver_source.matches(sender_source):
                communicable_vars.append(var["name"])

    return communicable_vars


def create_comm_graph(configs: List[Dict]) -> Tuple[nx.DiGraph, Dict]:
    configs_: List[Dict] = create_configs(configs)
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]] = collect_vars(configs_)
    vars_by_alias: Dict[Alias, List[VarTuple]] = order_vars_by_alias(vars_by_module)

    g = nx.DiGraph()
    g.add_nodes_from(vars_by_module.keys())

    for alias, var_list in vars_by_alias.items():
        for var1, var2 in itertools.combinations(var_list, 2):
            if var1.ag_id == var2.ag_id:
                continue

            # Check communication from var1.ag_id to var2.ag_id
            communicable_vars_1_to_2 = check_comm_conditions(
                var1.ag_id, var2.ag_id, configs, vars_by_module
            )
            if var2.var_name in communicable_vars_1_to_2:
                if g.has_edge(var1.ag_id, var2.ag_id):
                    g[var1.ag_id][var2.ag_id]["label"] = set(
                        g[var1.ag_id][var2.ag_id]["label"].split("\n")
                    )
                    g[var1.ag_id][var2.ag_id]["label"].add(alias)
                    g[var1.ag_id][var2.ag_id]["label"] = "\n".join(
                        g[var1.ag_id][var2.ag_id]["label"]
                    )
                else:
                    g.add_edge(var1.ag_id, var2.ag_id, label=alias)

            # Check communication from var2.ag_id to var1.ag_id
            communicable_vars_2_to_1 = check_comm_conditions(
                var2.ag_id, var1.ag_id, configs, vars_by_module
            )
            if var1.var_name in communicable_vars_2_to_1:
                if g.has_edge(var2.ag_id, var1.ag_id):
                    g[var2.ag_id][var1.ag_id]["label"] = set(
                        g[var2.ag_id][var1.ag_id]["label"].split("\n")
                    )
                    g[var2.ag_id][var1.ag_id]["label"].add(alias)
                    g[var2.ag_id][var1.ag_id]["label"] = "\n".join(
                        g[var2.ag_id][var1.ag_id]["label"]
                    )
                else:
                    g.add_edge(var2.ag_id, var1.ag_id, label=alias)

    return g, vars_by_module


def create_dash_app(G, vars_by_module):
    app = dash.Dash(__name__)

    # Get all unique variables
    all_variables = set()
    for edge in G.edges(data=True):
        all_variables.update(edge[2]["label"].split("\n"))
    all_variables = sorted(list(all_variables), key=str.casefold)

    layouts = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "shell": nx.shell_layout,
    }

    pos = layouts["spring"](G)

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Agent Communication Network"),
                    dcc.Graph(id="network-graph", style={"height": "80vh"}),
                    html.Div(
                        [
                            html.Button(
                                "Spring Layout", id="spring-button", n_clicks=0
                            ),
                            html.Button(
                                "Circular Layout", id="circular-button", n_clicks=0
                            ),
                            html.Button("Shell Layout", id="shell-button", n_clicks=0),
                            html.Button(
                                "Shuffle Layout", id="shuffle-button", n_clicks=0
                            ),
                        ],
                        style={"padding": "10px"},
                    ),
                ],
                style={
                    "width": "80%",
                    "display": "inline-block",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.H3("Filter Variables"),
                    dcc.Checklist(
                        id="variable-checklist",
                        options=[
                            {
                                "label": html.Span(
                                    var, id={"type": "var-span", "index": var}
                                ),
                                "value": var,
                            }
                            for var in all_variables
                        ],
                        value=all_variables,
                        labelStyle={"display": "block"},
                    ),
                ],
                style={
                    "width": "20%",
                    "display": "inline-block",
                    "vertical-align": "top",
                    "padding": "20px",
                    "overflow-y": "auto",
                    "max-height": "80vh",
                },
            ),
        ],
        style={"display": "flex"},
    )

    @app.callback(
        Output("network-graph", "figure"),
        [
            Input("spring-button", "n_clicks"),
            Input("circular-button", "n_clicks"),
            Input("shell-button", "n_clicks"),
            Input("shuffle-button", "n_clicks"),
            Input("variable-checklist", "value"),
            Input({"type": "var-span", "index": dash.dependencies.ALL}, "n_hover"),
        ],
        [State("network-graph", "figure")],
    )
    def update_layout(
        spring_clicks,
        circular_clicks,
        shell_clicks,
        shuffle_clicks,
        active_variables,
        var_hovers,
        current_fig,
    ):
        nonlocal pos
        ctx = dash.callback_context
        if not ctx.triggered:
            return create_graph_figure(pos, active_variables)
        else:
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "shuffle-button":
                pos = {
                    node: (
                        pos[node][0] + np.random.normal(0, 0.1),
                        pos[node][1] + np.random.normal(0, 0.1),
                    )
                    for node in pos
                }
            elif button_id == "variable-checklist":
                # Don't change the layout, just update active variables
                pass
            elif "var-span" in button_id:
                # Highlight the hovered variable
                hovered_var = json.loads(button_id)["index"]
                return create_graph_figure(
                    pos, active_variables, highlight_var=hovered_var
                )
            else:
                layout_name = button_id.split("-")[0]
                pos = layouts[layout_name](G)
        return create_graph_figure(pos, active_variables)

    def create_graph_figure(pos, active_variables, highlight_var=None):
        edge_x, edge_y = [], []
        highlight_edge_x, highlight_edge_y = [], []
        edge_label_x, edge_label_y, edge_labels, edge_hovers = [], [], [], []
        node_x, node_y, node_text, node_hovertext = [], [], [], []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_label = G.edges[edge].get("label", "")
            active_edge_vars = [
                var for var in edge_label.split("\n") if var in active_variables
            ]
            var_count = len(active_edge_vars)

            if var_count > 0:
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                edge_label_x.append(mid_x)
                edge_label_y.append(mid_y)
                edge_labels.append(f"{var_count} vars")
                edge_hovers.append("\n".join(active_edge_vars))

                if highlight_var and highlight_var in active_edge_vars:
                    highlight_edge_x.extend([x0, x1, None])
                    highlight_edge_y.extend([y0, y1, None])
                else:
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

        node_adjacencies = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            hover_text = f"Agent: {node}<br>"
            hover_text += f"Modules: {', '.join(list(vars_by_module[node].keys()))}<br>"
            # ... (rest of hover text preparation)

            node_hovertext.append(hover_text)
            node_adjacencies.append(
                len(list(G.successors(node))) + len(list(G.predecessors(node)))
            )

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        highlight_edge_trace = go.Scatter(
            x=highlight_edge_x,
            y=highlight_edge_y,
            line=dict(width=2, color="red"),
            hoverinfo="none",
            mode="lines",
        )

        edge_label_trace = go.Scatter(
            x=edge_label_x,
            y=edge_label_y,
            mode="text",
            text=edge_labels,
            textposition="middle center",
            hoverinfo="text",
            hovertext=edge_hovers,
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            textposition="top center",
            mode="markers+text",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                reversescale=True,
                color=node_adjacencies,
                size=20,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        node_trace.hovertext = node_hovertext

        return go.Figure(
            data=[edge_trace, highlight_edge_trace, node_trace, edge_label_trace],
            layout=go.Layout(
                title="Agent Communication Network",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                dragmode="pan",
                height=800,
            ),
        )

    return app


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

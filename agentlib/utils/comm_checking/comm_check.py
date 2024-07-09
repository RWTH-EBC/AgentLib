import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, get_type_hints, get_origin, get_args

import networkx as nx
import plotly.graph_objects as go

from agentlib import AgentVariable
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


def order_vars_by_alias(
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]]
) -> Dict[Alias, List[Tuple[str, bool, Optional[Dict]]]]:
    vars_by_alias: Dict[Alias, List[Tuple[str, bool, Optional[Dict]]]] = {}
    for ag_id, modules in vars_by_module.items():
        for mod_id, ag_vars in modules.items():
            for var in ag_vars:
                alias = var.get("alias", var["name"])
                if alias not in vars_by_alias:
                    vars_by_alias[alias] = []
                vars_by_alias[alias].append(
                    (f"{ag_id}.{mod_id}.{var['name']}", var.get("shared", False), None)
                )
    return vars_by_alias


def create_comm_graph(configs):
    configs_: List[Dict] = create_configs(configs)
    vars_by_module: Dict[AG_ID, Dict[MOD_ID, List[Dict]]] = collect_vars(configs_)
    vars_by_alias: Dict[Alias, List[Tuple[str, bool, Optional[Dict]]]] = (
        order_vars_by_alias(vars_by_module)
    )

    g = nx.Graph()
    for ag_id in vars_by_module:
        g.add_node(ag_id)

    for alias, var_list in vars_by_alias.items():
        if len(var_list) > 1:
            for i in range(len(var_list) - 1):
                for j in range(i + 1, len(var_list)):
                    ag1, mod1, var1 = var_list[i][0].split(".")
                    ag2, mod2, var2 = var_list[j][0].split(".")
                    if ag1 != ag2:
                        g.add_edge(ag1, ag2, label=alias)

    return g, vars_by_module


def create_interactive_graph(G, vars_by_module):
    pos = nx.spring_layout(G)

    edge_x, edge_y = [], []
    edge_label_x, edge_label_y, edge_labels = [], [], []
    node_x, node_y, node_text, node_hovertext = [], [], [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        edge_label = G.edges[edge].get("label", "")
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_label_x.append(mid_x)
        edge_label_y.append(mid_y)
        edge_labels.append(edge_label)

    node_adjacencies = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Prepare hover information
        modules = list(vars_by_module[node].keys())
        all_vars = set()
        for mod_vars in vars_by_module[node].values():
            all_vars.update(var["name"] for var in mod_vars)

        shared_vars = set()
        internal_shared_vars = set()
        unique_vars = set()

        for var in all_vars:
            var_count = sum(
                1
                for mod_vars in vars_by_module[node].values()
                if var in [v["name"] for v in mod_vars]
            )
            if var_count > 1:
                internal_shared_vars.add(var)

            is_shared = any(
                var in [v["name"] for v in mod_vars]
                for other_agent in vars_by_module
                if other_agent != node
                for mod_vars in vars_by_module[other_agent].values()
            )
            if is_shared:
                shared_vars.add(var)
            else:
                unique_vars.add(var)

        hover_text = f"Agent: {node}<br>"
        hover_text += f"Modules: {', '.join(modules)}<br>"
        hover_text += f"Shared variables: {', '.join(shared_vars)}<br>"
        hover_text += f"Internally shared variables: {', '.join(internal_shared_vars - shared_vars)}<br>"
        hover_text += f"Unique variables: {', '.join(unique_vars)}"

        node_hovertext.append(hover_text)
        node_adjacencies.append(len(list(G.neighbors(node))))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
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
            size=15,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    edge_label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode="text",
        text=edge_labels,
        textposition="middle center",
        hoverinfo="none",
    )

    node_trace.hovertext = node_hovertext

    fig = go.Figure(
        data=[edge_trace, node_trace, edge_label_trace],
        layout=go.Layout(
            title="Agent Communication Network",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",  # Enable panning
        ),
    )

    # Add buttons for zoom and pan
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(args=[{"dragmode": "pan"}], label="Pan", method="relayout"),
                    dict(args=[{"dragmode": "zoom"}], label="Zoom", method="relayout"),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    return fig


def load_json_to_dict(input_data):
    """
    Loads JSON data from various input formats and returns a list of dictionaries.

    Args:
        input_data (str, dict, or list): The input data, which can be a file path, a JSON string, or a dictionary.

    Returns:
        list[dict]: A list of dictionaries containing the JSON data.
    """
    results = []

    # Ensure input_data is a list
    if not isinstance(input_data, list):
        input_data = [input_data]

    for item in input_data:
        if isinstance(item, str):
            # Check if the string is a file path
            if os.path.isfile(item):
                with open(item, "r") as file:
                    data = json.load(file)
            else:
                # Assume the string is a JSON string
                data = json.loads(item)
        elif isinstance(item, dict):
            data = item
        else:
            raise ValueError(
                "Input data must be a string (file path or JSON), or a dictionary."
            )

        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)

    return results


if __name__ == "__main__":
    directory_path = Path(
        r"D:\repos\AgentLib\examples\multi-agent-systems\room_mas\configs"
    )
    configs = load_json_to_dict([str(file) for file in directory_path.glob("*")])
    G, vars_by_module = create_comm_graph(configs)

    fig = create_interactive_graph(G, vars_by_module)
    fig.show()
    # pio.write_html(fig, file="agent_network.html", auto_open=True)
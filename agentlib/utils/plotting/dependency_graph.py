import logging
import os
import socket
import webbrowser
from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

from agentlib.core.datamodels import AgentVariable
from agentlib.core.errors import OptionalDependencyError

try:
    import dash
    from dash import html, dcc, Input, Output
    import dash_cytoscape as cyto
except ImportError:
    raise OptionalDependencyError("mas_dependency_graph", "interactive (needs dash-cytoscape)")

if TYPE_CHECKING:
    from agentlib.utils.multi_agent_system import LocalMASAgency


logger = logging.getLogger(__name__)


def get_port():
    """Find a free port on localhost."""
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_free = s.connect_ex(("localhost", port)) != 0
        if is_free:
            return port
        port += 1


def _get_var_mechanism(var: AgentVariable, cfg) -> str:
    """Determine which mechanism made a variable shared.

    Returns 'svf' if the variable's field is in shared_variable_fields,
    'shared' otherwise (i.e. the variable itself was configured as shared).
    """
    svf = set(cfg.shared_variable_fields)
    for field_name in cfg.model_fields:
        field_val = getattr(cfg, field_name, None)
        if isinstance(field_val, AgentVariable) and field_val is var:
            return "svf" if field_name in svf else "shared"
        elif isinstance(field_val, list):
            for item in field_val:
                if isinstance(item, AgentVariable) and item is var:
                    return "svf" if field_name in svf else "shared"
    return "shared"


def _extract_dependencies(
    mas: "LocalMASAgency",
) -> List[Tuple[str, str, str, str]]:
    """Extract individual variable dependencies between agents.

    Returns a list of (producer_agent, subscriber_agent, variable_label, mechanism_tag)
    where mechanism_tag is one of "shared+sub", "svf+sub", or "source".
    """
    shared_vars: Dict[str, Dict[str, str]] = defaultdict(dict)
    for agent_id, agent in mas._agents.items():
        for module in agent.modules:
            for var in module.config.get_variables():
                if var.shared:
                    label = var.alias or var.name
                    if label not in shared_vars[agent_id]:
                        shared_vars[agent_id][label] = _get_var_mechanism(var, module.config)

    subs: Dict[str, List[str]] = defaultdict(list)
    for agent_id, agent in mas._agents.items():
        for module in agent.modules:
            subscriptions = getattr(module.config, "subscriptions", None)
            if subscriptions:
                for sub_agent_id in subscriptions:
                    if sub_agent_id != agent_id and sub_agent_id not in subs[agent_id]:
                        subs[agent_id].append(sub_agent_id)

    deps: List[Tuple[str, str, str, str]] = []
    seen: set = set()
    for subscriber, producers in subs.items():
        for producer in producers:
            for var_label, mech in shared_vars.get(producer, {}).items():
                key = (producer, subscriber, var_label)
                if key not in seen:
                    seen.add(key)
                    deps.append((producer, subscriber, var_label, f"{mech}+sub"))

    for agent_id, agent in mas._agents.items():
        for module in agent.modules:
            for var in module.config.get_variables():
                src_agent = var.source.agent_id
                if src_agent is not None and src_agent != agent_id:
                    label = var.alias or var.name
                    key = (src_agent, agent_id, label)
                    if key not in seen:
                        seen.add(key)
                        deps.append((src_agent, agent_id, label, "source"))

    return deps


def run_dashboard(deps: List[Tuple[str, str, str, str]], agent_ids: List[str]):

    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    """Bootstraps the Dash application."""
    app = dash.Dash(__name__)

    unique_vars = sorted(list(set(label for _, _, label, _ in deps)))
    dropdown_options = [{"label": v, "value": v} for v in unique_vars]

    # Define base stylesheet
    BASE_STYLESHEET = [
        {
            'selector': '.agent',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': "#0F4877",
                'color': '#FFFFFF',
                'shape': 'round-rectangle',
                'width': '150px',
                'height': '50px',
                'font-weight': 'bold',
            },
        },
        {
            'selector': '.dependency',
            'style': {
                'curve-style': 'bezier',
                'control-point-distance': 35,
                'target-arrow-shape': 'triangle',
                'font-size': '12px',
                'text-rotation': 'autorotate',
                'text-margin-y': '-15px',
                'text-background-opacity': 0.7,
                'text-background-color': '#FFFFFF',
                'text-background-padding': '2px',
                'text-background-shape': 'roundrectangle',
                'transition-property': 'opacity, line-color, target-arrow-color, width',
                'transition-duration': '0.15s'
            },
        },
        {
            'selector': '.standard-dependency',
            'style': {
                'content': 'data(label)',
                'width': 3,
                'line-color': "#4E4E4E",
                'target-arrow-color': "#4E4E4E",
            }
        },
        {
            'selector': '.summary-dependency',
            'style': {
                'content': 'data(label)',
                'width': 3,
                'line-color': "#0F4877",
                'target-arrow-color': "#0F4877",
                'line-style': 'dashed',
            }
        },
        {
            'selector': '.highlighted-dependency',
            'style': {
                'content': 'data(label)',
                'line-color': "#D88A30",
                'target-arrow-color': '#D88A30',
                'width': 6,
                'opacity': 1,
                'z-index': 9000
            }
        },
        {
            'selector': '.dimmed',
            'style': {
                'opacity': 0.15
            }
        },
        {
            'selector': 'edge:selected',
            'style': {
                'width': 8,
                'line-color': '#D88A30',
                'target-arrow-color': '#D88A30',
                'opacity': 1,
                'z-index': 9999
            }
        }
    ]

    app.layout = html.Div([
        html.H2("AgentLib MAS Dependency Graph", style={"fontFamily": "sans-serif"}),

        html.Div([
            dcc.Dropdown(
                id='variable-dropdown',
                options=dropdown_options,
                clearable=True,
                placeholder="Select a variable to highlight..."
            ),
        ], style={'width': '350px', 'marginBottom': '10px', 'fontFamily': 'sans-serif'}),

        html.Div([

            html.Div(
                id='hover-info-box',
                children="",
                style={'opacity': '0'}
            ),

            cyto.Cytoscape(
                id='mas-dependency-graph',
                layout={'name': 'breadthfirst', 'directed': True},
                style={'width': '100%', 'height': '800px'},
                stylesheet=BASE_STYLESHEET
            ),

        ], style={'position': 'relative', 'border': '1px solid #eee', 'borderRadius': '5px'}),

        html.Div([

            html.Div("Legend:", style={'fontWeight': 'bold', 'marginBottom': '4px'}),

            html.Div([
                html.Span("[shared+sub]", style={'marginRight': '5px', 'fontFamily': 'monospace'}),
                html.Span("variable marked shared: true, received via subscription"),
            ], style={'marginBottom': '4px'}),

            html.Div([
                html.Span("[svf+sub]", style={'marginRight': '5px', 'fontFamily': 'monospace'}),
                html.Span("field in shared_variable_fields, received via subscription"),
            ], style={'marginBottom': '4px'}),

            html.Div([
                html.Span("[source]", style={'marginRight': '5px', 'fontFamily': 'monospace'}),
                html.Span("variable source.agent_id points to another agent"),
            ]),

        ], style={

            'fontFamily': 'sans-serif', 'fontSize': '13px',

            'padding': '10px', 'marginTop': '8px',

            'backgroundColor': '#f9f9f9', 'border': '1px solid #eee', 'borderRadius': '5px'

        }), 

    ])

    @app.callback(
        Output('mas-dependency-graph', 'elements'),
        Input('variable-dropdown', 'value')
    )
    def update_elements(selected_variable):
        elements = []

        for agent_id in agent_ids:
            elements.append({
                "data": {"id": str(agent_id), "label": str(agent_id)},
                "classes": "agent"
            })

        edges_by_pair = defaultdict(list)
        for src, tgt, label, tag in deps:
            edges_by_pair[(src, tgt)].append((label, tag))

        edge_id_counter = 0

        for (src, tgt), label_tags in edges_by_pair.items():
            remaining = list(label_tags)
            dim_class = " dimmed" if selected_variable else ""

            if selected_variable:
                matching = [(l, t) for l, t in remaining if l == selected_variable]
                if matching:
                    l, t = matching[0]
                    elements.append({
                        "data": {
                            "id": f"e{edge_id_counter}",
                            "source": str(src),
                            "target": str(tgt),
                            "label": f"{l} [{t}]",
                            "hover_details": f"{l} [{t}]"
                        },
                        "classes": "dependency highlighted-dependency"
                    })
                    edge_id_counter += 1
                    remaining.remove((l, t))

            if len(remaining) > 3:
                summary_text = f"{len(remaining)} variables...(click for details)"
                elements.append({
                    "data": {
                        "id": f"e{edge_id_counter}",
                        "source": str(src),
                        "target": str(tgt),
                        "label": summary_text,
                        "hover_details": ", ".join(f"{l} [{t}]" for l, t in remaining)
                    },
                    "classes": f"dependency summary-dependency{dim_class}"
                })
                edge_id_counter += 1
            else:
                for lbl, tag in remaining:
                    elements.append({
                        "data": {
                            "id": f"e{edge_id_counter}",
                            "source": str(src),
                            "target": str(tgt),
                            "label": f"{lbl} [{tag}]",
                            "hover_details": f"{lbl} [{tag}]"
                        },
                        "classes": f"dependency standard-dependency{dim_class}"
                    })
                    edge_id_counter += 1

        return elements

    @app.callback(
        Output('hover-info-box', 'style'),
        Output('hover-info-box', 'children'),
        Input('mas-dependency-graph', 'selectedEdgeData')
    )
    def display_selected_data(selected_edges):
        base_hud_style = {
            'position': 'absolute',
            'top': '20px',
            'right': '20px',
            'zIndex': 1000,
            'backgroundColor': 'rgba(255, 255, 255, 0.95)',
            'border': '1px solid #ccc',
            'borderRadius': '8px',
            'padding': '15px',
            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
            'width': '250px',
            'maxHeight': '400px',
            'overflowY': 'auto',
            'fontFamily': 'sans-serif',
            'fontSize': '14px',
            'pointerEvents': 'none',
            'transition': 'opacity 0.2s ease-in-out'
        }

        hidden_style = base_hud_style.copy()
        hidden_style['opacity'] = '0'

        visible_style = base_hud_style.copy()
        visible_style['opacity'] = '1'

        if not selected_edges:
            return hidden_style, ""

        edge_data = selected_edges[-1]

        details = edge_data.get('hover_details', 'Unknown')
        var_list = [v.strip() for v in details.split(",")]

        content = html.Div([
            html.Strong("Variables in this link:", style={'display': 'block', 'marginBottom': '10px'}),
            html.Ul([html.Li(var) for var in var_list], style={'margin': '0', 'paddingLeft': '20px', 'color': '#333'})
        ])

        return visible_style, content

    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port, use_reloader=False)


import multiprocessing

def show_dependency_graph(mas: "LocalMASAgency") -> multiprocessing.Process:
    """Starts the visualizer in a background process and returns the process."""
    # 1. Extract data BEFORE creating the process to avoid Pickling errors
    deps = _extract_dependencies(mas)
    agent_ids = list(mas._agents.keys())
    
    # 2. Spawn the process with simple, picklable data
    process = multiprocessing.Process(
        target=run_dashboard, 
        args=(deps, agent_ids)
    )
    
    # 3. Start and return the process
    process.start()
    return process
import socket
import threading
import webbrowser
from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING

from agentlib.core.errors import OptionalDependencyError

try:
    import dash
    from dash import html, dcc, Input, Output
    import dash_cytoscape as cyto
except ImportError:
    raise OptionalDependencyError("mas_dependency_graph", "interactive (needs dash-cytoscape)")

if TYPE_CHECKING:
    from agentlib.utils.multi_agent_system import LocalMASAgency


def get_port():
    """Find a free port on localhost."""
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_free = s.connect_ex(("localhost", port)) != 0
        if is_free:
            return port
        port += 1


def _extract_dependencies(
    mas: "LocalMASAgency",
) -> List[Tuple[str, str, str]]:
    """Extract individual variable dependencies between agents."""
    # 1. Collect shared variables per agent
    shared_vars: Dict[str, List[str]] = defaultdict(list)
    for agent_id, agent in mas._agents.items():
        for module in agent.modules:
            for var in module.config.get_variables():
                if var.shared:
                    label = var.alias or var.name
                    if label not in shared_vars[agent_id]:
                        shared_vars[agent_id].append(label)

    # 2. Collect subscriptions per agent (who they listen to)
    subs: Dict[str, List[str]] = defaultdict(list)
    for agent_id, agent in mas._agents.items():
        for module in agent.modules:
            subscriptions = getattr(module.config, "subscriptions", None)
            if subscriptions:
                for sub_agent_id in subscriptions:
                    if sub_agent_id != agent_id and sub_agent_id not in subs[agent_id]:
                        subs[agent_id].append(sub_agent_id)

    # 3. Cross-reference: edge from producer → subscriber for each shared variable
    deps: List[Tuple[str, str, str]] = []
    seen: set = set()
    for subscriber, producers in subs.items():
        for producer in producers:
            for var_label in shared_vars.get(producer, []):
                key = (producer, subscriber, var_label)
                if key not in seen:
                    seen.add(key)
                    deps.append(key)

    # 4. Also handle case where a variable has explicit source.agent_id set
    for agent_id, agent in mas._agents.items():
        for module in agent.modules:
            for var in module.config.get_variables():
                src_agent = var.source.agent_id
                if src_agent is not None and src_agent != agent_id:
                    label = var.alias or var.name
                    key = (src_agent, agent_id, label)
                    if key not in seen:
                        seen.add(key)
                        deps.append(key)

    return deps


def run_dashboard(mas: "LocalMASAgency"):
    """Bootstraps the Dash application."""
    app = dash.Dash(__name__)

    # Extract Agents as Nodes
    elements = []
    for agent_id in mas._agents.keys():
        elements.append({
            "data": {"id": str(agent_id), "label": str(agent_id)},
            "classes": "agent"
        })

    # Extract dependencies as edges
    deps = _extract_dependencies(mas)
    
    # Extract unique variables for the dropdown
    unique_vars = sorted(list(set(label for _, _, label in deps)))
    dropdown_options = [{"label": v, "value": v} for v in unique_vars]

    for i, (source_agent, target_agent, label) in enumerate(deps):
        elements.append({
            "data": {
                "id": f"e{i}",
                "source": str(source_agent),
                "target": str(target_agent),
                "label": label,
            },
            "classes": "dependency",
        })

    # Defined default stylesheet separately so it can be reused in the callback
    default_stylesheet = [
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
                'content': 'data(label)',
                'width': 3,
                'line-color': "#4E4E4E",
                'target-arrow-color': "#4E4E4E",
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'font-size': '12px',
                'color': '#333333',
                'text-rotation': 'autorotate',
                'text-margin-y': '-15px',
                'text-background-opacity': 0.7,
                'text-background-color': '#FFFFFF',
                'text-background-padding': '2px',
                'text-background-shape': 'roundrectangle',
                'transition-property': 'opacity, line-color, target-arrow-color, width',
                'transition-duration': '0.3s'
            },
        },
    ]

    app.layout = html.Div([
        html.H2("AgentLib MAS Dependency Graph", style={"fontFamily": "sans-serif"}),
        
        # Dropdown container
        html.Div([
            dcc.Dropdown(
                id='variable-dropdown',
                options=dropdown_options,
                clearable=True,
                placeholder="Select a variable to highlight..."
            )
        ], style={'width': '300px', 'marginBottom': '20px', 'fontFamily': 'sans-serif'}),

        cyto.Cytoscape(
            id='mas-dependency-graph',
            elements=elements,
            layout={'name': 'breadthfirst', 'directed': True},
            style={'width': '100%', 'height': '800px'},
            stylesheet=default_stylesheet
        ),
    ])

    # Callback to handle highlighting
    @app.callback(
        Output('mas-dependency-graph', 'stylesheet'),
        Input('variable-dropdown', 'value')
    )
    def update_stylesheet(selected_variable):
        if not selected_variable:
            return default_stylesheet
        
        # Copy the base styles
        highlight_style = default_stylesheet.copy()
        
        # Dim all edges
        highlight_style.append({
            'selector': '.dependency',
            'style': {
                'opacity': 0.15
            }
        })
        
        # Highlight the selected edge
        highlight_style.append({
            'selector': f'.dependency[label = "{selected_variable}"]',
            'style': {
                'line-color': "#D88A30",        # Strong Red
                'target-arrow-color': '#D88A30',
                'width': 6,                     # Thicker line
                'opacity': 1,                   # Full opacity
                'z-index': 9999                 # Bring to front
            }
        })
        
        return highlight_style

    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run(debug=False, port=port)


def show_dependency_graph(mas: "LocalMASAgency"):
    """Starts the visualizer in a background thread."""
    thread = threading.Thread(target=run_dashboard, args=(mas,), daemon=True)
    thread.start()
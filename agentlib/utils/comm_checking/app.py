import json

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


def create_dash_app(G: nx.DiGraph, vars_by_module):
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
        edge_traces = []
        edge_label_x, edge_label_y, edge_labels, edge_hovers = [], [], [], []
        node_x, node_y, node_text, node_hovertext = [], [], [], []

        # Define colors for different directions
        color_forward = "blue"
        color_backward = "red"
        color_bidirectional = "purple"

        # Collect all variables that are part of edges
        edge_variables = set()
        for edge in G.edges(data=True):
            edge_variables.update(edge[2].get("label", "").split("\n"))

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_label = edge[2].get("label", "")
            active_edge_vars = [
                var for var in edge_label.split("\n") if var in active_variables
            ]
            var_count = len(active_edge_vars)

            if var_count > 0:
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                edge_label_x.append(mid_x)
                edge_label_y.append(mid_y)

                # Check if there's a reverse edge
                reverse_edge = G.has_edge(edge[1], edge[0])
                edge_labels.append(f"{var_count} vars")

                if reverse_edge:
                    color = color_bidirectional
                else:
                    color = color_forward

                # Create colored hover text
                hover_text = "<br>".join(
                    [
                        f"<span style='color:"
                        f"{color_forward if not reverse_edge or not G.edges[edge[1], edge[0]].get('label', '').split('\n') else color_bidirectional if var in G.edges[edge[1], edge[0]].get('label', '').split('\n') else color_backward};'>{var}</span>"
                        for var in active_edge_vars
                    ]
                )
                edge_hovers.append(hover_text)

                # Create edge trace
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=1.5, color="#888"),
                    hoverinfo="none",
                    mode="lines+markers",
                    marker=dict(
                        size=15,
                        symbol="arrow",
                        angleref="previous",
                        color=color,
                    ),
                )

                edge_traces.append(edge_trace)

                # If it's bidirectional, add a reverse arrow
                if reverse_edge:
                    reverse_trace = go.Scatter(
                        x=[x1, x0, None],
                        y=[y1, y0, None],
                        line=dict(width=1.5, color="#888"),
                        hoverinfo="none",
                        mode="lines+markers",
                        marker=dict(
                            size=15,
                            symbol="arrow",
                            angleref="previous",
                            color=color,
                        ),
                    )
                    edge_traces.append(reverse_trace)

        node_adjacencies = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            hover_text = f"Agent: {node}<br>"
            hover_text += f"Modules: {', '.join(list(vars_by_module[node].keys()))}<br>"

            # Detailed module and variable information
            for mod_id, variables in vars_by_module[node].items():
                hover_text += f"<br>{mod_id}:<br>"
                for var in variables:
                    alias = var.get("alias", var["name"])
                    if alias in edge_variables:
                        hover_text += f"  - <i>{alias}</i><br>"
                    else:
                        hover_text += f"  - {alias}<br>"
            node_hovertext.append(hover_text)
            node_adjacencies.append(
                len(list(G.successors(node))) + len(list(G.predecessors(node)))
            )

        edge_label_trace = go.Scatter(
            x=edge_label_x,
            y=edge_label_y,
            mode="text",
            text=edge_labels,
            textposition="middle center",
            hoverinfo="text",
            hovertext=edge_hovers,
            textfont=dict(size=10),
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
            data=edge_traces + [node_trace, edge_label_trace],
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

import networkx as nx
import plotly.graph_objects as go


def create_dummy_graph():
    g = nx.Graph()
    g.add_node("Agent1")
    g.add_node("Agent2")
    g.add_node("Agent3")
    g.add_edge("Agent1", "Agent2", label="var1")
    g.add_edge("Agent2", "Agent3", label="var2")
    g.add_edge("Agent1", "Agent3", label="var3")
    return g


def visualize_graph(g):
    pos = nx.spring_layout(g)
    edge_x = []
    edge_y = []
    edge_labels = []

    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_labels.append(g.edges[edge]["label"])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            line_width=2,
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(g.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(adjacencies[0])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<br>Communication Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=edge_label,
                    showarrow=False,
                    xref="x",
                    yref="y",
                    x=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                    y=(pos[edge[0]][1] + pos[edge[1]][1]) / 2,
                    textangle=0,
                )
                for edge, edge_label in zip(g.edges(), edge_labels)
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()


if __name__ == "__main__":
    dummy_graph = create_dummy_graph()
    visualize_graph(dummy_graph)

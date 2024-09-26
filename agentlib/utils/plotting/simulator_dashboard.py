import io
import socket
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import List, Union

from agentlib.core.errors import OptionalDependencyError

try:
    import dash
    from dash import dcc, html, callback_context
    from dash.dependencies import Input, Output, State, ALL, ClientsideFunction
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
except ImportError:
    raise OptionalDependencyError("simulator_dashboard", "interactive")
import pandas as pd

from agentlib.core import datamodels

# Global variable to store the last read position for each file
file_positions = defaultdict(int)
data = {}  # Global variable to store loaded data


def get_port():
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            is_free = s.connect_ex(("localhost", port)) != 0
        if is_free:
            return port
        else:
            port += 1


def load_new_data(file_path: Path) -> pd.DataFrame:
    try:
        with file_path.open("r") as f:
            f.seek(file_positions[file_path])
            header = [0, 1, 2] if file_positions[file_path] == 0 else None
            new_data = f.read()
            file_positions[file_path] = f.tell()

        if not new_data:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(new_data), index_col=0, header=header)
        if header:
            df.columns = df.columns.droplevel(2)
        return df
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()


def update_data(existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    if not existing_data.empty:
        new_data.columns = existing_data.columns
    return pd.concat([existing_data, new_data], axis=0).drop_duplicates()


def format_time_axis(seconds):
    """Formats units on the time axis, scaling to minutes, hours etc. for longer
    simulations."""
    if seconds < 60 * 5:
        return seconds, "s", "{:.0f}"
    elif seconds < 3600 * 4:
        return seconds / 60, "min", "{:.1f}"
    elif seconds < 86400 * 3:
        return seconds / 3600, "h", "{:.1f}"
    elif seconds < 604800 * 2:
        return seconds / 86400, "d", "{:.1f}"
    elif seconds < 2592000 * 2:
        return seconds / 604800, "w", "{:.1f}"
    else:
        return seconds / 2592000, "mo", "{:.1f}"


def create_plot(df: pd.Series, title: str, plot_id: str) -> html.Div:
    # Convert index to seconds if it's not already
    if df.index.dtype != "float64":
        df.index = pd.to_numeric(df.index)

    # Determine the appropriate time unit
    time_range = df.index.max() - df.index.min()
    scaled_time, time_unit, tick_format = format_time_axis(time_range)

    # Scale the x-axis values
    scale_factor = time_range / scaled_time
    x_values = df.index / scale_factor

    return html.Div(
        [
            dcc.Graph(
                id={"type": "plot", "index": plot_id},
                figure={
                    "data": [
                        go.Scatter(x=x_values, y=df.values, mode="lines", name=title)
                    ],
                    "layout": go.Layout(
                        title=title,
                        xaxis={
                            "title": f"Time ({time_unit})",
                            "tickformat": tick_format,
                            "hoverformat": ".2f",
                        },
                        yaxis={"title": "Value"},
                        margin=dict(l=40, r=20, t=40, b=30),
                        height=250,
                        uirevision=plot_id,  # This helps maintain zoom state
                    ),
                },
                config={"displayModeBar": False},
                style={"height": "100%", "width": "100%"},
            )
        ]
    )


def create_layout(file_names: List[Union[str, Path]]) -> html.Div:
    file_names = [Path(n) for n in file_names]
    return html.Div(
        [
            dcc.Tabs(
                id="agent-tabs",
                children=[
                    dcc.Tab(label=file_name.stem, value=str(file_name))
                    for file_name in file_names
                ],
                value=str(file_names[0]) if file_names else None,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(id="tab-content"), width=12, lg=9, className="pr-lg-0"
                    ),
                    dbc.Col(
                        html.Div(id="variable-checkboxes", className="mt-3 mt-lg-0"),
                        width=12,
                        lg=3,
                        className="pl-lg-0",
                    ),
                ],
                className="mt-3",
            ),
            dcc.Interval(
                id="interval-component",
                interval=2.5 * 1000,
                n_intervals=0,
            ),
        ]
    )


index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .checkbox-scroll {
                max-height: calc(100vh - 100px);
                overflow-y: auto;
                padding-right: 15px;
            }
            @media (min-width: 992px) {
                .checkbox-scroll {
                    position: sticky;
                    top: 20px;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def simulator_dashboard(*file_names: Union[str, Path]):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_layout(file_names)
    app.index_string = index_string

    @app.callback(
        Output("variable-checkboxes", "children"), Input("agent-tabs", "value")
    )
    def update_checkboxes(selected_tab):
        if not selected_tab:
            return html.Div("Please select a tab to view variables.")

        file_path = Path(selected_tab)
        if str(file_path) not in data:
            data[str(file_path)] = pd.DataFrame()

        file_data = data[str(file_path)]
        checkbox_groups = []

        for causality in datamodels.Causality:
            try:
                causality_data = file_data[causality]
            except KeyError:
                continue

            checkboxes = [
                dbc.Checkbox(
                    id={
                        "type": "variable-checkbox",
                        "index": f"{causality.name}-{column}",
                    },
                    label=column,
                    value=True,
                )
                for column in causality_data.columns
            ]

            checkbox_groups.append(
                html.Div([html.H5(causality.name.capitalize()), html.Div(checkboxes)])
            )

        return html.Div(checkbox_groups, className="checkbox-scroll")

    @app.callback(
        Output("tab-content", "children"),
        Input("agent-tabs", "value"),
        Input("interval-component", "n_intervals"),
        Input({"type": "variable-checkbox", "index": ALL}, "value"),
        State({"type": "variable-checkbox", "index": ALL}, "id"),
    )
    def update_tab_content(selected_tab, n_intervals, checkbox_values, checkbox_ids):
        if not selected_tab:
            return html.Div(
                "Please select a tab to view data.", style={"padding": "20px"}
            )

        file_path = Path(selected_tab)
        if str(file_path) not in data:
            data[str(file_path)] = pd.DataFrame()

        new_data = load_new_data(file_path)
        if not new_data.empty:
            data[str(file_path)] = update_data(data[str(file_path)], new_data)

        file_data = data[str(file_path)]

        # Create a dictionary of selected variables
        selected_variables = {
            checkbox_id["index"]: value
            for checkbox_id, value in zip(checkbox_ids, checkbox_values)
        }

        sections = []
        for causality in [
            datamodels.Causality.output,
            datamodels.Causality.input,
            datamodels.Causality.local,
            datamodels.Causality.parameter,
        ]:
            try:
                causality_data = file_data[causality]
            except KeyError:
                continue

            plots = []
            for column in causality_data.columns:
                checkbox_key = f"{causality.name}-{column}"
                if selected_variables.get(checkbox_key, True):
                    plot_id = f"{causality.name}-{column}"
                    plots.append(
                        html.Div(
                            create_plot(causality_data[column], column, plot_id),
                            style={
                                "width": "33%",
                                "display": "inline-block",
                                "padding": "10px",
                            },
                        )
                    )

            if plots:
                sections.append(
                    html.Div(
                        [
                            html.H3(
                                causality.name.capitalize(),
                                style={"padding-left": "10px"},
                            ),
                            html.Div(
                                plots, style={"display": "flex", "flexWrap": "wrap"}
                            ),
                        ]
                    )
                )

        return html.Div(sections)

    port = get_port()
    webbrowser.open_new_tab(f"http://localhost:{port}")
    app.run_server(debug=False, port=port)

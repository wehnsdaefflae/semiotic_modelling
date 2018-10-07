# coding=utf-8
import random
import time


from dash import Dash, dependencies
import dash_core_components
import dash_html_components
from plotly import graph_objs


app = Dash(__name__)
app.layout = dash_html_components.Div([
    dash_core_components.Graph(id="live-graph_a", animate=True),
    dash_core_components.Interval(id="graph-update", interval=1000)
])


class DataStorage:
    X = []
    Y = []


@app.callback(dependencies.Output("live-graph_a", "figure"), events=[dependencies.Event("graph-update", "interval")])
def update_graph_a():
    new_x = len(DataStorage.X)
    new_y = 0 if len(DataStorage.Y) < 1 else DataStorage.Y[-1] + random.random() * 2. - 1.

    print("{:f}, {:f}".format(new_x, new_y))

    DataStorage.X.append(new_x)
    DataStorage.Y.append(new_y)

    data = graph_objs.Scatter(x=DataStorage.X, y=DataStorage.Y, name="scatter", mode="lines+markers")
    layout = graph_objs.Layout(xaxis={"range": [min(DataStorage.X), max(DataStorage.X)]}, yaxis={"range": [min(DataStorage.Y), max(DataStorage.Y)]})

    return {"data": [data], "layout": layout}


if __name__ == "__main__":
    app.run_server(debug=True)
    print("sth")

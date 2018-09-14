# coding=utf-8

from dash import Dash, dependencies
import dash_core_components
import dash_html_components
from plotly import graph_objs

from visualization_.live_dash_plotly.make_data import DataStorage

X = []
Y = []

X.append(1)
Y.append(1)

app = Dash(__name__)
app.layout = dash_html_components.Div([
    dash_core_components.Graph(id="live-graph_a", animate=True),
    dash_core_components.Interval(id="graph-update", interval=1000)
])


@app.callback(dependencies.Output("live-graph_a", "figure"), events=[dependencies.Event("graph-update", "interval")])
def update_graph_a():
    l, v = DataStorage.get_average()

    global X
    global Y

    #new_x = 1
    #new_y = Y[-1] * random.uniform(-.1, .1)

    new_x, new_y = DataStorage.get_average()

    X.append(X[-1] + new_x)
    Y.append(Y[-1] + new_y)

    data = graph_objs.Scatter(x=X, y=Y, name="scatter", mode="lines+markers")
    layout = graph_objs.Layout(xaxis={"range": [min(X), max(X)]}, yaxis={"range": [min(Y), max(Y)]})

    return {"data": [data], "layout": layout}


if __name__ == "__main__":
    app.run_server(debug=True)

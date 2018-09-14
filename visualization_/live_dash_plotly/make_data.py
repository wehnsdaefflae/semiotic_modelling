# coding=utf-8
import random
import time


from dash import Dash, dependencies
import dash_core_components
import dash_html_components
from plotly import graph_objs


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
    global X
    global Y

    #new_x = 1
    #new_y = Y[-1] * random.uniform(-.1, .1)

    new_x, new_y = DataStorage.get_average()

    print("{:f}, {:f}".format(new_x, new_y))

    X.append(new_x)
    Y.append(new_y)

    data = graph_objs.Scatter(x=X, y=Y, name="scatter", mode="lines+markers")
    layout = graph_objs.Layout(xaxis={"range": [min(X), max(X)]}, yaxis={"range": [min(Y), max(Y)]})

    return {"data": [data], "layout": layout}


class DataStorage:
    current_time = 0
    last_series = []

    @staticmethod
    def log(value):
        DataStorage.last_series.append(value)

    @staticmethod
    def get_average():
        l, s = len(DataStorage.last_series), sum(DataStorage.last_series)
        DataStorage.last_series.clear()
        return l, s


class DataGeneration:
    def __init__(self):
        self.value = 0.

    def get_value(self):
        self.value += random.random() * .1
        return self.value


def main():
    dg = DataGeneration()
    t = 0

    while True:
        v = dg.get_value()

        print("{:f}".format(v))

        DataStorage.log(v)

        time.sleep(.1)
        t += 1


if __name__ == "__main__":
    app.run_server(debug=True)
    main()

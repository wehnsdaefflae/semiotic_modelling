# coding=utf-8
import json
from typing import Tuple, Sequence, Dict

import dash_core_components
import dash_html_components
from dash import Dash, dependencies
from flask import request, Flask, jsonify
from plotly import graph_objs

from tools.functionality import get_min_max


class VisualizationModel:
    def __init__(self, axes: Sequence[Tuple[str, int, int]]):
        self._axes_order = tuple(_name for _name, _, _ in axes)
        self._axes_data = {_name: (dict(), _length, _width) for _name, _length, _width in axes}

    def get_axes_lengths(self) -> Tuple[Tuple[str, int], ...]:
        return tuple((_name, self._axes_data[_name][1]) for _name in self._axes_order)

    def new_plot(self, axis_name: str, plot_name: str):
        _axis_series, _length, _width = self._axes_data[axis_name]
        new_series = []
        _axis_series[plot_name] = new_series

    def get_plot_names(self, axis_name: str) -> Tuple[str, ...]:
        _axis_data = self._axes_data[axis_name]
        _named_series = _axis_data[0]
        return tuple(_named_series.keys())

    def get_plot(self, axis_name: str, plot_name: str) -> Tuple[Tuple[float, ...], ...]:
        _axis_series, _, _ = self._axes_data[axis_name]
        series = _axis_series[plot_name]
        return tuple(series)

    def add_data(self, axis_name: str, plot_name: str, *value: float):
        _axis_series, _length, _width = self._axes_data[axis_name]
        if len(value) != _width:
            raise ValueError("inconsistent width")

        series = _axis_series[plot_name]
        series.append(tuple(value))
        for _ in range(len(series) - _length):
            series.pop(0)


class VisualizationView:
    flask = Flask(__name__)

    # https://github.com/plotly/dash/issues/214
    dash = Dash(__name__, server=flask)

    dash.layout = dash_html_components.Div(children=[
        dash_html_components.Div(children=[
            dash_html_components.H2("Live Graphs", style={"float": "left"})
        ]),
        dash_html_components.Div(children=[
            dash_html_components.Div(id="graphs")
        ], className="row"),  # column!
        dash_core_components.Interval(
                id="graph-update",
                interval=1000
            )
        ], className="container", style={"width": "98%", "margin-left": 10, "margin-right": 10, "max-width": 50000}
    )

    model = None

    @staticmethod
    @flask.route("/init_axes", methods=["POST"])
    def init_axes():
        data = request.data
        print(f"received {str(data):s}")

        d = json.loads(data)
        axes = d["axes"]

        VisualizationView.model = VisualizationModel(axes)
        VisualizationView.model.new_plot("axis_dummy", "plot_dummy")
        return jsonify(f"initialized {str(axes):s}")

    @staticmethod
    @flask.route("/data", methods=["POST"])
    def add_data():
        if VisualizationView.model is None:
            raise ValueError("visualization not started")

        data = request.data
        print(f"received {str(data):s}")
        d = json.loads(data)
        axis_name = d["axis_name"]
        plot_name = d["plot_name"]
        values = d["values"]

        no_values = len(values)

        if no_values == 0:
            raise ValueError("no value passed")

        VisualizationView.model.add_data(axis_name, plot_name, *values)

        return jsonify(f"passed on {str(values):s} to plot '{plot_name:s}' in axis '{axis_name:s}'")

    @staticmethod
    @dash.callback(dependencies.Output("graphs", "children"), events=[dependencies.Event("graph-update", "interval")])
    def __update_graph():
        if VisualizationView.model is None:
            return

        graphs = []
        for _axis_name, _axis_len in VisualizationView.model.get_axes_lengths():
            axis_data = []
            y_min = float("inf")
            y_max = -y_min

            for _plot_name in VisualizationView.model.get_plot_names(_axis_name):
                series = VisualizationView.model.get_plot(_axis_name, _plot_name)
                _min, _max = get_min_max(series)
                y_min, y_max = min(y_min, _min), max(y_max, _max)

                data = graph_objs.Scatter(
                    x=list(range(_axis_len)),
                    y=series,
                    name=_axis_name + ", " + _plot_name
                )
                axis_data.append(data)

            layout = graph_objs.Layout(xaxis={"range": [0, _axis_len]}, yaxis={"range": [y_min - .1 * y_min, y_max + .1 * y_max]}, title=_axis_name)
            graphs.append(dash_html_components.Div(children=[
                dash_core_components.Graph(
                    id=_axis_name,
                    animate=True,
                    figure={"data": axis_data, "layout": graph_objs.Layout}
                )
            ]))

        axis_name = "axis_dummy"
        plot_name = "plot_dummy"

        # get from file
        series = VisualizationView.model.get_plot(axis_name, plot_name)
        length = len(series)

        unzipped = list(zip(*series))[0]

        data = graph_objs.Scatter(
            x=list(range(length)),
            y=unzipped,
            name="scatter",
            mode="lines+markers")

        layout = graph_objs.Layout(
            xaxis={"range": [0, length]},
            yaxis={"range": [0. if length < 1 else min(unzipped), 1. if length < 1 else max(unzipped)]})

        # todo: add all plots from one axis to this data
        return {"data": [data], "layout": layout}


if __name__ == "__main__":
    # VisualizationInterface.init()
    # VisualizationInterface.app.run_server(debug=True)
    # NewVisualization.flask.run(debug=True)
    # NewVisualization.dash.run_server(debug=True)

    # http://127.0.0.1:8050/data?axis_name=axis_dummy&plot_name=plot_dummy&values=[2.1]
    VisualizationView.dash.run_server(debug=True)
    print("over it")

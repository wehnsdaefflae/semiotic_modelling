# coding=utf-8
import json
from typing import Tuple, Sequence, Dict, List

import dash_core_components
import dash_html_components
from dash import Dash, dependencies
from flask import request, Flask, jsonify
from plotly import graph_objs

from tools.functionality import get_min_max
from tools.logger import Logger


class VisualizationModel:
    def __init__(self, axes: Sequence[Tuple[str, int]], length: int = -1):
        self.axes = tuple(_name for _name, _ in axes)
        self._axes_width = {_name: (dict(), _width) for _name, _width in axes}
        self._length = length

    def __len__(self) -> int:
        return self._length

    def new_plot(self, axis_name: str, plot_name: str) -> List[Tuple[float, ...]]:
        _axis_series, _width = self._axes_width[axis_name]
        new_series = []
        _axis_series[plot_name] = new_series
        return new_series

    def get_plot_names(self, axis_name: str) -> Tuple[str, ...]:
        _named_series, _ = self._axes_width[axis_name]
        return tuple(_named_series.keys())

    def get_plot(self, axis_name: str, plot_name: str) -> Tuple[Tuple[float, ...], ...]:
        _named_series, _ = self._axes_width[axis_name]
        series = _named_series[plot_name]
        return tuple(series)

    def add_data(self, axis_name: str, plot_name: str, *value: float):
        _named_series,  _width = self._axes_width[axis_name]
        if len(value) != _width:
            raise ValueError("inconsistent width")

        series = _named_series.get(plot_name)
        if series is None:
            series = self.new_plot("axis_dummy", "plot_dummy")

        series.append(tuple(value))
        for _ in range(len(series) - self._length):
            series.pop(0)


class VisualizationView:
    model = None

    # https://github.com/plotly/dash/issues/214
    flask = Flask(__name__)
    dash = Dash(__name__, server=flask)

    dash.layout = dash_html_components.Div(children=[
        dash_html_components.Div(children=[
            dash_html_components.H2("Live Graphs", style={"float": "left"})
        ]),
        dash_html_components.Div(children=[
            dash_html_components.Div(id="graphs")
        ], className="row"),  # todo: column!
        dash_core_components.Interval(
                id="graph-update",
                interval=1000
            )
        ], className="container", style={"width": "98%", "margin-left": 10, "margin-right": 10, "max-width": 50000}
    )

    @staticmethod
    @flask.route("/init_model", methods=["POST"])
    def init_model():
        data = request.data
        Logger.log(f"initializing {str(data):s}")

        d = json.loads(data)
        axes = d["axes"]
        length = d.get("length", -1)

        VisualizationView.model = VisualizationModel(axes, length=length)
        return jsonify(f"initialized {str(axes):s}, length {length:d}")

    @staticmethod
    @flask.route("/data", methods=["POST"])
    def add_data():
        if VisualizationView.model is None:
            raise ValueError("visualization model not initialized")

        data = request.data
        Logger.log(f"adding {str(data):s}")

        d = json.loads(data)
        axis_name = d["axis_name"]
        plot_name = d["plot_name"]
        values = d["values"]

        if len(values) == 0:
            raise ValueError("no values passed")

        VisualizationView.model.add_data(axis_name, plot_name, *values)

        return jsonify(f"passed on {str(values):s} to plot '{plot_name:s}' in axis '{axis_name:s}'")

    @staticmethod
    @dash.callback(dependencies.Output("graphs", "children"), events=[dependencies.Event("graph-update", "interval")])
    def __update_graph():
        if VisualizationView.model is None:
            return

        graphs = []
        _axis_len = len(VisualizationView.model)
        for _axis_name in VisualizationView.model.axes:
            axis_data = []

            y_min = float("inf")
            y_max = -y_min

            for _plot_name in VisualizationView.model.get_plot_names(_axis_name):
                series = VisualizationView.model.get_plot(_axis_name, _plot_name)
                _min, _max = get_min_max({_x for _p in series for _x in _p})
                y_min, y_max = min(y_min, _min), max(y_max, _max)

                for each_series in zip(*series):
                    data = graph_objs.Scatter(
                        x=list(range(_axis_len)),
                        y=each_series,
                        name=_axis_name + ", " + _plot_name
                    )
                    axis_data.append(data)

            y_margin = (y_max - y_min) * .1

            layout = graph_objs.Layout(
                xaxis={"range": [0, _axis_len]},
                yaxis={"range": [y_min - y_margin, y_max + y_margin]},
                title=_axis_name)

            graphs.append(dash_html_components.Div(children=[
                dash_core_components.Graph(
                    id=_axis_name,
                    animate=True,
                    figure={
                        "data": axis_data,
                        "layout": layout}
                )
            ]))

        return graphs


if __name__ == "__main__":
    VisualizationView.dash.run_server(debug=True)
    print("over it")
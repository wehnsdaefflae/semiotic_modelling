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
    def __init__(self, axes: Sequence[Tuple[str, int]], length: int = 0):
        self.axes = tuple(_name for _name, _ in axes)
        self._axes_width = {_name: (dict(), _width) for _name, _width in axes}
        self._length = length

    def __len__(self) -> int:
        return abs(self._length)

    def new_plot(self, axis_name: str, plot_name: str) -> Tuple[List[float], ...]:
        _named_series, _width = self._axes_width[axis_name]
        new_series = tuple([] for _ in range(_width))
        _named_series[plot_name] = new_series
        return new_series

    def get_plot_names(self, axis_name: str) -> Tuple[str, ...]:
        _named_series, _ = self._axes_width[axis_name]
        return tuple(_named_series.keys())

    def get_plot(self, axis_name: str, plot_name: str) -> Tuple[Sequence[float], ...]:
        _named_series, _ = self._axes_width[axis_name]
        series = _named_series[plot_name]
        return series

    def add_data(self, axis_name: str, plot_name: str, *value: float):
        _named_series, _width = self._axes_width[axis_name]
        if len(value) != _width:
            raise ValueError("inconsistent width")

        series = _named_series.get(plot_name)
        if series is None:
            series = self.new_plot(axis_name, plot_name)

        for _v, _s in zip(value, series):
            _s.append(_v)

        if 0 >= self._length:
            return

        for _s in series:
            del _s[:-self._length]


class VisualizationView:
    model = None    # type: VisualizationModel

    # https://github.com/plotly/dash/issues/214
    flask = Flask(__name__)
    dash = Dash(__name__, server=flask)

    axis_styles = dict()
    plot_styles = dict()

    _trailing = False

    dash.layout = dash_html_components.Div(children=[
        dash_html_components.Div(children=[
            dash_html_components.H2("semiotic modelling", style={"float": "left"})
        ]),
        dash_html_components.Div(children=[
            dash_html_components.Div(
                id="graphs"
            )
        ]),
        dash_core_components.Interval(
            id="graph-update",
            interval=1000
        )
    ],
        className="container",
        # style={"width": "98%", "margin-left": 10, "margin-right": 10, "max-width": 50000}
    )

    _iterations = 0

    @staticmethod
    @flask.route("/init_model", methods=["POST"])
    def init_model():
        data = request.data
        Logger.log(f"initializing {str(data):s}")

        d = json.loads(data)
        axes = d["axes"]
        length = d.get("length", 0)

        VisualizationView._trailing = length < 0
        VisualizationView.model = VisualizationModel(axes, length=abs(length))
        return jsonify(f"initialized {str(axes):s}, length {length:d}")

    @staticmethod
    @flask.route("/style", methods=["POST"])
    def style():
        data = request.data
        Logger.log(f"styling {str(data):s}")

        d = json.loads(data)

        VisualizationView.axis_styles.clear()
        VisualizationView.axis_styles.update(d["axes"])

        VisualizationView.plot_styles.clear()
        VisualizationView.plot_styles.update(d["plots"])

        return jsonify("styling done")

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
        return jsonify(f"added {str(values):s} to plot '{plot_name:s}' in axis '{axis_name:s}'")

    @staticmethod
    @flask.route("/tick", methods=["POST"])
    def tick():
        data = request.data
        Logger.log(f"progressing {str(data):s}")

        d = json.loads(data)
        steps = d.get("steps", 1)

        VisualizationView._iterations += steps
        return jsonify(f"progressed {steps:d}")

    @staticmethod
    @dash.callback(dependencies.Output("graphs", "children"), events=[dependencies.Event("graph-update", "interval")])
    def __update_graph():
        graphs = []
        if VisualizationView.model is None:
            return graphs

        _axis_len = len(VisualizationView.model)
        for _axis_name in VisualizationView.model.axes:
            axis_data = []

            y_min = float("inf")
            y_max = -y_min

            if _axis_len == 0:
                x_min = 0
                x_max = VisualizationView._iterations

            elif VisualizationView._trailing:
                x_min = max(VisualizationView._iterations - _axis_len, 0)
                x_max = max(VisualizationView._iterations, _axis_len)

            else:
                x_min = 0
                x_max = _axis_len

            this_plot_style = VisualizationView.plot_styles.get(_axis_name, dict())

            for _plot_name in VisualizationView.model.get_plot_names(_axis_name):
                plot_properties = this_plot_style.get(_plot_name, dict())

                series = VisualizationView.model.get_plot(_axis_name, _plot_name)
                _min, _max = get_min_max({_x for _p in series for _x in _p})
                y_min, y_max = min(y_min, _min), max(y_max, _max)

                for each_series in series:
                    data = graph_objs.Scatter(
                        **plot_properties,
                        x=list(range(x_min, x_max)),
                        y=each_series,
                        name=_plot_name
                    )
                    axis_data.append(data)

            axis_properties = VisualizationView.axis_styles.get(_axis_name, dict())

            y_margin = (y_max - y_min) * .1

            layout = graph_objs.Layout(
                **axis_properties,
                xaxis={
                    "range": [x_min, x_max]},
                yaxis={
                    "range": [y_min - y_margin, y_max + y_margin],
                    "title": _axis_name},
            )

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
    VisualizationView.dash.run_server(host="192.168.178.20", debug=True)
    print("over it")

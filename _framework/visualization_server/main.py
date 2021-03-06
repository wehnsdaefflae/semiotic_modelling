# coding=utf-8
import json
from typing import Tuple, Sequence, Dict, List, Any

import dash_core_components
import dash_html_components
from dash import Dash, dependencies
from flask import request, Flask, jsonify
from matplotlib.colors import hsv_to_rgb
from plotly import graph_objs
from plotly.basedatatypes import BasePlotlyType

from tools.logger import Logger
from tools.math_functions import distribute_circular

IP = "127.0.0.1"
# IP = "192.168.178.20"


# TODO: refactor visualization server

class VisualizationModel:
    def __init__(self, axes: Sequence[Tuple[str, int]], length: int = 0):
        self.axes = tuple(_name for _name, _ in axes)
        self._axes_width = {_name: (dict(), _width) for _name, _width in axes}
        self._is_trailing = length < 0
        self._length = abs(length)
        self.x_range = []

    def __len__(self) -> int:
        return abs(self._length)

    def _new_plot(self, axis_name: str, plot_name: str) -> Tuple[List[float], ...]:
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

    def add_batch(self, iteration: int, batch: Sequence[Tuple[str, str, Sequence[float]]]):
        self.x_range.append(iteration)

        if self._is_trailing:
            while iteration - self.x_range[0] >= self._length:
                self.x_range.pop(0)

        no_iterations = len(self.x_range)

        for axis_name, plot_name, values in batch:
            _named_series, _width = self._axes_width[axis_name]
            no_values = len(values)
            if no_values != _width:
                raise ValueError("inconsistent width")

            series = _named_series.get(plot_name)
            if series is None:
                series = self._new_plot(axis_name, plot_name)

            values = sorted(values)
            for _v, _s in zip(values, series):
                _s.append(_v)
                del _s[:-no_iterations]


def get_layout():
    return dash_html_components.Div(children=[
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


# todo: always plot average
class VisualizationView:
    _model = None    # type: VisualizationModel

    # https://github.com/plotly/dash/issues/214
    _flask = Flask(__name__)
    _dash = Dash(__name__, server=_flask)

    _axis_styles = dict()
    _plot_styles = dict()
    _dist_axes = None

    length = 0

    _dash.layout = get_layout()

    _iterations = 0

    @staticmethod
    @_flask.route("/init_model", methods=["POST"])
    def init_model():
        data = request.data
        Logger.log(f"initializing {str(data):s}")

        d = json.loads(data)
        axes = d["axes"]

        VisualizationView._dist_axes = {_axis_name for _axis_name, _, _is_dist in axes if _is_dist}
        VisualizationView._axis_styles.clear()
        VisualizationView._plot_styles.clear()
        VisualizationView.length = d.get("length", 0)

        axes_model = tuple((_axis_name, _width) for _axis_name, _width, _ in axes)
        VisualizationView._model = VisualizationModel(axes_model, length=VisualizationView.length)

        VisualizationView._dash.layout = get_layout()
        _iterations = 0

        return jsonify(f"initialized {str(axes):s}, length {VisualizationView.length:d}")

    @staticmethod
    @_flask.route("/style", methods=["POST"])
    def style():
        data = request.data
        Logger.log(f"styling {str(data):s}")

        d = json.loads(data)

        VisualizationView._axis_styles.clear()
        VisualizationView._axis_styles.update(d["axes"])

        VisualizationView._plot_styles.clear()
        VisualizationView._plot_styles.update(d["plots"])

        return jsonify("styling done")

    @staticmethod
    @_flask.route("/data", methods=["POST"])
    def add_data():
        if VisualizationView._model is None:
            raise ValueError("visualization model not initialized")

        data = request.data
        Logger.log(f"adding batch")

        d = json.loads(data)
        batch = d["batch"]
        iteration = d["iteration"]

        VisualizationView._model.add_batch(iteration, batch)

        return jsonify(f"added batch of size {len(batch):d}")

    @staticmethod
    def _get_concentration(_axis_name: str, this_plot_style: Dict[str, Any]) -> Sequence[BasePlotlyType]:
        axis_data = []

        for _j, _plot_name in enumerate(VisualizationView._model.get_plot_names(_axis_name)):
            series = VisualizationView._model.get_plot(_axis_name, _plot_name)
            no_series = len(series)
            half_plus_one = no_series // 2 + 1
            no_bands = no_series - half_plus_one + 1
            alpha = 1. / no_bands

            color = hsv_to_rgb((distribute_circular(_j), .7, .7))
            color_str = ", ".join([f"{int(_x * 255.):d}" for _x in color])
            fillcolor = f"rgba({color_str:s}, {alpha:.2f})"
            plot_properties = this_plot_style.get(_plot_name, {"fillcolor": fillcolor})

            for _i in range(no_bands):
                series_a = series[_i]
                series_b = series[_i + half_plus_one - 1]
                outline = series_a + series_b[::-1]
                range_a = VisualizationView._model.x_range
                each_range = range_a + range_a[::-1]

                data = graph_objs.Scatter(
                    **plot_properties,
                    showlegend=_i == 0,
                    x=each_range,
                    y=outline,
                    name=_plot_name,
                    fill="tozerox",
                    mode="lines",
                    line={
                        "color": f"rgba({color_str:s}, 1)",
                        "width": 0,
                        "shape": "hv",
                        #"shape": "spline",
                    },
                    # line={"color": "rgba(255, 255, 255, 0)"},
                )
                axis_data.append(data)

        return axis_data

    @staticmethod
    def _get_lines(_axis_name: str, this_plot_style: Dict[str, Any]) -> Sequence[BasePlotlyType]:
        axis_data = []

        for _j, _plot_name in enumerate(VisualizationView._model.get_plot_names(_axis_name)):
            color = hsv_to_rgb((distribute_circular(_j), .7, .7))
            color_str = ", ".join([f"{int(_x * 255.):d}" for _x in color])

            plot_properties = this_plot_style.get(_plot_name, dict())

            series = VisualizationView._model.get_plot(_axis_name, _plot_name)

            for each_series in series:
                data = graph_objs.Scatter(
                    **plot_properties,
                    showlegend=True,
                    x=VisualizationView._model.x_range,
                    y=each_series,
                    name=_plot_name,
                    mode="lines",
                    line={
                        "color": f"rgba({color_str:s}, 1)",
                        "width": 1,
                        "shape": "hv",
                        #"shape": "spline",
                    },
                )
                axis_data.append(data)

        return axis_data

    @staticmethod
    @_dash.callback(dependencies.Output("graphs", "children"), events=[dependencies.Event("graph-update", "interval")])
    def __update_graph():
        graphs = []
        if VisualizationView._model is None or len(VisualizationView._model.x_range) < 1:
            return graphs

        if VisualizationView.length < 0:
            x_min = max(0, VisualizationView._model.x_range[-1] + VisualizationView.length)
            x_max = x_min - VisualizationView.length

        elif 0 < VisualizationView.length:
            x_min = 0
            x_max = VisualizationView.length

        else:
            x_min = 0
            x_max = VisualizationView._model.x_range[-1]

        for _axis_name in VisualizationView._model.axes:
            this_plot_style = VisualizationView._plot_styles.get(_axis_name, dict())

            if _axis_name not in VisualizationView._dist_axes:
                axis_data = VisualizationView._get_lines(_axis_name, this_plot_style)

            else:
                axis_data = VisualizationView._get_concentration(_axis_name, this_plot_style)

            axis_properties = VisualizationView._axis_styles.get(_axis_name, dict())

            # todo: get axis, update with default values
            # todo: also set y axis in constructor
            # todo: try axis property "autorange=True"

            layout = graph_objs.Layout(
                **axis_properties,
                xaxis={
                    "range": [x_min, x_max],
                    "title": "iterations",
                },
                yaxis={
                    "title": _axis_name,
                    #"type": "log",
                    "autorange": True   # only works if Graph animate=False, see https://community.plot.ly/t/how-to-enable-automatic-autoscale/5276/9
                },
                legend={
                    "x": 1,
                    "y": 1
                }
            )

            graphs.append(dash_html_components.Div(children=[
                dash_core_components.Graph(
                    id=_axis_name,
                    animate=False,
                    animation_options={
                        "mode": "immediate",
                        #"frame": {
                        #    "duration": 200,
                        #    "redraw": False,
                        #},
                        #"transition": {
                        #    "duration": 0,
                        #}
                    },
                    figure={
                        "data": axis_data,
                        "layout": layout}
                )
            ]))

        return graphs


if __name__ == "__main__":
    VisualizationView._dash.run_server(host=IP, debug=True)
    print("over it")

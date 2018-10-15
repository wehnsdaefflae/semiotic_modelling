# coding=utf-8
import _thread
import json
import random
import time
from collections import deque
from typing import List, Tuple, Sequence, Union

import dash_core_components
import dash_html_components
from dash import Dash, dependencies
from flask import request, Response, Flask, jsonify
from plotly import graph_objs


class Borg:
    _instance_states = dict()

    def __init__(self):
        _class_state = Borg._instance_states.get(self.__class__)
        if _class_state is None:
            Borg._instance_states[self.__class__] = self.__dict__
        else:
            self.__dict__ = _class_state


class VisualizationModel:
    def __init__(self, axes: Sequence[Tuple[str, int]]):
        self._axis_sizes = axes
        self._series = {_name: dict() for _name, _ in axes}
        self._ranged = {_name: dict() for _name, _ in axes}

    def get_size(self, axis_name: str) -> int:
        for _each_name, _each_size in self._axis_sizes:
            if _each_name == axis_name:
                return _each_size
        raise ValueError(f"no axis called '{axis_name:s}'")

    def get_series_point(self, axis_name: str, plot_name: str) -> List[float]:
        if self._is_ranged(axis_name, plot_name):
            raise ValueError(f"plot {plot_name:s} is ranged")
        _axis_series = self._series[axis_name]
        return _axis_series[plot_name]

    def get_series_range(self, axis_name: str, plot_name: str) -> List[Tuple[float, float]]:
        if not self._is_ranged(axis_name, plot_name):
            raise ValueError(f"plot {plot_name:s} is not ranged")
        _axis_series = self._series[axis_name]
        return _axis_series[plot_name]

    def _is_ranged(self, axis_name: str, plot_name: str) -> bool:
        _accumulated_axis = self._ranged[axis_name]
        return _accumulated_axis[plot_name]

    def new_plot(self, axis_name: str, plot_name: str, is_ranged: bool = False):
        _axis_series = self._series[axis_name]
        _ranged_axis = self._ranged[axis_name]

        _size = self.get_size(axis_name)
        new_series = [] if _size < 1 else deque(maxlen=_size)
        _axis_series[plot_name] = new_series
        _ranged_axis[plot_name] = is_ranged

    def add_point(self, axis_name: str, plot_name: str, value: float):
        _series = self.get_series_point(axis_name, plot_name)
        _series.append(value)

    def add_range(self, axis_name: str, plot_name: str, mean: float, dev: float):
        _series = self.get_series_range(axis_name, plot_name)
        value = mean, dev
        _series.append(value)


class VisualizationView:
    flask = Flask(__name__)

    # https://github.com/plotly/dash/issues/214
    dash = Dash(__name__, server=flask)
    dash.layout = dash_html_components.Div([
        dash_core_components.Graph(id="live-graph_a", animate=True),
        dash_core_components.Interval(id="graph-update", interval=1000)
    ])

    model = None

    @staticmethod
    def initialize(axes: Sequence[Tuple[str, int]]):
        VisualizationView.model = VisualizationModel(axes)
        VisualizationView.model.new_plot("axis_dummy", "plot_dummy", is_ranged=False)
        VisualizationView.dash.run_server(debug=True)
        print("initialized")

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

        elif no_values == 1:
            VisualizationView.model.add_point(axis_name, plot_name, values[0])

        else:
            VisualizationView.model.add_range(axis_name, plot_name, *values[:2])

        return jsonify(f"passed on {str(values):s} to plot '{plot_name:s}' in axis '{axis_name:s}'")

    @staticmethod
    @dash.callback(dependencies.Output("live-graph_a", "figure"), events=[dependencies.Event("graph-update", "interval")])
    def _update_graph():
        if VisualizationView.model is None:
            return

        axis_name = "axis_dummy"
        plot_name = "plot_dummy"

        # get from file
        series = VisualizationView.model.get_series_point(axis_name, plot_name)
        data = graph_objs.Scatter(x=list(range(len(series))), y=list(series), name="scatter", mode="lines+markers")
        layout = graph_objs.Layout(
            xaxis={"range": [0, VisualizationView.model.get_size(axis_name)]},
            yaxis={"range": [0. if len(series) < 1 else min(series), 1. if len(series) < 1 else max(series)]})

        return {"data": [data], "layout": layout}


if __name__ == "__main__":
    # VisualizationInterface.init()
    # VisualizationInterface.app.run_server(debug=True)
    # NewVisualization.flask.run(debug=True)
    # NewVisualization.dash.run_server(debug=True)

    # http://127.0.0.1:8050/data?axis_name=axis_dummy&plot_name=plot_dummy&values=[2.1]
    VisualizationView.initialize([("axis_dummy", 100)])
    print("over it")

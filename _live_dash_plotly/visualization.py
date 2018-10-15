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
    def __init__(self, axes: Sequence[Tuple[str, int, int]]):
        self._axis_order = tuple(_name for _name, _, _ in axes)
        self._series = {_name: (dict(), _length, _width) for _name, _length, _width in axes}

    def new_plot(self, axis_name: str, plot_name: str):
        _axis_series, _length, _width = self._series[axis_name]

        new_series = []
        _axis_series[plot_name] = new_series

    def get_plot(self, axis_name: str, plot_name: str) -> Tuple[Tuple[float, ...], ...]:
        _axis_series, _, _ = self._series[axis_name]
        series = _axis_series[plot_name]
        return tuple(series)

    def add_data(self, axis_name: str, plot_name: str, *value: float):
        _axis_series, _length, _width = self._series[axis_name]
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

    dash.layout = dash_html_components.Div(
        children=[
            dash_core_components.Graph(
                id="live-graph_00",
                animate=True
            ),
            dash_core_components.Interval(
                id="graph-update_00",
                interval=1000
            ),
        ]
    )
    #dash.layout = dash_html_components.Div(
    #    children=[
    #        dash_html_components.Div(id="title"),
    #        dash_html_components.Div(id="graphs"),
    #    ]
    #)

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
    @dash.callback(dependencies.Output("live-graph_00", "figure"), events=[dependencies.Event("graph-update_00", "interval")])
    def _update_axis():
        if VisualizationView.model is None:
            return

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

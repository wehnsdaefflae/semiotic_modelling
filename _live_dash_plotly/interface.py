# coding=utf-8
import _thread
import random
import time
from collections import deque
from typing import List, Tuple, Sequence, Union

import dash_core_components
import dash_html_components
from dash import Dash, dependencies
from flask import request, Response, Flask
from plotly import graph_objs


class Borg:
    _instance_states = dict()

    def __init__(self):
        _class_state = Borg._instance_states.get(self.__class__)
        if _class_state is None:
            Borg._instance_states[self.__class__] = self.__dict__
        else:
            self.__dict__ = _class_state


class VisualizationView(Borg):
    flask = Flask(__name__)

    # https://github.com/plotly/dash/issues/214
    dash = Dash(__name__, server=flask)
    dash.layout = dash_html_components.Div([
        dash_core_components.Graph(id="live-graph_a", animate=True),
        dash_core_components.Interval(id="graph-update", interval=1000)
    ])

    def __init__(self, axes: Sequence[Tuple[str, int]]):
        super().__init__()
        VisualizationView.dash.run_server(debug=True)
        self.model = VisualizationModel(axes)

    @flask.route('/data')
    def add_data(self):
        # read multidict
        # differentiate point from range_data
        # add data to model
        # return response
        print(request.values)
        return Response('We received something…')

    @staticmethod
    @dash.callback(dependencies.Output("live-graph_a", "figure"), events=[dependencies.Event("graph-update", "interval")])
    def _update_graph():
        # get from file
        series = self.model.get_series_point("axis_dummy", "plot_dummy")
        data = graph_objs.Scatter(x=list(range(len(series))), y=series, name="scatter", mode="lines+markers")
        layout = graph_objs.Layout(
            xaxis={"range": [0, VisualizationInterface._size]},
            yaxis={"range": [0. if len(series) < 1 else min(series), 1. if len(series) < 1 else max(series)]})

        return {"data": [data], "layout": layout}


class VisualizationModel:
    def __init__(self, axes: Sequence[Tuple[str, int]]):
        self._axis_sizes = axes
        self._series = {_name: dict() for _name, _ in axes}
        self._ranged = {_name: dict() for _name, _ in axes}

    def _get_size(self, axis_name: str) -> int:
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

        _size = self._get_size(axis_name)
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


class NewVisualization:
    _size = 0
    _plots = dict()
    _path = ""
    flask = Flask(__name__)

    # https://github.com/plotly/dash/issues/214
    dash = Dash(__name__, server=flask)
    dash.layout = dash_html_components.Div([
        dash_core_components.Graph(id="live-graph_a", animate=True),
        dash_core_components.Interval(id="graph-update", interval=1000)
    ])

    @staticmethod
    @flask.route('/data')
    def get_data():
        print(request.values)
        return Response('We received something…')


class VisualizationInterface:
    _size = 0
    _queues = dict()
    _stacks = dict()
    _path = ""

    app = Dash(__name__)
    app.layout = dash_html_components.Div([
        dash_core_components.Graph(id="live-graph_a", animate=True),
        dash_core_components.Interval(id="graph-update", interval=1000)
    ])

    @staticmethod
    def init(data_folder_path: str, window_size: int):
        VisualizationInterface._path = data_folder_path
        # use semaphore, maybe?
        # folder name is full figure name
        # filename is axis name
        # column name is plot name
        VisualizationInterface._size = window_size

    @staticmethod
    def _get_key(figure_id: str, plot_id: str) -> str:
        return figure_id + "_" + plot_id

    @staticmethod
    def _add_series(key: str, new_series: Tuple[List[float], List[float]]):
        if key in VisualizationInterface._queues:
            raise ValueError("series already exists")
        VisualizationInterface._queues[key] = new_series

    @staticmethod
    def _get_series(key: str) -> Tuple[List[float], List[float]]:
        series = VisualizationInterface._queues.get(key)
        if series is None:
            mean, deviation = [], []
            series = mean, deviation
            VisualizationInterface._add_series(key, series)

        else:
            mean, deviation = series

        if key == "dummy":
            if len(mean) < 1:
                mean.append(0.)
            else:
                new_value = mean[-1] + (2. * random.random()) - 1.
                mean.append(new_value)
            for _ in range(len(mean) - VisualizationInterface._size):
                mean.pop(0)

        return series

    @staticmethod
    def _update_data(key: str, value: float, stack: bool):
        mean_list, deviation_list = VisualizationInterface._get_series(key)

        if stack:
            try:
                last_mean_value = mean_list[-1]
                last_deviation_value = deviation_list[-1]

            except IndexError:
                raise ValueError("no previous value to stack on.")

        else:
            last_mean_value = 0.
            last_deviation_value = 0.

        no_data_points = VisualizationInterface._stacks.get(key, 0)

        new_mean_value = (last_mean_value * no_data_points + value) / (no_data_points + 1)
        mean_list.append(new_mean_value)
        for _ in range(len(mean_list) - VisualizationInterface._size):
            mean_list.pop(0)

        deviation_value = (new_mean_value - value) ** 2.
        new_deviation_value = (last_deviation_value * no_data_points + deviation_value) / (no_data_points + 1)
        deviation_list.append(new_deviation_value)
        for _ in range(len(deviation_list) - VisualizationInterface._size):
            deviation_list.pop(0)

        VisualizationInterface._stacks[key] = no_data_points + 1

    @staticmethod
    @app.callback(dependencies.Output("live-graph_a", "figure"), events=[dependencies.Event("graph-update", "interval")])
    def _update_graph():
        # get from file
        mean_list, deviation_list = VisualizationInterface._get_series("dummy")
        data = graph_objs.Scatter(x=list(range(len(mean_list))), y=mean_list, name="scatter", mode="lines+markers")
        layout = graph_objs.Layout(
            xaxis={"range": [0, VisualizationInterface._size]},
            yaxis={"range": [0. if len(mean_list) < 1 else min(mean_list), 1. if len(mean_list) < 1 else max(mean_list)]})

        return {"data": [data], "layout": layout}


if __name__ == "__main__":
    # VisualizationInterface.init()
    # VisualizationInterface.app.run_server(debug=True)
    # NewVisualization.flask.run(debug=True)
    # NewVisualization.dash.run_server(debug=True)
    v = VisualizationView([("dummy_axis", 100)])
    print("over it")

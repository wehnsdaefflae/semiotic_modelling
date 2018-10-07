# coding=utf-8
import _thread
import time
from typing import List, Tuple

import dash_core_components
import dash_html_components
from dash import Dash, dependencies
from plotly import graph_objs


class VisualizationInterface:
    _size = 1000
    _queues = dict()
    _stacks = dict()

    app = Dash(__name__)
    app.layout = dash_html_components.Div([
        dash_core_components.Graph(id="live-graph_a", animate=True),
        dash_core_components.Interval(id="graph-update", interval=1000)
    ])

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
            series = [], []
            VisualizationInterface._add_series(key, series)
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
    VisualizationInterface.app.run_server(debug=True)
    print("over it")

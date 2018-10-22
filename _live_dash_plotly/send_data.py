# coding=utf-8
import random
import time
from typing import Sequence, Tuple, Dict, Any

import requests

from tools.functionality import Borg
from tools.logger import Logger

IP = "127.0.0.1"
# IP = "192.168.178.20"

URL = f"http://{IP}:8050/"


def initialize(axes: Sequence[Tuple[str, int]], length: int = 0):
    assert len(axes) >= 1
    params = {
        "axes": tuple((_name, _width, 1 < _width) for _name, _width in axes),
        "length": length,
    }  # nest dicts for axis properties
    r = requests.post(URL + "init_model?", json=params)
    return r.status_code, r.json()


def send_data(axis_name: str, plot_name: str, *values: float):
    params = {
        "axis_name": axis_name,
        "plot_name": plot_name,
        "values": values,
    }
    r = requests.post(URL + "data?", json=params)
    return r.status_code, r.json()


def update(steps: int = 1):
    params = {
        "steps": steps
    }
    r = requests.post(URL + "tick?", json=params)
    return r.status_code, r.json()


def style(axis_styles: Dict[str, Dict[str, Any]], plot_styles: Dict[str, Dict[str, Dict[str, Any]]]):
    params = {
        "axes": axis_styles,
        "plots": plot_styles
    }
    r = requests.post(URL + "style?", json=params)
    return r.status_code, r.json()


class SemioticVisualization:
    _last_plot = -1.
    _interval_seconds = 1.

    @staticmethod
    def initialize(axes: Sequence[str], no_experiments: int, length: int = 0):
        status, json_response = initialize(tuple((_name, no_experiments) for _name in axes), length=length)
        Logger.log(f"{status:d}\n{json_response:s}")

        axis_styles = dict()
        plot_styles = dict()
        status, json_response = style(axis_styles, plot_styles)
        Logger.log(f"{status:d}\n{json_response:s}")

    @staticmethod
    def plot(axis_name: str, plot_name: str, values: Sequence[float]):
        now = time.time()
        if SemioticVisualization._last_plot < 0. or now - SemioticVisualization._last_plot >= SemioticVisualization._interval_seconds:
            status, json_response = send_data(axis_name, plot_name, *values)
            Logger.log(f"{status:d}\n{json_response:s}")

            status, json_response = update()
            Logger.log(f"{status:d}\n{json_response:s}")

            SemioticVisualization._last_plot = now


def main():
    no_values = 10

    status, json_response = initialize([("individual", 1), ("concentration", no_values)], length=-100)
    print(f"{status:d}\n{json_response:s}")

    plot_styles = dict()
    status, json_response = style(dict(), {"individual": plot_styles})
    print(f"{status:d}\n{json_response:s}")

    values_01 = [1. for _ in range(no_values)]
    values_02 = [-1. for _ in range(no_values)]
    while True:
        for _i, _v in enumerate(values_01):
            status, json_response = send_data("individual", f"reading 01 {_i:02d}", _v)
            print(f"{status:d}\n{json_response:s}")

        for _i, _v in enumerate(values_02):
            status, json_response = send_data("individual", f"reading 02 {_i:02d}", _v)
            print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data("concentration", "reading 01", *values_01)
        print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data("concentration", "reading 02", *values_02)
        print(f"{status:d}\n{json_response:s}")

        for _i in range(no_values):
            values_01[_i] += random.random() * .2 - .1
            values_02[_i] += random.random() * .2 - .1

        status, json_response = update()
        print(f"{status:d}\n{json_response:s}")

        time.sleep(1.)


if __name__ == "__main__":
    main()

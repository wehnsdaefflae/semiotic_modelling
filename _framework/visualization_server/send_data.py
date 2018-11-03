# coding=utf-8
from typing import Sequence, Tuple, Dict, Any

import requests

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


def send_data(iteration: int, batch: Sequence[Tuple[str, str, Sequence[float]]]):
    params = {
        "batch": batch,
        "iteration": iteration,
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
    @staticmethod
    def initialize(axes: Sequence[str], no_experiments: int, length: int = 0):
        status, json_response = initialize(tuple((_name, no_experiments) for _name in axes), length=length)
        Logger.log(f"{status:d}\n{json_response:s}")

        axis_styles = dict()
        plot_styles = dict()
        status, json_response = style(axis_styles, plot_styles)
        Logger.log(f"{status:d}\n{json_response:s}")

    @staticmethod
    def plot(iteration: int, batch: Sequence[Tuple[str, str, Sequence[float]]]):
        status, json_response = send_data(iteration, batch)
        Logger.log(f"{status:d}\n{json_response:s}")

    @staticmethod
    def update(steps: int = 1):
        status, json_response = update(steps=steps)
        Logger.log(f"{status:d}\n{json_response:s}")

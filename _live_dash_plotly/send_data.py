# coding=utf-8
import random
import time
from typing import Sequence, Tuple, Dict, Any

import requests

from tools.functionality import Borg

URL = "http://192.168.178.20:8050/"
# URL = "http://localhost:8050/"


def initialize(axes: Sequence[Tuple[str, int]], length: int = 0):
    assert len(axes) >= 1
    params = {
        "axes": axes,
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


class SemioticVisualization(Borg):
    # make plotly properties generally passable over initialize's rest and fix them here
    pass


def main():
    status, json_response = initialize([("axis_dummy_01", 1), ("axis_dummy_02", 2), ("axis_dummy_03", 1)], length=-10)
    print(f"{status:d}\n{json_response:s}")

    plot_range_style = {
        "plot_dummy_01": {
            "mode": "none",
            "fill": "tonexty"},
        "plot_dummy_02": {
            "mode": "none",
            "fill": "none"},
        }

    axis_range_style = {"axis"}
    status, json_response = style(dict(), {"axis_dummy_01": dict()})
    print(f"{status:d}\n{json_response:s}")

    value_01 = random.random()
    value_02 = random.random()
    for _ in range(100):
        status, json_response = send_data("axis_dummy_01", "plot_dummy_01", value_01)
        print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data("axis_dummy_01", "plot_dummy_02", value_02)
        print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data("axis_dummy_02", "plot_dummy_01", value_01, value_02)
        print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data("axis_dummy_03", "plot_dummy_01", value_01 - value_02)
        print(f"{status:d}\n{json_response:s}")

        value_01 += random.random() * .2 - .1
        value_02 += random.random() * .2 - .1

        update()

        time.sleep(1.)


if __name__ == "__main__":
    main()

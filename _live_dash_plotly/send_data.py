# coding=utf-8
import random
import time
from typing import Sequence, Tuple, Dict, Any

import requests

from tools.functionality import Borg

# IP = "127.0.0.1"
IP = "192.168.178.20"

URL = f"http://{IP}:8050/"


def initialize(axes: Sequence[Tuple[str, int, bool]], length: int = 0):
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


def get_range_styles(no_points: int) -> Dict[str, Any]:
    styles = dict()
    half_plus_one = no_points // 2 + 1
    for each_range in range(no_points - half_plus_one):
        styles[f"start_plot_{each_range:02d}"] = {
            "mode": "lines",
            "fill": None,
            "showlegend": False,
            "line": {"color": "rgba(255, 255, 255, 0)"},
        }
        styles[f"end_plot_{each_range:02d}"] = {
            "mode": "lines",
            "fill": "tonexty",
            "fillcolor": "rgba(0, 100, 80, .2)",
            "line": {"color": "rgba(255, 255, 255, 0)"},
        }
    return styles


def get_ranges(points: Sequence[float]) -> Sequence[Tuple[float, float]]:
    s = sorted(points)
    no_points = len(points)
    half_plus_one = no_points // 2 + 1
    return tuple((s[_r], s[_r + half_plus_one]) for _r in range(no_points - half_plus_one))


def send_distribution(axis_name: str, points: Sequence[float]):
    ranges = get_ranges(points)
    for _i, (_s, _e) in enumerate(ranges):
        status, json_response = send_data(axis_name, f"start_plot_{_i:02d}", _s)
        print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data(axis_name, f"end_plot_{_i:02d}", _e)
        print(f"{status:d}\n{json_response:s}")


def density_range():
    status, json_response = initialize([("axis_dummy_01", 1)], length=-10)
    print(f"{status:d}\n{json_response:s}")

    no_points = 5
    range_style = get_range_styles(no_points)

    status, json_response = style(dict(), {"axis_dummy_01": range_style})
    print(f"{status:d}\n{json_response:s}")

    values = [random.random() for _ in range(no_points)]

    for _i in range(100):
        send_distribution("axis_dummy_01", values)

        for _j in range(no_points):
            values[_j] += random.random() * .2 - .1

        update()
        time.sleep(1.)


def simple_range():
    no_values = 1

    status, json_response = initialize([("axis_dummy_01", 1, False), ("axis_dummy_02", no_values, True)], length=0)
    print(f"{status:d}\n{json_response:s}")

    plot_styles = dict() #{
#        "plot_dummy_01": {
#            "mode": "lines",
#        },
#        "plot_dummy_02": {
#            "mode": "lines",
#        },
#    }
    status, json_response = style(dict(), {"axis_dummy_01": plot_styles})
    print(f"{status:d}\n{json_response:s}")

    values = [random.random() for _ in range(no_values)]
    for _ in range(100):
        for _i, _v in enumerate(values):
            status, json_response = send_data("axis_dummy_01", f"plot_dummy_{_i:02d}", _v)
            print(f"{status:d}\n{json_response:s}")

        status, json_response = send_data("axis_dummy_02", "plot_dummy", *values)
        print(f"{status:d}\n{json_response:s}")

        for _i in range(no_values):
            values[_i] += random.random() * .2 - .1

        update()

        time.sleep(1.)


if __name__ == "__main__":
    simple_range()
    # density_range()

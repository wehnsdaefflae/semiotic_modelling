# coding=utf-8
import random
import time
from typing import Sequence, Tuple

import requests

URL = "http://127.0.0.1:8050/"


def initialize(axes: Sequence[Tuple[str, int]], length: int):
    assert len(axes) >= 1
    params = {
        "axes": axes,
        "length": length,
    }
    r = requests.post(URL + "init_model?", json=params)
    return r.status_code, r.json()


def send_data(axis_name: str, plot_name: str, *values: float):
    assert len(values) == 1
    params = {
        "axis_name": axis_name,
        "plot_name": plot_name,
        "values": values,
    }
    r = requests.post(URL + "data?", json=params)
    return r.status_code, r.json()


def main():
    status, json_response = initialize([("axis_dummy", 1)], 100)
    print(f"{status:d}\n{json_response:s}")

    value = random.random()
    for _ in range(1000):
        status, json_response = send_data("axis_dummy", "plot_dummy", value)
        value += random.random() * .2 - .1
        print(f"{status:d}\n{json_response:s}")
        time.sleep(.5)


if __name__ == "__main__":
    main()

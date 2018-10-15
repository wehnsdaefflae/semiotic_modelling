# coding=utf-8
import random
import time

import requests

URL = "http://127.0.0.1:8050/data?"


def send_data(axis_name: str, plot_name: str, *values: float):
    assert len(values) == 1
    params = {
        "axis_name": axis_name,
        "plot_name": plot_name,
        "values": values,
    }
    r = requests.post(URL, json=params)
    return r.status_code, r.json()


def main():
    value = random.random()
    for _ in range(1000):
        status, json_response = send_data("axis_dummy", "plot_dummy", value)
        value += random.random() * .2 - .1
        print(f"{status:d}")
        print(json_response)
        time.sleep(.5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding=utf-8
import itertools
import json
from math import sqrt
from typing import Sequence, Tuple, List, Generator, Optional

from matplotlib import pyplot

from data.data_processing import series_generator

RANGE = Tuple[float, float]
POINT = Tuple[float, ...]
SAMPLE = Tuple[POINT, float]
AREA = Tuple[POINT, POINT]
PRIORITY_ELEMENT = Tuple[float, POINT, AREA]


# TODO: make into generator with send
def stateful_optimizer(ranges: Sequence[RANGE], limit: int = 1000) -> Generator[POINT, Optional[float], None]:
    dimensionality = len(ranges)                                        # type: int
    origin = tuple(min(_x) for _x in ranges)                            # type: POINT
    destination = tuple(max(_x) for _x in ranges)                       # type: POINT
    complete_region = origin, destination                               # type: AREA

    def __check_edges(_point_a: POINT, _point_b: POINT):
        len_a, len_b = len(_point_a), len(_point_b)                     # type: int, int
        if not (len_a == len_b == dimensionality):
            raise ValueError("Not all edges have a dimensionality of {:d}.".format(dimensionality))

    def __diagonal(_region: AREA) -> float:
        point_a, point_b = region                                       # type: POINT, POINT
        __check_edges(point_a, point_b)
        return sqrt(sum((point_a[_i] - point_b[_i]) ** 2. for _i in range(dimensionality)))

    def __enqueue(_value: float, _center: POINT, _region: AREA):
        no_values = len(priority_list)                                  # type: int
        priority = __diagonal(_region) * _value                         # type: float
        i = 0                                                           # type: int
        while i < no_values and priority < priority_list[i][0]:
            i += 1
        priority_element = priority, _center, _region                   # type: PRIORITY_ELEMENT
        priority_list.insert(i, priority_element)

    def __center(_region: AREA) -> POINT:
        point_a, point_b = _region                                      # type: POINT, POINT
        __check_edges(point_a, point_b)
        return tuple((point_a[_i] + point_b[_i]) / 2. for _i in range(dimensionality))

    def _divide(_borders: AREA, _center: POINT) -> Tuple[AREA, ...]:
        return tuple((_x, _center) for _x in itertools.product(*zip(*_borders)))

    genesis_element = 0., __center(complete_region), complete_region    # type: PRIORITY_ELEMENT
    priority_list = [genesis_element]                                   # type: List[PRIORITY_ELEMENT]
    cache_list = []                                                     # type: List[AREA]

    while True:
        if len(cache_list) < 1:
            _, center, region = priority_list.pop(0)                    # type: float, POINT, AREA
            sub_regions = _divide(region, center)                       # type: Tuple[AREA, ...]
            cache_list.extend(sub_regions)
            # cache_list = list(sub_regions) + cache_list

        current_region = cache_list.pop()                               # type: AREA
        current_center = __center(current_region)                       # type: POINT
        current_value = yield current_center

        __enqueue(current_value, current_center, current_region)
        while 0 < limit < len(priority_list):
            priority_list.pop()


def test_optimizer():
    with open("../configs/config.json", mode="r") as file:
        config = json.load(file)

    time_series = series_generator(config["data_dir"] + "binance/QTUMETH.csv")
    y_values = [_x[1] for _x in time_series]
    length = len(y_values)
    x_values = list(range(length))
    f = lambda _x: y_values[round(_x)]

    #length = 1000
    #x_values = list(range(length))
    #f = lambda _x: sin(_x * .07) + cos(_x * .03) + 5.
    #y_values = [f(x) for x in x_values]

    max_value = max(y_values)
    parameter_ranges = (0., length),
    optimizer = stateful_optimizer(parameter_ranges)

    pyplot.plot(x_values, y_values, color="white")

    parameters = optimizer.send(None)
    optimal_value = 0.
    for _i in range(1000000):
        value = f(*parameters)
        if optimal_value < value:
            print("{:05.2f}% of maximum after {:d} iterations".format(100. * value / max_value, _i))
            optimal_parameters = parameters
            optimal_value = value
            # pyplot.plot(optimal_parameters, [optimal_value], "o")

        parameters = optimizer.send(value)
        # pyplot.axvline(x=parameters[0], alpha=.2)
        pyplot.plot(parameters, [value], "o", alpha=.2, color="blue")
        pyplot.draw()
        pyplot.pause(.01)

    pyplot.show()


if __name__ == "__main__":
    test_optimizer()


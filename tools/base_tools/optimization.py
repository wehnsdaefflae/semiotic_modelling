#!/usr/bin/env python3
# coding=utf-8
import itertools
import json
import math
import random
from math import sqrt, sin, cos
from typing import Sequence, Tuple, List, Generator, Optional, Callable

import numpy
from matplotlib import pyplot

from data_generation.data_processing import series_generator
from tools.functionality import normalize_vector, signum, cartesian_distance, get_min_max

RANGE = Tuple[float, float]
POINT = Tuple[float, ...]
SAMPLE = Tuple[POINT, float]
AREA = Tuple[POINT, POINT]
PRIORITY_ELEMENT = Tuple[float, POINT, AREA]


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

        current_region = cache_list.pop()                               # type: AREA
        current_center = __center(current_region)                       # type: POINT
        current_value = yield current_center
        assert current_value >= 0.

        __enqueue(current_value, current_center, current_region)
        del(priority_list[limit:])


class GradientOptimizer:
    def __init__(self, ranges: Sequence[RANGE], step_size: float):
        self._ranges = ranges
        self._step_size = step_size
        self._last_value = 0.
        self._this_parameters = tuple(sum(_range) / 2. for _range in ranges)
        self._last_step = tuple(_i + 1. for _i in range(len(ranges)))
        self._initial = True

    def optimize(self, this_value: float) -> POINT:
        if self._initial:
            self._initial = False

        else:
            value_difference = this_value - self._last_value
            for _i, _s in enumerate(self._last_step):
                self._last_step[_i] = value_difference * _s

        self._last_step = [_x for _x in normalize_vector(self._last_step, target_length=self._step_size)]
        print(self._last_step)
        self._this_parameters = tuple(_p + _c for _p, _c in zip(self._this_parameters, self._last_step))
        self._last_value = this_value
        return self._this_parameters


def gradient_optimizer(ranges: Sequence[RANGE], step_size: float) -> Generator[POINT, Optional[float], None]:
    assert 0. < step_size
    go = GradientOptimizer(ranges, step_size)
    value = 0.

    while True:
        value = yield go.optimize(value)


def test_optimizer_2d():
    length = 1000
    x_values = list(range(length))
    f = lambda _x: sin(_x * .07) + cos(_x * .03) + 5.
    y_values = [f(x) for x in x_values]

    pyplot.plot(x_values, y_values, color="C0")

    max_value = max(y_values)
    parameter_ranges = (0., length),
    # optimizer = stateful_optimizer(parameter_ranges)
    optimizer = gradient_optimizer(parameter_ranges, 1.)

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

        pyplot.plot(parameters, [value], "o", alpha=.2, color="blue")
        pyplot.draw()
        pyplot.pause(.01)

        parameters = optimizer.send(value)
        # pyplot.axvline(x=parameters[0], alpha=.2)

    pyplot.show()


def plot_surface(axis: pyplot.Axes.axes, _fun: Callable[[float, float], float], dim_ranges: Tuple[Tuple[float, float], Tuple[float, float]], colormap=None, resize: bool = False):
    _x = numpy.linspace(dim_ranges[0][0], dim_ranges[0][1], endpoint=True, num=100)
    _y = numpy.linspace(dim_ranges[1][0], dim_ranges[1][1], endpoint=True, num=100)

    _X, _Y = numpy.meshgrid(_x, _y)

    _z = numpy.array(tuple(_fun(__x, __y) for __x, __y in zip(numpy.ravel(_X), numpy.ravel(_Y))))

    _Z = _z.reshape(_X.shape)

    if resize:
        z_min, z_max = get_min_max(_z)
        z_margin = (z_max - z_min) * .1
        min_margin = z_min - z_margin
        max_margin = z_max + z_margin

        try:
            axis.set_zlim((min_margin, max_margin))
        except ValueError:
            print("infinite axis value")

    if colormap is None:
        return axis.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False)
    return axis.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False, cmap=colormap)


def test_optimizer_3d():
    dim_range = -10., 10.

    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    plot_axis = fig.add_subplot(111, projection='3d')

    go = gradient_optimizer([dim_range, dim_range], .1)

    # fun = lambda _x, _y: 10. + 1. * _x ** 1. + 1. * _y ** 1. + 4. * _x * _y + 1. * _x ** 2. + -2.6 * _y ** 2.
    fun = lambda _x, _y: cos((_x + 0.) / (1. * math.pi)) + cos((_y + 0.) / (1. * math.pi)) - cos((_x + 0.) / (.5 * math.pi)) - cos((_y + 0.) / (.5 * math.pi)) - _x
    plot_surface(plot_axis, fun, (dim_range, dim_range))
    pyplot.pause(.001)
    pyplot.draw()

    iterations = 0

    x, y = go.send(None)
    while True:
        z = fun(x, y)
        plot_axis.scatter([x], [y], [z], alpha=.5, color="black")

        x, y = go.send(z)

        pyplot.pause(.25)
        pyplot.draw()

        iterations += 1

    pyplot.show()


if __name__ == "__main__":
    test_optimizer_3d()


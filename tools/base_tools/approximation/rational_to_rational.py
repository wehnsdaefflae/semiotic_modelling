# coding=utf-8
import itertools
import math
import random
from collections import deque
from functools import reduce
from math import cos
from typing import Sequence, Tuple, Callable

import numpy
from matplotlib import pyplot

from tools.base_tools.approximation.functions import MultiplePolynomialFunction, Function, LinearFunction, MultipleLinearFunction

from tools.functionality import smear, get_min_max, combinations
from tools.timer import Timer


class FulcrumApproximation:
    pass
    # TODO: implement


class GradientDescentApproximation:
    pass
    # https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/


class SingleLinearRegression:
    # https://towardsdatascience.com/implementation-linear-regression-in-python-in-5-minutes-from-scratch-f111c8cc5c99
    def __init__(self, past_scope: int = -1, learning_drag: int = -1):
        self._past_scope = past_scope
        self._learning_drag = learning_drag
        self._mean_x = 0.
        self._mean_y = 0.
        self._variance_x = 0.
        self._cross_variance_xy = 0.
        self.coefficients = [0., 0.]

    def derivation_output(self) -> float:
        return self.coefficients[1]

    def fit(self, input_value: float, output_value: float, past_scope: int = -1, learning_drag: int = -1) -> float:
        assert self._past_scope >= 0 or past_scope >= 0
        assert self._learning_drag >= 0 or learning_drag >= 0
        _past_scope = max(self._past_scope, past_scope)
        _learning_drag = max(self._learning_drag, learning_drag)

        error = abs(self.output(input_value) - output_value)

        dy = output_value - self._mean_y
        dx = input_value - self._mean_x
        self._variance_x = smear(self._variance_x, dx ** 2., _past_scope)                                 # remove smear?
        self._cross_variance_xy = smear(self._cross_variance_xy, dx * dy, _past_scope)

        self._mean_x = smear(self._mean_x, input_value, _past_scope)
        self._mean_y = smear(self._mean_y, output_value, _past_scope)

        a1 = smear(self.coefficients[1], 0. if self._variance_x == 0. else self._cross_variance_xy / self._variance_x, _learning_drag)
        a0 = smear(self.coefficients[0], self._mean_y - a1 * self._mean_x, _learning_drag)
        self.coefficients[0] = a0
        self.coefficients[1] = a1

        return error

    def output(self, input_value: float) -> float:
        # todo: fix regression drag
        return self.coefficients[0] + self.coefficients[1] * input_value


class MultipleRegression:
    def __init__(self, input_dimensionality: int):
        self._in_dim = input_dimensionality

    def derivation_output(self, input_values: Sequence[float], derive_by: int) -> float:
        raise NotImplementedError()

    def _fit(self, input_values: Sequence[float], output_value: float, past_scope: int, learning_drag: int) -> float:
        raise NotImplementedError()

    def fit(self, input_values: Sequence[float], output_value: float, past_scope: int = -1, learning_drag: int = -1) -> float:
        assert len(input_values) == self._in_dim
        return self._fit(input_values, output_value, past_scope, learning_drag)

    def _output(self, input_values: Sequence[float]) -> float:
        raise NotImplementedError()

    def output(self, input_values: Sequence[float]) -> float:
        assert len(input_values) == self._in_dim
        return self._output(input_values)


class MultipleLinearRegression(MultipleRegression):
    def __init__(self, input_dimensionality: int, past_scope: int = -1, learning_drag: int = -1):
        super().__init__(input_dimensionality)
        self.regressions = tuple(SingleLinearRegression(past_scope=past_scope, learning_drag=learning_drag) for _ in range(input_dimensionality))

    def derivation_output(self, input_values: Sequence[float], derive_by: int) -> float:
        reg = self.regressions[derive_by]
        return reg.derivation_output()

    def _output(self, input_values: Sequence[float]) -> float:
        return sum(_regression.output(_x) for _regression, _x in zip(self.regressions, input_values))

    def _fit(self, input_values: Sequence[float], output_value: float, past_scope: int, learning_drag: int) -> float:
        _subtract = tuple(_each_regression.output(_each_input) for _each_regression, _each_input in zip(self.regressions, input_values))
        error = 0.
        for _i, (_each_input, _each_regression) in enumerate(zip(input_values, self.regressions)):
            _each_output = output_value - sum(_subtract[__i] for __i in range(self._in_dim) if _i != __i)
            error += _each_regression.fit(_each_input, _each_output, past_scope=past_scope, learning_drag=learning_drag)
        return error


class MultiplePolynomialFromLinearRegression(MultipleRegression):
    def __init__(self, input_dimensionality: int, degree: int, past_scope: int = -1., learning_drag: int = -1):
        no_polynomial_coefficients = len(MultiplePolynomialFromLinearRegression.polynomial_inputs(tuple(0. for _ in range(input_dimensionality)), degree))
        # no_polynomial_parameters = sum(combinations(_i + 1, _i + input_dimensionality) for _i in range(degree))
        super().__init__(no_polynomial_coefficients)
        self._raw_in_dim = input_dimensionality
        self._degree = degree
        self._regression = MultipleLinearRegression(no_polynomial_coefficients, past_scope=past_scope, learning_drag=learning_drag)
        self._past_scope = past_scope
        self._learning_drag = learning_drag
        self._input_distribution = tuple(
            ((-1,),) if _i < 0 else tuple(itertools.combinations_with_replacement(range(self._raw_in_dim), _i + 1))
            for _i in range(-1, degree)
        )

    def derivation_output(self, input_values: Sequence[float], derive_by: int) -> float:
        # TODO: check derivation against source
        # get coefficients from regressions
        coefficients = tuple(
            [
                sum(
                    each_regression.coefficients[0]
                    for each_regression in self._regression.regressions
                )
            ] if _i == 0 else []
            for _i in range(self._degree + 1)
        )
        #print(coefficients)

        # format for derivation function
        lengths = tuple(combinations(_d + 1, _d + self._raw_in_dim) for _d in range(self._degree))
        this_l = 1
        for each_regression in self._regression.regressions:
            coefficients[this_l].append(each_regression.coefficients[1])
            if len(coefficients) >= lengths[this_l]:
                this_l += 1
                if this_l >= len(lengths):
                    break

        # derive coefficients
        derived_coefficients = MultiplePolynomialFunction.derive_coefficients(coefficients, self._input_distribution, derive_by)
        #print(derived_coefficients)

        # derivation
        polynomial_values = MultiplePolynomialFunction.polynomial_values(input_values, self._degree)
        assert len(polynomial_values) == self._degree
        s = 0.
        for _i, (_v, _c) in enumerate(zip(((0.,),) + polynomial_values, derived_coefficients)):
            s += sum(__c * __v ** _i for __c, __v in zip(_c, _v))
        return s

    @staticmethod
    def polynomial_inputs(input_values: Sequence[float], degree: int) -> Tuple[float, ...]:
        """
        generates exhaustive polynomial combinations up to defined degree
        for example input values (x, y, z) and degree 2:
        (
            x, y, z,
            x*y, x*z, y*z, x^2, y^2, z^2,
            x*y*z, x^2 * y, x^2 * z, y^2 * x, y^2 * z, z^2 * x, z^2 * y, x^3, y^3, z^3
        )
        """
        assert degree >= 1
        return tuple(
            reduce(lambda _x, _y: _x * _y, each_combination)
            for _i in range(degree)
            for each_combination in itertools.combinations_with_replacement(input_values, _i + 1)
        )

    def _output(self, input_values: Sequence[float]):
        return self._regression._output(input_values)

    def _fit(self, input_values: Sequence[float], output_value: float, past_scope: int, learning_drag: int) -> float:
        poly_inputs = MultiplePolynomialFromLinearRegression.polynomial_inputs(input_values, self._degree)
        return self._regression._fit(poly_inputs, output_value, past_scope, learning_drag)

    def fit(self, input_values: Sequence[float], output_value: float, past_scope: int = -1, learning_drag: int = -1) -> float:
        assert past_scope >= 0 or self._past_scope >= 0
        assert learning_drag >= 0 or self._learning_drag >= 0
        assert len(input_values) == self._raw_in_dim
        poly_inputs = MultiplePolynomialFromLinearRegression.polynomial_inputs(input_values, self._degree)
        assert len(poly_inputs) == self._in_dim
        _past_scope = max(past_scope, self._past_scope)
        _learning_drag = max(learning_drag, self._learning_drag)
        return self._fit(poly_inputs, output_value, _past_scope, _learning_drag)

    def output(self, input_values: Sequence[float]) -> float:
        assert len(input_values) == self._raw_in_dim
        poly_inputs = MultiplePolynomialFromLinearRegression.polynomial_inputs(input_values, self._degree)
        assert len(poly_inputs) == self._in_dim
        return self._output(poly_inputs)


class MultivariateRegression:
    def __init__(self, input_dimensionality: int, output_dimensionality: int, past_scope: int = -1, learning_drag: int = -1):
        self._in_dim = input_dimensionality
        self._out_dim = output_dimensionality
        self._past_scope = past_scope
        self._learning_drag = learning_drag

    def derivation_output(self, input_values: Sequence[float], derive_by: int) -> Tuple[float, ...]:
        raise NotImplementedError()

    def _fit(self, input_values: Sequence[float], output_values: Sequence[float], past_scope: int, learning_drag: int) -> Tuple[float, ...]:
        raise NotImplementedError()

    def fit(self, input_values: Sequence[float], output_values: Sequence[float], past_scope: int = -1, learning_drag: int = -1) -> Tuple[float, ...]:
        assert len(input_values) == self._in_dim
        assert len(output_values) == self._out_dim
        assert past_scope >= 0 or self._past_scope >= 0
        _past_scope = max(past_scope, self._past_scope)
        assert learning_drag >= 0 or self._learning_drag >= 0
        _learning_drag = max(learning_drag, self._learning_drag)
        return self._fit(input_values, output_values, _past_scope, _learning_drag)

    def _output(self, input_values: Sequence[float]) -> Tuple[float, ...]:
        raise NotImplementedError()

    def output(self, input_values: Sequence[float]) -> Tuple[float, ...]:
        assert len(input_values) == self._in_dim
        output_values = self._output(input_values)
        assert len(output_values) == self._out_dim
        return output_values


class MultivariatePolynomialRegression(MultivariateRegression):
    def __init__(self, input_dimensionality: int, output_dimensionality: int, degree: int, past_scope: int = -1, learning_drag: int = -1):
        super().__init__(input_dimensionality, output_dimensionality, past_scope=past_scope, learning_drag=learning_drag)
        self._regressions = tuple(
            MultiplePolynomialFromLinearRegression(input_dimensionality, degree, past_scope=past_scope, learning_drag=learning_drag)
            for _ in range(output_dimensionality)
        )
        self._degree = degree

    def derivation_output(self, input_values: Sequence[float], derive_by: int) -> Tuple[float, ...]:
        return tuple(_each_regression.derivation_output(input_values, derive_by) for _each_regression in self._regressions)

    def _fit(self, input_values: Sequence[float], output_values: Sequence[float], past_scope: int, learning_drag: int) -> Tuple[float, ...]:
        return tuple(
            _each_regression.fit(input_values, _each_output, past_scope=past_scope, learning_drag=learning_drag)
            for _each_regression, _each_output in zip(self._regressions, output_values)
        )

    def _output(self, input_values: Sequence[float]) -> Tuple[float, ...]:
        return tuple(_each_regression.output(input_values) for _each_regression in self._regressions)


def setup_2d_axes():
    fig = pyplot.figure()
    plot_axis = fig.add_subplot(211)
    plot_axis.set_xlabel("x")
    plot_axis.set_ylabel("y")

    error_axis = fig.add_subplot(212)
    error_axis.set_xlabel("t")
    error_axis.set_ylabel("error")
    return plot_axis, error_axis


def test_2d():
    dim_range = -10., 10.

    plot_axis, error_axis = setup_2d_axes()

    r = MultiplePolynomialFromLinearRegression(1, 3, -1)
    # r = MultiplePolynomialOptimizationRegression(1, 3)

    # fun = lambda _x: -cos(_x / (1. * math.pi))
    fun = lambda _x: 0. + 0. * _x ** 1. + 0. * _x ** 2. + 1. * _x ** 3. #  + 1. * _x ** 4.
    x_range = tuple(_x / 10. for _x in range(int(dim_range[0]) * 10, int(dim_range[1]) * 10))
    y_range = tuple(fun(_x) for _x in x_range)
    plot_axis.plot(x_range, y_range, color="C0")
    plot_axis.set_xlim(*dim_range)

    iterations = 0

    window_size = 100000
    error_development = deque(maxlen=window_size)

    while True:
        x = random.uniform(*dim_range)
        y_o = r.output([x])

        y_t = fun(x)
        error = 0 if iterations < 1 else smear(error_development[-1], abs(y_o - y_t), iterations)
        error_development.append(error)

        if Timer.time_passed(1000):
            print(f"{iterations:d} iterations finished")

            values = tuple(r.output([_x]) for _x in x_range)
            l, = plot_axis.plot(x_range, values, color="C1")
            values_d = tuple(r.derivation_output([_x], 0) for _x in x_range)
            l_d, = plot_axis.plot(x_range, values_d, color="C2")
            plot_axis.set_ylim((min(values + values_d), max(values + values_d)))

            x_min = max(0, iterations - window_size)
            x_max = x_min + window_size
            error_axis.set_xlim((x_min, x_max))

            x_min_limit = max(0, iterations - len(error_development))
            e, = error_axis.plot(range(x_min_limit, x_min_limit + len(error_development)), error_development, color="black")
            error_axis.set_ylim((min(error_development), max(error_development)))

            pyplot.pause(.001)
            pyplot.draw()

            l.remove()
            l_d.remove()
            e.remove()

        r.fit([x], y_t, past_scope=1000, learning_drag=0)

        iterations += 1

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


def plot_4d(axis: pyplot.Axes.axes, _fun: Callable[[float, float, float], float], dim_ranges: Tuple[Tuple[float, float], ...]):
    resolution = 10
    _X, _Y, _Z = tuple(numpy.linspace(*_range, endpoint=True, num=resolution) for _range in dim_ranges)

    _X = []
    _Y = []
    _Z = []
    _V = []

    cm = pyplot.cm.get_cmap("rainbow")

    for _x in numpy.linspace(*dim_ranges[0], num=resolution, endpoint=True):
        for _y in numpy.linspace(*dim_ranges[1], num=resolution, endpoint=True):
            for _z in numpy.linspace(*dim_ranges[2], num=resolution, endpoint=True):
                _X.append(_x)
                _Y.append(_y)
                _Z.append(_z)
                _v = _fun(_x, _y, _z)
                _V.append(_v)
    print(f"evaluation range: {min(_V):.2f}, {max(_V):.2f}")
    return axis.scatter(_X, _Y, _Z, c=_V, cmap=cm, alpha=.2)


def setup_3d_axes():
    fig = pyplot.figure()
    plot_axis = fig.add_subplot(211, projection='3d')
    plot_axis.set_aspect('equal')
    plot_axis.set_xlabel("x")
    plot_axis.set_ylabel("y")
    plot_axis.set_zlabel("z")

    error_axis = fig.add_subplot(212)
    error_axis.set_xlabel("t")
    error_axis.set_ylabel("error")
    return plot_axis, error_axis


def test_3d():
    from mpl_toolkits.mplot3d import Axes3D

    dim_range = -10., 10.

    plot_axis, error_axis = setup_3d_axes()

    r = MultiplePolynomialFromLinearRegression(2, 4, past_scope=100, learning_drag=0)

    # fun = lambda _x, _y: 10. + 1. * _x ** 1. + 1. * _y ** 1. + 4. * _x * _y + 1. * _x ** 2. + -2.6 * _y ** 2.
    fun = lambda _x, _y: -cos(_x / (1. * math.pi)) + -cos(_y / (1. * math.pi))
    #plot_surface(plot_axis, fun, (dim_range, dim_range), resize=True)
    #pyplot.pause(.001)
    #pyplot.draw()

    iterations = 0

    error_development = deque(maxlen=10000)

    while True:
        x = random.uniform(*dim_range)
        y = random.uniform(*dim_range)
        z_o = r.output([x, y])

        z_t = fun(x, y)
        error = 0 if iterations < 1 else smear(error_development[-1], abs(z_o - z_t), iterations)
        error_development.append(error)

        if Timer.time_passed(1000):
            print(f"{iterations:d} finished")

            ln = plot_surface(plot_axis, lambda _x, _y: r.output([_x, _y]), (dim_range, dim_range), resize=True)
            ln_d = plot_surface(plot_axis, lambda _x, _y: r.derivation_output([_x, _y], 0), (dim_range, dim_range), resize=False)
            e, = error_axis.plot(range(len(error_development)), error_development, color="black")
            error_axis.set_ylim((min(error_development), max(error_development)))

            pyplot.pause(.001)
            pyplot.draw()

            ln.remove()
            ln_d.remove()
            e.remove()

        r.fit([x, y], z_t)  # , past_scope=iterations)
        iterations += 1


if __name__ == "__main__":
    test_3d()
    # test_2d()

# coding=utf-8
import itertools
import random
from functools import reduce
from typing import Sequence, Tuple, List, Callable

import numpy
from matplotlib import pyplot

from tools.functionality import combinations, smear, get_min_max
from tools.timer import Timer


class MultipleRegression:
    def __init__(self, input_dimensionality: int, drag: int = -1):
        self._in_dim = input_dimensionality
        self._drag = drag

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        raise NotImplementedError()

    def fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        assert len(input_values) == self._in_dim
        assert drag >= 0 or self._drag >= 0
        _drag = max(drag, self._drag)
        self._fit(input_values, output_value, _drag)

    def _output(self, input_values: Sequence[float]) -> float:
        raise NotImplementedError()

    def output(self, input_values: Sequence[float]) -> float:
        assert len(input_values) == self._in_dim
        return self._output(input_values)


class MySingleLinearRegression:
    # https://towardsdatascience.com/implementation-linear-regression-in-python-in-5-minutes-from-scratch-f111c8cc5c99
    def __init__(self, drag: int = -1):
        self._drag = drag
        self._mean_x = 0.
        self._mean_y = 0.
        self._var_x = 0.
        self._var_y = 0.
        self._cov_xy = 0.
        self._initial = True

    def fit(self, input_value: float, output_value: float, drag: int = -1):
        assert self._drag >= 0 or drag >= 0
        _drag = max(self._drag, drag)

        dy = output_value - self._mean_y
        dx = input_value - self._mean_x
        self._var_x = smear(self._var_x, dx ** 2., _drag)
        self._cov_xy = smear(self._cov_xy, dx * dy, _drag)
        self._var_y = smear(self._var_y, dy ** 2., _drag)

        if self._initial:
            self._mean_x = input_value
            self._mean_y = output_value
            self._initial = False

        self._mean_x = smear(self._mean_x, input_value, _drag)
        self._mean_y = smear(self._mean_y, output_value, _drag)

    def output(self, input_values: float) -> float:
        a1 = 0. if self._var_x == 0. else self._cov_xy / self._var_x
        a0 = self._mean_y - a1 * self._mean_x
        return a0 + a1 * input_values


class NumpySingleLinearRegression:
    def __init__(self, drag: int = -1):
        degree = 1
        assert degree >= 1
        self._degree = degree
        self._drag = drag
        self._var_matrix = tuple([0. for _ in range(degree + 1)] for _ in range(degree + 1))
        self._cov_matrix = [0. for _ in range(degree + 1)]

    def fit(self, in_value: float, out_value: float, drag: int = -1):
        assert self._drag >= 0 or drag >= 0
        _drag = max(self._drag, drag)

        for _r, _var_row in enumerate(self._var_matrix):
            for _c in range(self._degree + 1):
                _var_row[_c] = smear(_var_row[_c], in_value ** (_r + _c), _drag)
            self._cov_matrix[_r] = smear(self._cov_matrix[_r], out_value * in_value ** _r, _drag)

    def _get_parameters(self) -> Tuple[float, ...]:
        try:
            return tuple(numpy.linalg.solve(self._var_matrix, self._cov_matrix))
        except numpy.linalg.linalg.LinAlgError:
            return tuple(0. for _ in range(self._degree + 1))

    def output(self, in_value: float) -> float:
        parameters = self._get_parameters()
        return sum(_c * in_value ** _i for _i, _c in enumerate(parameters))


class MultipleLinearRegression(MultipleRegression):
    def __init__(self, input_dimensionality: int, drag: int = -1):
        super().__init__(input_dimensionality, drag)
        # self._regressions = tuple(NumpySingleLinearRegression(drag) for _ in range(input_dimensionality))
        self._regressions = tuple(MySingleLinearRegression(drag) for _ in range(input_dimensionality))

    def _output(self, input_values: Sequence[float]) -> float:
        return sum(_regression.output(_x) for _regression, _x in zip(self._regressions, input_values))

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        _subtract = tuple(_each_regression.output(_each_input) for _each_regression, _each_input in zip(self._regressions, input_values))
        for _i, (_each_input, _each_regression) in enumerate(zip(input_values, self._regressions)):
            _each_regression.fit(_each_input, output_value - sum(_subtract[__i] for __i in range(self._in_dim) if _i != __i), drag=drag)


class MultiplePolynomialFromLinearRegression(MultipleRegression):
    def __init__(self, input_dimensionality: int, degree: int, drag: int = -1.):
        no_polynomial_parameters = sum(combinations(_i + 1, _i + input_dimensionality) for _i in range(degree))
        super().__init__(no_polynomial_parameters, drag)
        self._degree = degree
        self._regression = MultipleLinearRegression(no_polynomial_parameters, drag=drag)

    @staticmethod
    def _make_polynomial_inputs(input_values: Sequence[float], degree: int) -> Tuple[float, ...]:
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

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        poly_inputs = MultiplePolynomialFromLinearRegression._make_polynomial_inputs(input_values, self._degree)
        self._regression._fit(poly_inputs, output_value, drag=drag)

    def fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        assert drag >= 0 or self._drag >= 0
        poly_inputs = MultiplePolynomialFromLinearRegression._make_polynomial_inputs(input_values, self._degree)
        assert len(poly_inputs) == self._in_dim
        _drag = max(drag, self._drag)
        self._fit(poly_inputs, output_value, _drag)

    def output(self, input_values: Sequence[float]) -> float:
        poly_inputs = MultiplePolynomialFromLinearRegression._make_polynomial_inputs(input_values, self._degree)
        assert len(poly_inputs) == self._in_dim
        return self._output(poly_inputs)


class MultiplePolynomialHillClimbingRegression(MultipleRegression):
    def __init__(self, input_dimensionality: int, degree: int, drag: int, epsilon: float):
        super().__init__(input_dimensionality, drag)
        self._no_polynomial_parameters = sum(combinations(_i + 1, _i + input_dimensionality) for _i in range(degree))
        self._last_step = [0. for _ in range(self._no_polynomial_parameters)]
        self._parameters = [0. for _ in range(self._no_polynomial_parameters)]
        self._epsilon = epsilon
        self._degree = degree

    @staticmethod
    def _randomize(vector: List[float], mean: float, deviance: float):
        for _i in range(len(vector)):
            vector[_i] += random.gauss(mean, deviance)

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        _prediction = self._output(input_values)
        _error = _prediction - output_value
        for _i, _s in enumerate(self._last_step):
            self._last_step[_i] = smear(_s, _s - _error, drag)
            # self._last_step[_i] -= _error

        MultiplePolynomialHillClimbingRegression._randomize(self._last_step, 0., self._epsilon)

        for _i, (_p, _s) in enumerate(zip(self._parameters, self._last_step)):
            # self._parameters[_i] = smear(_p, _p + _s, drag)
            self._parameters[_i] += _s

    def _output(self, input_values: Sequence[float]) -> float:
        poly_inputs = MultiplePolynomialFromLinearRegression._make_polynomial_inputs(input_values, self._degree)
        return sum(_p * _x for _p, _x in zip(poly_inputs, self._parameters))


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

    r = MultiplePolynomialFromLinearRegression(1, 1, -1)
    # r = MultiplePolynomialHillClimbingRegression(1, 4, -1, 4.)

    fun = lambda _x: 900. + 1. * _x ** 1. # + -10. * _x ** 2. + 0. * _x ** 3. + 1. * _x ** 4.
    x_range = tuple(_x / 10. for _x in range(int(dim_range[0]) * 10, int(dim_range[1]) * 10))
    y_range = tuple(fun(_x) for _x in x_range)
    plot_axis.plot(x_range, y_range, color="C0")
    plot_axis.set_xlim(*dim_range)

    iterations = 0
    total_time = 1000000

    error_development = []

    while True:
        x = random.uniform(*dim_range)
        y_o = r.output([x])

        y_t = fun(x)
        error = 0 if iterations < 1 else smear(error_development[-1], abs(y_o - y_t), iterations)
        error_development.append(error)

        if Timer.time_passed(1000):
            print(f"{iterations * 100. / total_time:05.2f}% finished")

            l, = plot_axis.plot(x_range, tuple(r.output([_x]) for _x in x_range), color="C1")
            e, = error_axis.plot(range(len(error_development)), error_development, color="black")

            pyplot.pause(.001)
            pyplot.draw()

            l.remove()
            e.remove()

        r.fit([x], y_t, drag=iterations)

        iterations += 1

    l, = plot_axis.plot(x_range, tuple(r.output([_x]) for _x in x_range), color="C1")
    e, = error_axis.plot(range(len(error_development)), error_development, color="black")
    pyplot.show()


def plot_surface(axis: pyplot.Axes.axes, _fun: Callable[[float, float], float], dim_range: Tuple[float, float], colormap=None, resize: bool = False):
    _x = numpy.linspace(dim_range[0], dim_range[1], endpoint=True, num=int(dim_range[1] - dim_range[0]))
    _y = numpy.linspace(dim_range[0], dim_range[1], endpoint=True, num=int(dim_range[1] - dim_range[0]))
    _z = tuple(_fun(__x, __y) for __x, __y in zip(_x, _y))

    _X, _Y = numpy.meshgrid(_x, _y)
    _Z = _fun(_X, _Y)

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


def setup_3d_axes():
    from mpl_toolkits.mplot3d import Axes3D

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
    dim_range = -10., 10.

    plot_axis, error_axis = setup_3d_axes()

    r = MultiplePolynomialFromLinearRegression(2, 2, -1)

    fun = lambda _x, _y: 10. + 1. * _x ** 1. + 1. * _y ** 1. + 4. * _x * _y + 1. * _x ** 2. + -2.6 * _y ** 2.
    plot_surface(plot_axis, fun, dim_range, resize=True)
    pyplot.pause(.001)
    pyplot.draw()

    iterations = 0
    total_time = 1000000

    error_development = []

    for _ in range(total_time):
        x = random.uniform(*dim_range)
        y = random.uniform(*dim_range)
        z_o = r.output([x, y])

        z_t = fun(x, y)
        error = 0 if iterations < 1 else smear(error_development[-1], abs(z_o - z_t), iterations)
        error_development.append(error)

        if Timer.time_passed(10000):
            print(f"{iterations * 100. / total_time:05.2f}% finished")

            l = plot_surface(plot_axis, lambda _x, _y: r.output([_x, _y]), dim_range)
            e, = error_axis.plot(range(len(error_development)), error_development, color="black")

            pyplot.pause(.001)
            pyplot.draw()

            l.remove()
            e.remove()

        r.fit([x, y], z_t, drag=iterations)

        iterations += 1

    l = plot_surface(plot_axis, lambda _x, _y: r.output([_x, _y]), dim_range)
    e, = error_axis.plot(range(len(error_development)), error_development, color="black")
    pyplot.show()


if __name__ == "__main__":
    test_3d()

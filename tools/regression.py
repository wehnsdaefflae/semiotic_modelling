# coding=utf-8
import itertools
import random
from functools import reduce
from typing import Sequence, Tuple, Callable, List

from matplotlib import pyplot

from tools.functionality import combinations, smear
from tools.timer import Timer


class MultipleRegression:
    def __init__(self, input_dimensionality: int, drag: int):
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


class SingleLinearRegression:
    def __init__(self, drag: int):
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


class MultipleLinearRegression(MultipleRegression):
    def __init__(self, input_dimensionality: int, drag: int):
        super().__init__(input_dimensionality, drag)
        self._regressions = tuple(SingleLinearRegression(drag) for _ in range(input_dimensionality))

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

    def _output(self, input_values: Sequence[float]) -> float:
        return sum(_regression.output(_x) for _regression, _x in zip(self._regressions, input_values)) / self._in_dim

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        for _each_input, _each_regression in zip(input_values, self._regressions):
            _each_regression.fit(_each_input, output_value, drag=drag)


class MultiplePolynomialFromLinearRegression(MultipleLinearRegression):
    def __init__(self, input_dimensionality: int, degree: int, drag: int):
        no_polynomial_parameters = sum(combinations(_i + 1, _i + input_dimensionality) for _i in range(degree))
        super().__init__(no_polynomial_parameters, drag)
        self._degree = degree

    def output(self, input_values: Sequence[float]):
        poly_inputs = MultipleLinearRegression._make_polynomial_inputs(input_values, self._degree)
        return MultipleRegression.output(self, poly_inputs)

    def fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        poly_inputs = MultipleLinearRegression._make_polynomial_inputs(input_values, self._degree)
        MultipleRegression.fit(self, poly_inputs, output_value, drag=drag)


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
        poly_inputs = MultipleLinearRegression._make_polynomial_inputs(input_values, self._degree)
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

    # r = MultiplePolynomialFromLinearRegression(1, 4, -1)
    r = MultiplePolynomialHillClimbingRegression(1, 4, -1, 4.)

    fun = lambda _x: -11. + 4. * _x ** 1. + -7.2 * _x ** 2. + .4 * _x ** 3.
    x_range = tuple(_x / 10. for _x in range(int(dim_range[0]) * 10, int(dim_range[1]) * 10))
    y_range = tuple(fun(_x) for _x in x_range)
    plot_axis.plot(x_range, y_range, color="C0")
    plot_axis.set_xlim(*dim_range)

    iterations = 0
    total_time = 1000000

    error_development = []

    for _ in range(total_time):
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

        r.fit([x], y_t, drag=100000)
        # r.fit([x], y_t, drag=iterations)

        iterations += 1

    l, = plot_axis.plot(x_range, tuple(r.output([_x]) for _x in x_range), color="C1")
    e, = error_axis.plot(range(len(error_development)), error_development, color="black")
    pyplot.show()


def test_3d():
    # TODO: implement using function in regression_experiments.py
    pass


if __name__ == "__main__":
    test_2d()

# coding=utf-8
import itertools
import random
from functools import reduce
from typing import Sequence, Tuple, Callable

from matplotlib import pyplot

from tools.functionality import combinations, smear
from tools.timer import Timer


class Regression:
    def __init__(self, input_dimensionality: int, drag: int):
        self._in_dim = input_dimensionality
        self._drag = drag

    def fitted_function(self) -> Callable[[Sequence[float]], float]:
        raise NotImplementedError()

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        raise NotImplementedError()

    def fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        assert len(input_values) == self._in_dim
        assert drag >= 0 or self._drag >= 0
        _drag = max(drag, self._drag)
        self._fit(input_values, output_value, _drag)

    def output(self, input_values: Sequence[float]) -> float:
        assert len(input_values) == self._in_dim
        fit = self.fitted_function()
        return fit(input_values)


class LinearRegression(Regression):
    def __init__(self, drag: int):
        super().__init__(1, drag)
        self._drag = drag
        self._mean_x = 0.
        self._mean_y = 0.
        self._var_x = 0.
        self._var_y = 0.
        self._cov_xy = 0.
        self._initial = True

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        assert self._drag >= 0 or drag >= 0
        assert len(input_values) == 1
        x = input_values[0]
        _drag = max(self._drag, drag)

        dy = output_value - self._mean_y
        dx = x - self._mean_x
        self._var_x = smear(self._var_x, dx ** 2., _drag)
        self._cov_xy = smear(self._cov_xy, dx * dy, _drag)
        self._var_y = smear(self._var_y, dy ** 2., _drag)

        if self._initial:
            self._mean_x = x
            self._mean_y = output_value
            self._initial = False

        self._mean_x = smear(self._mean_x, x, _drag)
        self._mean_y = smear(self._mean_y, output_value, _drag)

    def fitted_function(self) -> Callable[[Sequence[float]], float]:
        a1 = 0. if self._var_x == 0. else self._cov_xy / self._var_x
        a0 = self._mean_y - a1 * self._mean_x
        return lambda _x: a0 + a1 * _x[0]


class MultipleLinearRegression(Regression):
    def __init__(self, input_dimensionality: int, drag: int):
        super().__init__(input_dimensionality, drag)
        self._regressions = tuple(LinearRegression(drag) for _ in range(input_dimensionality))

    def fitted_function(self) -> Callable[[Sequence[float]], float]:
        return lambda _x: sum(_regression.output([__x]) for _regression, __x in zip(self._regressions, _x)) / self._in_dim

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        for _each_input, _each_regression in zip(input_values, self._regressions):
            _each_regression.fit([_each_input], output_value, drag=drag)


class PolynomialRegression(MultipleLinearRegression):
    def __init__(self, input_dimensionality: int, degree: int, drag: int):
        no_polynomial_parameters = sum(combinations(_i + 1, _i + input_dimensionality) for _i in range(degree))
        super().__init__(no_polynomial_parameters, drag)
        self._degree = degree

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

    def _prediction(self, input_values: Sequence[float]) -> float:
        _poly_inputs = PolynomialRegression._make_polynomial_inputs(input_values, self._degree)
        poly_function = MultipleLinearRegression.fitted_function(self)
        return poly_function(_poly_inputs)

    def fitted_function(self) -> Callable[[Sequence[float]], float]:
        return self._prediction

    def _fit(self, input_values: Sequence[float], output_value: float, drag: int):
        _polynomial_inputs = PolynomialRegression._make_polynomial_inputs(input_values, self._degree)
        MultipleLinearRegression._fit(self, _polynomial_inputs, output_value, drag)

    def output(self, input_values: Sequence[float]):
        poly_inputs = PolynomialRegression._make_polynomial_inputs(input_values, self._degree)
        return Regression.output(self, poly_inputs)

    def fit(self, input_values: Sequence[float], output_value: float, drag: int = -1):
        poly_inputs = PolynomialRegression._make_polynomial_inputs(input_values, self._degree)
        Regression.fit(self, poly_inputs, output_value, drag=drag)


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

    r = PolynomialRegression(1, 4, -1)

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

        if Timer.time_passed(100):
            print(f"{iterations * 100. / total_time:05.2f}% finished")

            fitted_function = r.fitted_function()
            l, = plot_axis.plot(x_range, tuple(fitted_function([_x]) for _x in x_range), color="C1")
            e, = error_axis.plot(range(len(error_development)), error_development, color="black")

            pyplot.pause(.001)
            pyplot.draw()

            l.remove()
            e.remove()

        # r.fit([x], y_t, drag=50)
        r.fit([x], y_t, drag=iterations)

        iterations += 1

    fitted_function = r.fitted_function()
    l, = plot_axis.plot(x_range, tuple(fitted_function([_x]) for _x in x_range), color="C1")
    e, = error_axis.plot(range(len(error_development)), error_development, color="black")
    pyplot.show()


def test_3d():
    # TODO: implement using function in regression_experiments.py
    pass


if __name__ == "__main__":
    test_2d()

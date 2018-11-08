# coding=utf-8
import random
import time
from typing import Tuple, Sequence, Optional
from matplotlib import pyplot


# TODO: implement polynomial regressor for rational reinforcement learning
from mpl_toolkits.mplot3d import Axes3D

from tools.functionality import smear
import numpy


class SinglePolynomialRegressor:
    # TODO: use for multiple polynomial regressor and multiple linear regressor
    # https://arachnoid.com/sage/polynomial.html
    # https://www.quantinsti.com/blog/polynomial-regression-adding-non-linearity-to-a-linear-model
    # https://stats.stackexchange.com/a/294900/62453
    # https://stats.stackexchange.com/questions/92065/why-is-polynomial-regression-considered-a-special-case-of-multiple-linear-regres
    def __init__(self, degree: int):
        assert degree >= 1
        self._degree = degree
        self._var_matrix = tuple([0. for _ in range(degree + 1)] for _ in range(degree + 1))
        self._cov_matrix = [0. for _ in range(degree + 1)]

    def fit(self, in_value: float, out_value: float, drag: int):
        assert drag >= 0
        for _r, _var_row in enumerate(self._var_matrix):
            for _c in range(self._degree + 1):
                _var_row[_c] = smear(_var_row[_c], in_value ** (_r + _c), drag)
            self._cov_matrix[_r] = smear(self._cov_matrix[_r], out_value * in_value ** _r, drag)

    def get_parameters(self) -> Tuple[float, ...]:
        try:
            return tuple(numpy.linalg.solve(self._var_matrix, self._cov_matrix))
        except numpy.linalg.linalg.LinAlgError:
            return tuple(0. for _ in range(self._degree + 1))

    def output(self, in_value: float) -> float:
        parameters = self.get_parameters()
        return sum(_c * in_value ** _i for _i, _c in enumerate(parameters))


class MultiplePolynomialRegressor:
    def __init__(self, input_degrees: Sequence[int]):
        # input_degrees determines the degree of the respective input dimension
        self._input_dimensions = len(input_degrees)
        self._max_deg = max(input_degrees)
        self._regressors = tuple(SinglePolynomialRegressor(_degree) for _degree in input_degrees)

    def fit(self, in_values: Tuple[float, ...], out_value: float, drag: int):
        for _i, (_in_value, _regressor) in enumerate(zip(in_values, self._regressors)):
            _regressor.fit(_in_value, out_value, drag)
            if _i == 0:
                print(f"{_in_value:f}, {out_value:f}")

    def output(self, in_values: Tuple[float, ...]) -> float:
        return sum(_regressor.output(_in_value) for _in_value, _regressor in zip(in_values, self._regressors)) / self._input_dimensions

    def get_parameters(self) -> Tuple[Tuple[float, ...], ...]:
        # each row is one input
        # each col is one degree from 0 to max(input_degrees)

        parameters = []
        for _each_regressor in self._regressors:
            _parameters = _each_regressor.get_parameters()
            parameters.append(_parameters + (0.,) * (self._max_deg - len(_parameters)))

        return tuple(parameters)


class LinearRegressor:
    def __init__(self, input_dimensions: int, drag: int):
        # https://de.wikipedia.org/wiki/Multiple_lineare_Regression
        # https://mubaris.com/2017/09/28/linear-regression-from-scratch/
        self._drag = drag
        self._input_dimensions = input_dimensions
        self._mean_x = [0. for _ in range(input_dimensions)]
        self._mean_y = 0.
        self._var_x = [0. for _ in range(input_dimensions)]
        self._var_y = 0.
        self._cov_xy = [0. for _ in range(input_dimensions)]
        self._iterations = 0

    def __str__(self):
        parameters = self.get_parameters()
        component_list = ["{:.4f} * x{:d}".format(_p, _i) for _p, _i in zip(parameters[:-1], range(self._input_dimensions))]
        arguments = ", ".join(["x{:d}".format(_i) for _i in range(self._input_dimensions)])
        components = " + ".join(component_list)
        return "f({:s}) = {:s} + {:.4f}".format(arguments, components, parameters[-1])

    def sim(self, x: Tuple[float, ...], y: float, default: float = 1.) -> float:
        assert len(x) == self._input_dimensions
        if 0 >= self._iterations:
            return default

        d = (self.output(x) - y) ** 2.      # deviation
        if 0. >= d:
            return 1.

        if self._var_y == 0.:
            return 0.

        return 1. - min(1., d / self._var_y)

    def fit(self, x: Tuple[float, ...], y: float):
        assert len(x) == self._input_dimensions
        if self._drag < 0:
            return

        dy = y - self._mean_y
        for _i, (_var_x, _cov_xy) in enumerate(zip(self._var_x, self._cov_xy)):
            _dx = x[_i] - self._mean_x[_i]      # distance from mean x
            self._var_x[_i] = smear(_var_x, _dx ** 2., self._drag)
            self._cov_xy[_i] = smear(_cov_xy, _dx * dy, self._drag)

        self._var_y = smear(self._var_y, dy ** 2., self._drag)

        if 0 >= self._iterations:
            self._mean_x = list(x)
            self._mean_y = y

        for _i, (_mean_x, _x) in enumerate(zip(self._mean_x, x)):
            self._mean_x[_i] = smear(_mean_x, _x, self._drag)

        self._mean_y = smear(self._mean_y, y, self._drag)
        self._iterations = 1    # TODO: change

    def get_parameters(self) -> Tuple[float, ...]:
        xn = tuple(0. if _var_x == 0. else _cov_xy / _var_x for (_cov_xy, _var_x) in zip(self._cov_xy, self._var_x))
        x0 = self._mean_y - sum(_xn * _mean_x for (_xn, _mean_x) in zip(xn, self._mean_x))
        parameters = *xn, x0
        return parameters

    def output(self, x: Tuple[float, ...]) -> float:
        assert len(x) == self._input_dimensions
        xn = self.get_parameters()
        assert len(xn) == self._input_dimensions + 1
        return sum(_x * _xn for _x, _xn in zip(x, xn[:-1])) + xn[-1]


def plot_surface(axis: pyplot.Axes.axes, _x_coefficients: Sequence[float], _y_coefficients: Sequence[float], size: int, color: Optional[str] = None):
    _x = numpy.linspace(0, size, endpoint=True, num=size)
    _y = numpy.linspace(0, size, endpoint=True, num=size)

    _X, _Y = numpy.meshgrid(_x, _y)
    _Z = sum(_y_c * (_Y ** _i) + _x_c * (_X ** _i) for _i, (_x_c, _y_c) in enumerate(zip(_x_coefficients, _y_coefficients)))

    if color is None:
        axis.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False)
    else:
        axis.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False, color=color)


def sample(number: int, _x_coefficients: Tuple[float, ...], _y_coefficients: Tuple[float, ...], dim_range: Tuple[int, int]) -> Tuple[Tuple[float, float, float], ...]:
    _f = lambda _x, _y: sum(_x_c * (_x ** _i) + _y_c * (_y ** _i) for _i, (_x_c, _y_c) in enumerate(zip(_x_coefficients, _y_coefficients)))
    _points = []
    for _ in range(number):
        _x = random.uniform(*dim_range)
        _y = random.uniform(*dim_range)
        _p = _x, _y, _f(_x, _y)
        _points.append(_p)
    return tuple(_points)


if __name__ == "__main__":
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # x_coefficients = -10., -20., 70., #3.
    # y_coefficients = 40., 50., -10., #-2.

    x_coefficients = 0., 0.,
    y_coefficients = 0., 10.,

    number_of_points = 1000

    points = sample(number_of_points, x_coefficients, y_coefficients, (0, 10))
    plot_surface(ax, x_coefficients, y_coefficients, 10)
    ax.scatter(*zip(*points))

    r = MultiplePolynomialRegressor([len(x_coefficients) - 1, len(y_coefficients) - 1])
    for p_x, p_y, p_z in points:
        r.fit((p_x, p_y), p_z, number_of_points)

    fit_x_co, fit_y_co = r.get_parameters()

    print((x_coefficients, y_coefficients))
    print((fit_x_co, fit_y_co))
    print()
    plot_surface(ax, fit_y_co, fit_x_co, 10)

    pyplot.show()

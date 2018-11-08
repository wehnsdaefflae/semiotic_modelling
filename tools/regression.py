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

    def fit(self, x: float, y: float, drag: int):
        assert drag >= 0
        for _r, _var_row in enumerate(self._var_matrix):
            for _c in range(self._degree + 1):
                _var_row[_c] = smear(_var_row[_c], x ** (_r + _c), drag)
            self._cov_matrix[_r] = smear(self._cov_matrix[_r], y * x ** _r, drag)

    def _get_parameters(self) -> Tuple[float, ...]:
        try:
            return tuple(numpy.linalg.solve(self._var_matrix, self._cov_matrix))
        except numpy.linalg.linalg.LinAlgError:
            return tuple(0. for _ in range(self._degree + 1))

    def output(self, x: float) -> float:
        parameters = self._get_parameters()
        return sum(_c * x ** _i for _i, _c in enumerate(parameters))


class MultiplePolynomialRegressor:
    def __init__(self, input_degrees: Sequence[int]):
        # input_degrees determines the degree of the respective input dimension
        self._input_dimensions = len(input_degrees)
        self._max_deg = max(input_degrees)
        self._regressors = tuple(SinglePolynomialRegressor(_degree) for _degree in input_degrees)

    def fit(self, x: Tuple[float, ...], y: float, drag: int):
        for _x, _regressor in zip(x, self._regressors):
            _regressor.fit(_x, y, drag)

    def output(self, x: Tuple[float, ...]) -> float:
        return sum(_regressor.output(_x) for _x, _regressor in zip(x, self._regressors)) / self._input_dimensions

    def _get_parameters(self) -> Tuple[Tuple[float, ...], ...]:
        # each row is one input
        # each col is one degree from 0 to max(input_degrees)

        parameters = []
        for _each_regressor in self._regressors:
            _parameters = tuple(_p / self._input_dimensions if _i == 0 else _p for _i, _p in enumerate(_each_regressor._get_parameters()))
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
        parameters = self._get_parameters()
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

    def _get_parameters(self) -> Tuple[float, ...]:
        xn = tuple(0. if _var_x == 0. else _cov_xy / _var_x for (_cov_xy, _var_x) in zip(self._cov_xy, self._var_x))
        x0 = self._mean_y - sum(_xn * _mean_x for (_xn, _mean_x) in zip(xn, self._mean_x))
        parameters = *xn, x0
        return parameters

    def output(self, x: Tuple[float, ...]) -> float:
        assert len(x) == self._input_dimensions
        xn = self._get_parameters()
        assert len(xn) == self._input_dimensions + 1
        return sum(_x * _xn for _x, _xn in zip(x, xn[:-1])) + xn[-1]


def plot_surface(ax: pyplot.Axes.axes, a: float, b: float, c: float, size: int, color: Optional[str] = None):
    x = numpy.linspace(0, size, endpoint=False, num=size)
    y = numpy.linspace(0, size, endpoint=False, num=size)

    _X, _Y = numpy.meshgrid(x, y)
    _Z = a + b * _Y + c * _X

    if color is None:
        ax.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False)
    else:
        ax.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False, color=color)


def _plot_surface(axis: pyplot.Axes.axes, _x_coefficients: Sequence[float], _y_coefficients: Sequence[float], size: int, color: Optional[str] = None):
    x = numpy.linspace(0, size, endpoint=True, num=size)
    y = numpy.linspace(0, size, endpoint=True, num=size)

    _X, _Y = numpy.meshgrid(x, y)
    _Z = sum(_x_c * _Y ** _i + _y_c * _X ** _i for _i, (_x_c, _y_c) in enumerate(zip(_x_coefficients, _y_coefficients)))

    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")

    if color is None:
        axis.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False)
    else:
        axis.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False, color=color)


def test3d(size: int):
    from mpl_toolkits.mplot3d import Axes3D
    # https://stackoverflow.com/questions/48335279/given-general-3d-plane-equation-how-can-i-plot-this-in-python-matplotlib
    # https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
    x0 = -5.
    x1 = +2.7
    x2 = -1.7

    f = lambda _x, _y: (x2 * _x + x1 * _y) + x0

    d = 10
    p_regressor = MultiplePolynomialRegressor([1, 1])
    regressor = LinearRegressor(2, d)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_surface(ax, x0, x1, x2, size, color="black")

    X = []
    Y = []
    Z = []

    shuffled_a = list(range(size))
    shuffled_b = list(range(size))
    random.shuffle(shuffled_a)
    random.shuffle(shuffled_b)

    for each_x in shuffled_a:
        for each_y in shuffled_b:
            each_z = f(each_x, each_y) + (random.random() * 2. - 1.) * 20.
            X.append(each_x)
            Y.append(each_y)
            Z.append(each_z)

            p = each_x, each_y
            ax.scatter(each_x, each_y, each_z, antialiased=False, alpha=.5)
            regressor.fit(p, each_z)
            p_regressor.fit(p, each_z, d)

    p2, p1, p0 = regressor._get_parameters()
    plot_surface(ax, p0, p1, p2, size, color="blue")

    p_para = p_regressor._get_parameters()
    plot_surface(ax, (p_para[0][0] + p_para[1][0]) / 2., p_para[1][1], p_para[0][1], size, color="green")

    dev = 0.
    for each_x, each_y, each_z in zip(X, Y, Z):
        p = each_x, each_y
        each_o = regressor.output(p)
        ax.scatter(each_x, each_y, each_o, antialiased=False, alpha=.2, color="black", marker=None)
        dev += (each_z - each_o) ** 2.

    print(dev)
    pyplot.show()


def test2d(s: float, o: float):
        f = lambda _x: s * _x + o
        X = range(20)
        Y = [f(_x) + 4. * (random.random() - .5) for _x in X]

        fig, ax = pyplot.subplots(1, sharex="all")
        ax.scatter(X, Y, label="original")

        r = LinearRegressor(1, 10)
        for _x, _y in zip(X, Y):
            r.fit((_x,),  _y)

        (_a, ), t = r._get_parameters()
        Yd = [_a * _x + t for _x in X]
        ax.plot(X, Yd, label="fit")

        var = sum((r.output((_x,)) - _t) ** 2. for (_x, _t) in zip(X, Y))
        print("{:5.2f}".format(var))

        pyplot.legend()
        pyplot.tight_layout()
        pyplot.show()


def test_poly_regression():
    # https://arachnoid.com/sage/polynomial.html
    p = SinglePolynomialRegressor(3)
    points = (-1., -1.), (0., 3.), (1., 2.5), (2., 5.), (3., 4.), (5., 2.), (7., 5.), (9., 4.)

    for x, y in points:
        p.fit(x, y, drag=len(points))

    print(p._get_parameters())

    pyplot.scatter(*zip(*points))
    pyplot.plot([_x / 10. for _x in range(-10, 90)], [p.output(_x / 10.) for _x in range(-10, 90)])
    pyplot.show()


def test_random_poly_regression():
    p = SinglePolynomialRegressor(2)
    #pyplot.xlim([0., 10.])
    #pyplot.ylim([0., 10.])

    polynomial = lambda _x: .4 * _x ** .3 + .52 * _x ** .2 + - .17 * _x ** 1. - .9
    while True:
        x = random.random() * 10.
        # y = random.random() * 10.
        y = polynomial(x)
        p.fit(x, y, drag=10)
        pyplot.scatter([x], [y])
        line, = pyplot.plot([_x / 10. for _x in range(-10, 110)], [p.output(_x / 10.) for _x in range(-10, 110)])
        pyplot.draw()
        pyplot.pause(.01)
        time.sleep(.5)
        line.remove()

    pyplot.show()


def sample(number: int, _x_coefficients: Tuple[float, ...], _y_coefficients: Tuple[float, ...], dim_range: Tuple[int, int]) -> Tuple[Tuple[float, float, float], ...]:
    # TODO: x and y mixed up?
    _f = lambda _x, _y: sum(_x_c * _y ** _i + _y_c * _x ** _i for _i, (_x_c, _y_c) in enumerate(zip(_x_coefficients, _y_coefficients)))
    _points = []
    for _ in range(number):
        _x = random.uniform(*dim_range)
        _y = random.uniform(*dim_range)
        _z = _f(_x, _y)
        _p = _x, _y, _z
        _points.append(_p)
    return tuple(_points)


if __name__ == "__main__":
    #"""
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_coefficients = -10., -20., 70., #3.
    y_coefficients = 40., 50., -10., #-2.

    number_of_points = 1000

    _plot_surface(ax, x_coefficients, y_coefficients, 10)

    points = sample(number_of_points, x_coefficients, y_coefficients, (0, 10))
    ax.scatter(*zip(*points))

    r = MultiplePolynomialRegressor([len(x_coefficients), len(y_coefficients)])
    for _x, _y, _z in points:
        r.fit((_x, _y), _z, number_of_points)

    _fit_x_co, _fit_y_co = r._get_parameters()

    print((_fit_x_co, _fit_y_co))
    _plot_surface(ax, _fit_y_co, _fit_x_co, 10)

    pyplot.show()

    exit()
    
    """
    random.seed(8746587)

    while True:
        test3d(20)
    """
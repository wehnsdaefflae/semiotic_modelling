# coding=utf-8
import random
import time
from typing import Tuple
from matplotlib import pyplot


# TODO: implement polynomial regressor for rational reinforcement learning
from tools.functionality import smear
import numpy


class SinglePolynomialRegressor:
    # https://arachnoid.com/sage/polynomial.html
    # https://www.quantinsti.com/blog/polynomial-regression-adding-non-linearity-to-a-linear-model
    # https://stats.stackexchange.com/a/294900/62453
    # https://stats.stackexchange.com/questions/92065/why-is-polynomial-regression-considered-a-special-case-of-multiple-linear-regres
    def __init__(self, drag: int, degree: int):
        self._drag = drag
        self._degree = degree
        self._var_matrix = tuple([0. for _ in range(degree + 1)] for _ in range(degree + 1))
        self._cov_matrix = [0. for _ in range(degree + 1)]

    def fit(self, x: float, y: float, drag: int = -1):
        d = self._drag if drag < 0 else drag
        for _r, _var_row in enumerate(self._var_matrix):
            for _c in range(self._degree + 1):
                _var_row[_c] = smear(_var_row[_c], x ** (_r + _c), d)
            self._cov_matrix[_r] = smear(self._cov_matrix[_r], y * x ** _r, d)

    def _get_parameters(self):
        try:
            return tuple(numpy.linalg.solve(self._var_matrix, self._cov_matrix))
        except numpy.linalg.linalg.LinAlgError:
            return tuple(0. for _ in range(self._degree))

    def output(self, x: float):
        parameters = self._get_parameters()
        return sum(_c * x ** _i for _i, _c in enumerate(parameters))


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
        self._iterations = 1    # provisorial solution

    def output(self, x: Tuple[float, ...]) -> float:
        assert len(x) == self._input_dimensions
        xn = self._get_parameters()
        assert len(xn) == self._input_dimensions + 1
        return sum(_x * _xn for _x, _xn in zip(x, xn[:-1])) + xn[-1]

    def _get_parameters(self) -> Tuple[float, ...]:
        xn = tuple(0. if _var_x == 0. else _cov_xy / _var_x for (_cov_xy, _var_x) in zip(self._cov_xy, self._var_x))
        x0 = self._mean_y - sum(_xn * _mean_x for (_xn, _mean_x) in zip(xn, self._mean_x))
        parameters = *xn, x0
        return parameters


def plot_surface(ax: pyplot.Axes.axes, a: float, b: float, c: float, size: int):
    x = numpy.linspace(0, size, endpoint=False, num=size)
    y = numpy.linspace(0, size, endpoint=False, num=size)

    _X, _Y = numpy.meshgrid(x, y)
    _Z = a + b * _Y + c * _X

    ax.plot_surface(_X, _Y, _Z, alpha=.2, antialiased=False)


def test3d(x0: float, x1: float, x2: float, size: int = 15):
    from mpl_toolkits.mplot3d import Axes3D
    # https://stackoverflow.com/questions/48335279/given-general-3d-plane-equation-how-can-i-plot-this-in-python-matplotlib
    # https://stackoverflow.com/questions/36060933/matplotlib-plot-a-plane-and-points-in-3d-simultaneously
    f = lambda _x, _y: x2 * _x + x1 * _y + x0

    regressor = LinearRegressor(2, 10)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    X = []
    Y = []
    Z = []

    shuffled_a = list(range(size))
    shuffled_b = list(range(size))
    random.shuffle(shuffled_a)
    random.shuffle(shuffled_b)

    for each_x in shuffled_a:
        for each_y in shuffled_b:
            each_z = f(each_x, each_y) + (random.random() * 2. - 1.) * 50.
            X.append(each_x)
            Y.append(each_y)
            Z.append(each_z)

            p = each_x, each_y
            ax.scatter(each_x, each_y, each_z, antialiased=False, alpha=.8)
            regressor.fit(p, each_z)

    p2, p1, p0 = regressor._get_parameters()
    plot_surface(ax, p0, p1, p2, size)

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


def test_linear_regression():
    random.seed(8746587)

    for _ in range(100):
        a = random.random() * 20. - 10
        b = random.random() * 100. - 50
        c = random.random() * 40. - 20
        test3d(a, b, c, size=10)
        # test2d(_a, _b)


def test_poly_regression():
    # https://arachnoid.com/sage/polynomial.html
    p = SinglePolynomialRegressor(0, 3)
    points = (-1., -1.), (0., 3.), (1., 2.5), (2., 5.), (3., 4.), (5., 2.), (7., 5.), (9., 4.)

    for x, y in points:
        p.fit(x, y, drag=len(points))

    print(p._get_parameters())

    pyplot.scatter(*zip(*points))
    pyplot.plot([_x / 10. for _x in range(-10, 90)], [p.output(_x / 10.) for _x in range(-10, 90)])
    pyplot.show()


def test_random_poly_regression():
    p = SinglePolynomialRegressor(0, 2)
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


if __name__ == "__main__":
    test_random_poly_regression()

# coding=utf-8
import random
from typing import Tuple, Sequence, Optional, Callable
from matplotlib import pyplot, cm

# TODO: implement polynomial regressor for rational reinforcement learning
from mpl_toolkits.mplot3d import Axes3D

from tools.functionality import smear, get_min_max
import numpy

from tools.timer import Timer


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

        # one matrix for each input?

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
        for _in_value, _regressor in zip(in_values, self._regressors):
            _regressor.fit(_in_value, out_value, drag)

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


class FullPolynomialRegressor:
    def __init__(self, input_degrees: Sequence[int], output_dimensionality: int):
        self._out_dim = output_dimensionality
        self._regressors = tuple(MultiplePolynomialRegressor(input_degrees) for _ in range(output_dimensionality))

    def fit(self, in_values: Tuple[float, ...], out_values: Tuple[float, ...], drag: int):
        for _each_regressor, _each_output in zip(self._regressors, out_values):
            _each_regressor.fit(in_values, _each_output, drag)

    def output(self, in_values: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple(_each_regressor.output(in_values) for _each_regressor in self._regressors)


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


def plot_surface(axis: pyplot.Axes.axes, _fun: Callable[[float, float], float], dim_range: Tuple[float, float], colormap=None):
    _x = numpy.linspace(dim_range[0], dim_range[1], endpoint=True, num=int(dim_range[1] - dim_range[0]))
    _y = numpy.linspace(dim_range[0], dim_range[1], endpoint=True, num=int(dim_range[1] - dim_range[0]))
    _z = tuple(_fun(__x, __y) for __x, __y in zip(_x, _y))

    _X, _Y = numpy.meshgrid(_x, _y)
    _Z = _fun(_X, _Y)

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


def plot_line(axis: pyplot.Axes.axes, _coefficients: Sequence[float], dim_range: Tuple[float, float], color: Optional[str] = None):
    _X = numpy.linspace(dim_range[0], dim_range[1], endpoint=True, num=int(dim_range[1] - dim_range[0]))
    _Z = tuple(sum(_c * _x ** _i for _i, _c in enumerate(_coefficients)) for _x in _X)

    if color is None:
        axis.plot(_X, _Z)
    else:
        axis.plot(_X, _Z, color=color)


def poly_function(_x_coefficients: Tuple[float, ...], _y_coefficients: Tuple[float, ...]) -> Callable[[float, float], float]:
    return lambda _x, _y: sum(_x_c * (_x ** _i) + _y_c * (_y ** _i) for _i, (_x_c, _y_c) in enumerate(zip(_x_coefficients, _y_coefficients)))


def trig_function() -> Callable[[float, float], float]:
    return lambda _x, _y: numpy.sin(numpy.sqrt(_y ** 2. + _x ** 2.))


def test_2d():
    fig = pyplot.figure()
    plot_axis = fig.add_subplot(211)
    plot_axis.set_xlabel("x")
    plot_axis.set_ylabel("y")

    error_axis = fig.add_subplot(212)
    error_axis.set_xlabel("t")
    error_axis.set_ylabel("error")

    dim = -10., 10.
    # r = MyRegression(4)
    r = LinearRegressor(1, 100)

    # fun = lambda _x: ((_x*.35) ** 2. - 4.) * ((_x*.35) + 3.) * ((_x*.35) - 4.) ** 2.
    # fun = lambda _x: 1. * _x ** 4. + -4. * _x ** 2. + 3 * _x ** 1.
    fun = lambda _x: 2. * _x ** 3. + -5. * _x ** 2. + 23. * _x + -112.
    # fun = lambda _x: sin(.5 * _x / math.pi)

    X = tuple(_x / 10. for _x in range(int(dim[0] * 10.), int(dim[1] * 10.)))
    Y = tuple(fun(_x) for _x in X)

    x_max, x_min = max(X), min(X)
    x_margin = (x_max - x_min) * .1

    y_max, y_min = max(Y), min(Y)
    y_margin = (y_max - y_min) * .1

    plot_axis.set_xlim((x_min - x_margin, x_max + x_margin))
    plot_axis.set_ylim((y_min - y_margin, y_max + y_margin))
    plot_axis.plot(X, Y, color="C0")

    number_of_points = 1000000
    error_axis.set_xlim((0, number_of_points))
    drag = number_of_points

    predictions = []
    training = []

    E = []
    error = 0
    for _t in range(number_of_points):
        x = random.uniform(*dim)
        p = r.output((x,))
        predictions.append((x, p))

        y = fun(x)
        training.append((x, y))

        r.fit((x,), y)  # , drag)

        error = smear(error, abs(p - y), _t)

        E.append(error)

        #fit_fun = lambda _x: sum(_c * _x ** _i for _i, _c in enumerate(r._weights))
        #P = [fit_fun(_x) for _x in X]

        e_line, = error_axis.plot(E, color="black")
        #line, = plot_axis.plot(X, P, color="C1")
        sc = plot_axis.scatter(*zip(*predictions), alpha=.2, color="black")

        pyplot.pause(.001)
        pyplot.draw()

        if _t < number_of_points - 1:
            #line.remove()
            sc.remove()
            e_line.remove()

    pyplot.show()


def test_3d():
    fig = pyplot.figure()
    axis_3d = fig.add_subplot(221, projection='3d')
    axis_3d.set_xlabel("x")
    axis_3d.set_ylabel("y")
    axis_3d.set_zlabel("z")

    axis_2d_error = fig.add_subplot(222)
    axis_2d_error.set_xlabel("t")
    axis_2d_error.set_ylabel("error")

    axis_2d_yz = fig.add_subplot(223)
    axis_2d_yz.set_xlabel("y")
    axis_2d_yz.set_ylabel("z")

    axis_2d_xz = fig.add_subplot(224)
    axis_2d_xz.set_xlabel("x")
    axis_2d_xz.set_ylabel("z")

    # x_coefficients = 0., 1.,
    # y_coefficients = 0., -1.,

    # degree = number of coefficients - 1
    x_coefficients = -375., 400., -140., 20., -1.,
    y_coefficients = 375., -400., 140., -20., 1.,

    # fun = poly_function(x_coefficients, y_coefficients)
    fun = trig_function()

    number_of_points = 1000
    drag_value = number_of_points

    value_range = 0., 10.

    plot_surface(axis_3d, fun, value_range, colormap=cm.viridis)  # , color="C0")

    # r = MultiplePolynomialRegressor([6, 6])
    r = LinearRegressor(2, number_of_points)

    error = []
    tar_z = []
    out_z = []
    z_min, z_max = .0, .0
    for _t in range(number_of_points):
        p_x = random.uniform(*value_range)
        p_y = random.uniform(*value_range)

        input_values = p_x, p_y
        output_value = r.output(input_values)

        out_z.append((p_x, p_y, output_value))

        p_z = fun(p_x, p_y)
        if _t < 1:
            z_min = p_z
            z_max = p_z
        else:
            z_min = min(z_min, p_z)
            z_max = max(z_max, p_z)

        tar_z.append((p_x, p_y, p_z))

        e = abs(output_value - p_z)
        error_value = e if _t < 1 else smear(error[-1], e, _t)
        error.append(error_value)

        r.fit(input_values, p_z, drag_value)

        #fit_fun = poly_function(*r.get_parameters())
        #fit_surface = plot_surface(axis_3d, fit_fun, value_range, colormap=cm.brg)
        #pyplot.pause(.001)
        #pyplot.draw()
        #if _t < number_of_points - 1:
        #    fit_surface.remove()
        if Timer.time_passed(2000):
            print(f"{_t * 100. / number_of_points:05.2f}% finished.")

    margin = (z_max - z_min) * .1
    axis_3d.set_zlim([z_min - margin, z_max + margin])
    print(error[-1])
    axis_3d.scatter(*zip(*out_z), alpha=.4, color="C1")
    # axis_3d.scatter(*zip(*tar_z), color="blue")

    # axis_3d.scatter(*zip(*points))
    axis_2d_error.plot(error)
    axis_2d_yz.scatter([_p[1] for _p in tar_z], [_p[2] for _p in tar_z], color="C0")
    axis_2d_xz.scatter([_p[0] for _p in tar_z], [_p[2] for _p in tar_z], color="C0")

    fit_x_co, fit_y_co = r.get_parameters()

    # print((x_coefficients, y_coefficients))
    # print((fit_x_co, fit_y_co))
    # print()
    plot_line(axis_2d_xz, fit_x_co, value_range, color="C1")
    plot_line(axis_2d_yz, fit_y_co, value_range, color="C1")

    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    test_2d()
    # test_3d()

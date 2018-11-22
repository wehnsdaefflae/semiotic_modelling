import itertools
from functools import reduce
from typing import Sequence, Tuple


class Function:
    def __init__(self, input_dimensionality: int):
        self._in_dim = input_dimensionality

    def __str__(self):
        raise NotImplementedError()

    def derive(self, derive_by: int) -> "Function":
        raise NotImplementedError()

    def output(self, *input_values: float) -> float:
        assert len(input_values) == self._in_dim
        return self._output(*input_values)

    def _output(self, *input_values: float) -> float:
        raise NotImplementedError()


class LinearFunction(Function):
    def __init__(self,):
        super().__init__(1)
        self.c0 = 0.
        self.c1 = 0.

    def __str__(self):
        return f"f(x0) = {self.c0:.1f} + {self.c1:.1f} x0"

    def derive(self, derive_by: int) -> "Function":
        assert derive_by == 0
        fun = LinearFunction()
        fun.c0 = self.c1
        return fun

    def _output(self, *input_values: float) -> float:
        return self.c0 + self.c1 * input_values[0]


class MultipleLinearFunction(Function):
    def __init__(self, input_dimensionality: int):
        super().__init__(input_dimensionality)
        self.c0 = 0.
        self.c1 = [0. for _ in range(input_dimensionality)]

    def __str__(self):
        return f"f(x0) = {self.c0:.1f} + " + " + ".join([f"{_c:.1f} x{_i:d}" for _i, _c in enumerate(self.c1)])

    def derive(self, derive_by: int) -> "Function":
        fun = LinearFunction()
        fun.c1 = self.c1[derive_by]
        return fun

    def _output(self, *input_values: float) -> float:
        return self.c0 + sum(_c * _i for _c, _i in zip(self.c1, input_values))


class MultiplePolynomialFunction(Function):
    def __init__(self, input_dimensionality: int, degree: int):
        super().__init__(input_dimensionality)
        self._degree = degree

        self._input_indices = tuple(
            ((-1,),) if _i < 0 else tuple(itertools.combinations_with_replacement(range(self._in_dim), _i + 1))
            for _i in range(-1, degree)
        )
        self.coefficients = tuple([0. for _ in _d] for _d in self._input_indices)

    @staticmethod
    def polynomial_values(input_values: Sequence[float], degree: int) -> Tuple[Tuple[float, ...], ...]:
        """
        generates exhaustive polynomial combinations up to defined degree
        for example input values (x, y, z) and degree 2:
        (
            (x, y, z),
            (x*y, x*z, y*z, x^2, y^2, z^2),
            (x*y*z, x^2 * y, x^2 * z, y^2 * x, y^2 * z, z^2 * x, z^2 * y, x^3, y^3, z^3)
        )
        """
        assert degree >= 1
        return tuple(
            tuple(
                reduce(lambda _x, _y: _x * _y, each_combination)
                for each_combination in itertools.combinations_with_replacement(input_values, _i + 1)
            )
            for _i in range(degree)

        )

    def __str__(self):
        lst = []
        for _j, (_c, _i) in enumerate(zip(self.coefficients, self._input_indices)):
            if _j < 1:
                assert len(_c) == 1
                lst.append(f"{_c[0]:.1f}")
            else:
                for __c, __i in zip(_c, _i):
                    i = " ".join([f"x{_j:d}" for _j in __i])
                    s = f"{__c:.1f} {i:s}"
                    lst.append(s)
        polynomial_features = [f"x{_i:d}" for _i in range(self._in_dim)]
        _pf = ", ".join(polynomial_features)
        left_hand = f"f({_pf:s})"
        right_hand = " + ".join(lst)
        return left_hand + " = " + right_hand

    def derive(self, derive_by: int) -> "MultiplePolynomialFunction":
        assert self._degree >= 1
        assert derive_by < self._in_dim
        derivative = MultiplePolynomialFunction(self._in_dim, self._degree - 1)

        derived_coefficients = []
        for _i, (_coefficients, _inputs) in enumerate(zip(self.coefficients, self._input_indices)):
            if _i < 1:
                continue
            new_ = []
            for _each_input, _base_coef in zip(_inputs, _coefficients):
                degree = _each_input.count(derive_by)
                if 0 < degree:
                    new_.append(degree * _base_coef)
            if 0 < len(new_):
                derived_coefficients.append(new_)

        derivative.coefficients = derived_coefficients
        return derivative

    def _output(self, *input_values: float) -> float:
        polynomial_values = MultiplePolynomialFunction.polynomial_values(input_values, self._degree)
        assert len(polynomial_values) == self._degree
        s = 0.
        for _i, (_v, _c) in enumerate(zip(((0.,),) + polynomial_values, self.coefficients)):
            s += sum(__c * __v ** _i for __c, __v in zip(_c, _v))
        return s


if __name__ == "__main__":
    f = MultiplePolynomialFunction(1, 1)
    print(f)
    print(f.coefficients)
    f.coefficients = ((2.,), (2.,))
    print(f.output(-1.))
    print(f.output(-2.))

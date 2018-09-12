# coding=utf-8
from typing import TypeVar, Tuple, Generic

T = TypeVar("T")


class EXP_A(Tuple[T, float]):
    ...


EXP_B = Tuple[T, float]


V = TypeVar("V")


class MyClass(Generic[V]):
    def get_value_a(self, t: EXP_A[V]) -> V:
        return t[0]

    def get_value_b(self, t: EXP_B[V]) -> V:
        return t[0]


# StrClass = MyClass[int]
class StrClass(MyClass[str]):
    pass


instance = "a", .5
# instance = 3, .5

sc = StrClass()
a: str = sc.get_value_a(instance)
b: str = sc.get_value_b(instance)

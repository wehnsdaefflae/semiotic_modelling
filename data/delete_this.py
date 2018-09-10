# coding=utf-8
from typing import TypeVar, Tuple, Generic

T = TypeVar("T")


EXP = Tuple[T, float]


V = TypeVar("V")


class MyTuples(Generic[V]):
    @staticmethod
    def get_value(t: EXP[V]) -> V:
        return t[0]


class StrTuples(MyTuples[str]):
    pass


tp = "a", .5

StrTuples.get_value(tp)
# supposed to return "a"
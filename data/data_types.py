#!/usr/bin/env python3
# coding=utf-8

from typing import TypeVar, Hashable, Tuple, Generic, Union, List, DefaultDict

VALUE_TYPE = TypeVar("TYPE_A", Hashable, float)
INPUT = Tuple[VALUE_TYPE, ...]
OUTPUT = Tuple[VALUE_TYPE, ...]
EXAMPLE = Tuple[INPUT[VALUE_TYPE], OUTPUT[VALUE_TYPE]]
CURRENT_EXAMPLES = Tuple[EXAMPLE[VALUE_TYPE], ...]


class Input(Tuple[VALUE_TYPE]):
    def __init__(self, *k: List[VALUE_TYPE]):
        super().__init__(k)


def f(objects: List[object]) -> List[int]:
    return objects  # Type check error: incompatible types in assignment


if __name__ == "__main__":

    # z1 = INPUT[float]("s", "s")
    data = DefaultDict[int, bytes]()
    z2 = Tuple[float, ...](2.,)

    a: str = "a"

    #x: Tuple[str] = ("a",)
    x: INPUT[float] = ("a",)
    y: float = ("a", 5, None)
    z: Union[int, str] = 1.1

    o = [None, "None", -1.1]
    i = f(o)
    print(i)


# TODO: determine and fix general data structure


"""
NOMINAL_VALUE = Hashable
NOMINAL_INPUT = Tuple[NOMINAL_VALUE, ...]
NOMINAL_OUTPUT = Tuple[NOMINAL_VALUE, ...]
NOMINAL_EXAMPLE = Tuple[NOMINAL_INPUT, NOMINAL_OUTPUT]
CURRENT_NOMINAL_EXAMPLES = Tuple[NOMINAL_EXAMPLE, ...]

RATIONAL_VALUE = float
RATIONAL_INPUT = Tuple[RATIONAL_VALUE, ...]
RATIONAL_OUTPUT = Tuple[RATIONAL_VALUE, ...]
RATIONAL_EXAMPLE = Tuple[RATIONAL_INPUT, RATIONAL_OUTPUT]
CURRENT_RATIONAL_EXAMPLES = Tuple[RATIONAL_EXAMPLE, ...]


def rationalize(source: Iterable[CURRENT_NOMINAL_EXAMPLES]) -> Generator[CURRENT_RATIONAL_EXAMPLES, None, None]:
    in_values = dict()
    out_values = dict()

    def _convert(value: Any, c_dict: Dict[Hashable, float]) -> float:
        r_value = c_dict.get(value)
        if r_value is None:
            r_value = len(c_dict)
            c_dict[value] = r_value
        return distribute_circular(r_value)

    for input_values, target_values in source:
        rational_examples = []
        for each_input, each_target in zip(input_values, target_values):
            each_example = tuple(_convert(_x, in_values) for _x in each_input), tuple(_convert(_x, out_values) for _x in each_target)
            rational_examples.append(each_example)
        yield tuple(rational_examples)
"""
# coding=utf-8
from typing import Sequence, Union, List, Generator, TypeVar, Any, Tuple


def generate_values(file_path: str, columns: Union[Sequence[str], Sequence[int]]) -> Generator[Tuple[float, ...], None, None]:
    type_set = set(type(_x) for _x in columns)
    if 1 < len(type_set):
        raise ValueError("More than one type in column definition.")
    column_type, = type_set

    with open(file_path, mode="r") as file:
        if issubclass(column_type, int):
            column_indices = sorted(columns)                                    # type: List[int]

        elif issubclass(column_type, str):
            header_row = next(file).strip()
            column_header = header_row.split("\t")
            column_indices = sorted(column_header.index(_x) for _x in columns)

        else:
            raise TypeError("Wrong type in column definition.")

        for line in file:
            row = line.strip()
            row_values = row.split("\t")

            yield tuple(float(row_values[_i]) for _i in column_indices)


T = TypeVar("T")


def generate_window(generator: Generator[T, Any, Any], size: int) -> Generator[Tuple[T, ...], None, None]:
    window = [next(generator) for _ in range(size)]
    while True:
        yield tuple(window)
        window.append(next(generator))
        window.pop(0)

# coding=utf-8
import time
from typing import TypeVar, Generator, Tuple, Iterator, Optional, Sequence

TYPE_A = TypeVar("TYPE_A")


def merge_iterators(iterators: Sequence[Iterator[TYPE_A]]) -> Generator[Tuple[TYPE_A, ...], None, None]:
    no_iterators = len(iterators)
    while True:
        yield_value = tuple(next(_it) for _it in iterators)
        if len(yield_value) != no_iterators:
            raise StopIteration()
        yield yield_value


# TODO: add buffer!
def _next_value(source: Iterator[Tuple[TYPE_A, ...]], size: int) -> Generator[Tuple[TYPE_A, ...], Optional[int], None]:
    checked = [False for _ in range(size)]
    value = next(source)
    while True:
        index = yield value
        if all(checked):
            value = next(source)
            for _i in range(len(checked)):
                checked[_i] = False
        checked[index] = True


def _sub_iterator(index: int, callback: Generator[Tuple[TYPE_A, ...], int, None]) -> Generator[TYPE_A, None, None]:
    while True:
        value = callback.send(index)
        yield value[index]


def split_iterator(source: Iterator[Tuple[TYPE_A, ...]], size: int) -> Tuple[Generator[TYPE_A, Optional[TYPE_A], None], ...]:
    _cb = _next_value(source, size)
    _cb.send(None)

    return tuple(_sub_iterator(_i, _cb) for _i in range(size))


if __name__ == "__main__":
    def triple():
        _i = 0
        while True:
            yield tuple(range(_i, _i + 3))
            _i += 1

    g = triple()
    for i, each_value in enumerate(g):
        if i >= 5:
            break
        print(each_value)

    time.sleep(.5)
    print()

    g = triple()
    a_gen, b_gen, c_gen = split_iterator(g, 3)
    for i, (a_value, b_value, c_value) in enumerate(zip(a_gen, b_gen, c_gen)):
        if i >= 5:
            break
        print((a_value, b_value, c_value))

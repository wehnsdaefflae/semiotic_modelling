# coding=utf-8
import math
from typing import Tuple, Hashable, Sequence, Collection, Dict, TypeVar, Generic, List, Type, Any, Optional

from matplotlib import pyplot

from data_generation.data_sources.sequences.non_interactive import sequence_nominal_text, sequence_rational_crypto, examples_rational_trigonometric
from tools.load_configs import Config
from tools.timer import Timer


# TODO: make generic to handle x input dimensions and y output dimensions
def functionality_rational_linear(sequence: Sequence[Tuple[float, float]]) -> float:
    x_total = 0.
    y_total = 0.
    no_examples = 0

    for each_input, each_output in sequence:
        x_total += each_input
        y_total += each_output
        no_examples += 1

    x_mean = x_total / no_examples
    y_mean = y_total / no_examples

    cov = 0.
    var = 0.
    for each_input, each_output in sequence:
        x_diff = each_input - x_mean
        cov += x_diff * (each_output - y_mean)
        var += x_diff ** 2.

    a = 0. if var == 0. else cov / var
    t = y_mean - a * x_mean

    dist_total = 0.
    for each_input, each_output in sequence:
        predicted = each_input * a + t
        each_var = (each_output - predicted) ** 2.
        if var == 0.:
            normalized_distance = float(not each_var == 0.)
        else:
            normalized_distance = no_examples * each_var / var
        dist_total += min(1., max(0., normalized_distance))

    return 1. - dist_total / no_examples


TYPE_A = TypeVar("TYPE_A")
TYPE_B = TypeVar("TYPE_B")


class DictList(Dict[TYPE_A, List[TYPE_B]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, key: TYPE_A, value: TYPE_B):
        value_list = self.get(key)
        if value_list is None:
            value_list = [value]
            self[key] = value_list

        else:
            value_list.append(value)

    def update_lists(self, d: Dict[TYPE_A, TYPE_B]):
        for _k, _v in d.items():
            self.add(_k, _v)


def functionality_nominal(sequence: Sequence[Hashable]) -> float:
    examples = dict()
    for _t, (each_input, each_output) in enumerate(sequence):
        sub_dict = examples.get(each_input)
        if sub_dict is None:
            sub_dict = {each_output: 1}
            examples[each_input] = sub_dict
        else:
            sub_dict[each_output] = sub_dict.get(each_output, 0) + 1

        if Timer.time_passed(2000):
            print("{:05d} examples processed...".format(_t))

    best = 0
    total = 0
    for _i, (each_input, output_frequencies) in enumerate(examples.items()):
        frequencies = output_frequencies.values()
        max_frequency = max(frequencies)
        best += max_frequency
        total += sum(frequencies)

        if Timer.time_passed(2000):
            print("{:05d} examples processed...".format(total))

    return best / total


def test_nominal_functionality():
    # g = env_simple_nominal()
    config = Config("../configs/config.json")
    g = sequence_nominal_text(config["data_dir"] + "Texts/pride_prejudice.txt")
    example_sequence = []

    last_shape = next(g)
    for this_shape in g:
        each_example = last_shape, this_shape
        example_sequence.append(each_example)
        last_shape = this_shape

    print(functionality_nominal(example_sequence))


def test_debug_rational_linear_functionality():
    f = lambda _x: 2.1 * _x + 5.3
    x = range(100)
    y = [f(_x) for _x in x]
    x0 = [_i if _i < 0 else -1. for _i in range(len(x))]
    examples = list(zip(x, y))
    print(functionality_rational_linear(examples))


def test_crypto_linear_functionality():
    config = Config("../configs/config.json")
    g = sequence_rational_crypto(config["data_dir"] + "23Jun2017-23Jun2018-1m/EOSETH.csv", 60 * 60 * 24)
    examples = []
    last_value = next(g)
    for this_value in g:
        each_example = last_value, this_value - last_value
        examples.append(each_example)
        last_value = this_value
    print(functionality_rational_linear(examples))
    pyplot.plot(*zip(*examples))
    pyplot.show()


def test_trigonometry_rational_linear_functionality():
    g = examples_rational_trigonometric()
    examples = []
    for _ in range(1000):
        examples.append(next(g))
    print(functionality_rational_linear(examples))
    pyplot.plot(*zip(*examples))
    pyplot.show()


def main_test():
    # test_nominal_functionality()
    # test_trigonometry_rational_linear_functionality()
    # test_debug_rational_linear_functionality()
    test_crypto_linear_functionality()


def generic_functionality(examples, iterations: int, rational: bool = False) -> float:
    example_sequence = []
    for _i, each_example in enumerate(examples):
        if _i >= iterations:
            break
        example_sequence.append(each_example)
    if rational:
        return functionality_rational_linear(example_sequence)
    return functionality_nominal(example_sequence)


def get_min_max(sequence: Collection[float]) -> Tuple[float, float]:
    return min(sequence), max(sequence)


def combinations(drawn: int, total: int) -> int:
    return math.factorial(total) // (math.factorial(drawn) * math.factorial(total - drawn))


def smear(average: float, value: float, inertia: int) -> float:
    return (inertia * average + value) / (inertia + 1.)


def signum(number: float) -> int:
    return 1 if 0. < number else -1 if number < 0 else 0.


OBJECT_CLASS = TypeVar("OBJECT_CLASS")


class Factory(Generic[OBJECT_CLASS]):
    def __init__(self, template_class: Type[OBJECT_CLASS], template_args: Dict[str, Any]):
        self._class = template_class
        self._args = template_args

    def create(self, **additional_args: Dict[str, Any]) -> OBJECT_CLASS:
        args = dict(self._args)
        args.update(additional_args)
        return self._class(**args)


class Borg:
    _subclass_states = dict()

    def __init__(self):
        _class_state = Borg._subclass_states.get(self.__class__)
        if _class_state is None:
            Borg._subclass_states[self.__class__] = self.__dict__
        else:
            self.__dict__ = _class_state


if __name__ == "__main__":
    assert False

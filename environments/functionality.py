from typing import Tuple, Hashable, Sequence, Iterator, Any

from matplotlib import pyplot

from environments.non_interactive import sequence_nominal_text, sequence_rational_crypto, examples_rational_trigonometric
from tools.load_configs import Config
from tools.timer import Timer


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


def functionality_nominal(sequence: Sequence[Tuple[Hashable, Hashable]]) -> float:
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


if __name__ == "__main__":
    # test_nominal_functionality()
    # test_trigonometry_rational_linear_functionality()
    # test_debug_rational_linear_functionality()
    test_crypto_linear_functionality()


def prediction_functionality(environment: Iterator[Any], iterations: int, rational: bool = False) -> float:
    env_list = [next(environment) for _ in range(iterations)]
    examples = [(env_list[_i], env_list[_i + 1]) for _i in range(iterations - 1)]
    if rational:
        return functionality_rational_linear(examples)
    return functionality_nominal(examples)


def generic_functionality(examples: Iterator[Tuple[Any, Any]], iterations: int, rational: bool = False) -> float:
    example_sequence = []
    for _ in range(iterations):
        each_example = next(examples)
        example_sequence.append(each_example)
    if rational:
        return functionality_rational_linear(example_sequence)
    return functionality_nominal(example_sequence)

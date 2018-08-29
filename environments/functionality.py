from typing import Iterable, Tuple, Hashable

from tools.timer import Timer


def nominal(sequence: Iterable[Tuple[Hashable, Hashable]]) -> float:
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

    for _i, (each_input, output_frequencies) in enumerate(examples.values()):
        max_out = max(output_frequencies, key=lambda)
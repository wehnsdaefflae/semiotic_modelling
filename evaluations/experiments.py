from math import sqrt
from typing import Iterable, Iterator, Any, Sequence, Tuple

from matplotlib import pyplot

from modelling.model_types import Predictor
from tools.timer import Timer


def experiment_non_interactive(example_sequences: Iterable[Iterator[Tuple[Any, Any]]], model: Predictor[Any, Any],
                               rational: bool, iterations: int):
    time_axis = []
    total_errors = [0. for _ in example_sequences]
    errors = tuple([] for _ in example_sequences)

    for each_step in range(iterations):
        this_examples = tuple(next(each_sequence) for each_sequence in example_sequences)
        input_values, target_values = zip(*this_examples)

        output_values = model.predict(input_values)
        time_axis.append(each_step)
        if rational:
            for _i, (target, output) in enumerate(zip(target_values, output_values)):
                total_errors[_i] += sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                errors[_i].append(total_errors[_i] / (each_step + 1))
            model.fit(input_values, target_values)

        else:
            for _i, (target, output) in enumerate(zip(target_values, output_values)):
                total_errors[_i] += float(output != target)
                errors[_i].append(total_errors[_i] / (each_step + 1))
            model.fit(input_values, target_values)

        if Timer.time_passed(2000):
            print("{:05.2f}% finished: {:s}".format(100. * each_step / iterations, str(model.get_structure())))

    print(model.get_structure())
    for _i, each_success in enumerate(errors):
        pyplot.plot(time_axis, each_success, label="{:s} {:02d}/{:02d}".format(model.__class__.__name__, _i + 1, len(errors)))
    pyplot.draw()
    pyplot.pause(.001)

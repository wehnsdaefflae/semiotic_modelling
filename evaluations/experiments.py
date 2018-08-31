from math import sqrt
from typing import Iterable, Iterator, Any, Tuple

from matplotlib import pyplot

from environments.functionality import generic_functionality
from modelling.model_types import Predictor
from tools.timer import Timer


def experiment_non_interactive(examples_sequence: Iterable[Iterator[Tuple[Any, Any]]], predictors: Iterable[Predictor[Any, Any]],
                               rational: bool, iterations: int):

    for _i, each_sequence in enumerate(examples_sequence):
        if rational:
            print("not implemented yet")
        else:
            f = generic_functionality(each_sequence, iterations, rational=rational)
            print("sequence {:02d} functionality: {:05.3f}".format(_i, f))

    time_axis = []
    all_total_errors = tuple([0. for _ in examples_sequence] for _ in predictors)
    all_errors = tuple(tuple([] for _ in examples_sequence) for _ in predictors)

    for each_step in range(iterations):
        time_axis.append(each_step)
        this_examples = tuple(next(each_sequence) for each_sequence in examples_sequence)
        input_values, target_values = zip(*this_examples)

        for predictor_index, each_predictor in enumerate(predictors):
            total_errors = all_total_errors[predictor_index]
            errors = all_errors[predictor_index]
            output_values = each_predictor.predict(input_values)
            if rational:
                for example_index, (target, output) in enumerate(zip(target_values, output_values)):
                    total_errors[example_index] += sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                    errors[example_index].append(total_errors[example_index] / (each_step + 1))
                each_predictor.fit(input_values, target_values)

            else:
                for example_index, (target, output) in enumerate(zip(target_values, output_values)):
                    total_errors[example_index] += float(output != target)
                    errors[example_index].append(total_errors[example_index] / (each_step + 1))
                each_predictor.fit(input_values, target_values)

            if Timer.time_passed(2000):
                print("{:05.2f}% finished".format(100. * each_step / iterations))

    for predictor_index, each_predictor in enumerate(predictors):
        print(each_predictor.get_structure())
        errors = all_errors[predictor_index]
        for example_index, each_error in enumerate(errors):
            pyplot.plot(time_axis, each_error, label="{:02d}/{:02d} {:s}".format(example_index + 1, len(errors), each_predictor.__class__.__name__))
            pyplot.draw()
            pyplot.pause(.001)

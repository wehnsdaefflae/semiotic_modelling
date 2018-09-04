# coding=utf-8
import time
from math import sqrt
from typing import Iterable, Tuple, TypeVar

from matplotlib import pyplot

from data.data_types import CONCURRENT_EXAMPLES
from data.example_generation import EXAMPLE
from modelling.predictors.abstract_predictor import Predictor
from tools.timer import Timer


IN_TYPE = TypeVar("IN_TYPE")
OUT_TYPE = TypeVar("OUT_TYPE")


def experiment(examples: CONCURRENT_EXAMPLES[IN_TYPE, OUT_TYPE], predictors: Tuple[Predictor[IN_TYPE, OUT_TYPE], ...],
               rational: bool, iterations: int):
    time_axis = []
    errors = tuple([] for _ in predictors)

    # TODO: add time complexity plot and print functionality
    """
    
    for _i, each_sequence in enumerate(sequences):
        if rational:
            print("not implemented yet")
        else:
            f = generic_functionality(each_sequence, iterations, rational=rational)
            print("sequence {:02d} functionality: {:05.3f}".format(_i, f))
    """

    step_duration = 0.
    for each_step, current_examples in enumerate(examples):
        if each_step >= iterations:
            break

        time_axis.append(each_step)
        input_values, target_values = zip(*current_examples)

        for predictor_index, each_predictor in enumerate(predictors):
            last_time = time.time()
            output_values = each_predictor.predict(input_values)
            each_predictor.fit(input_values, target_values)
            step_duration += time.time() - last_time

            predictor_errors = errors[predictor_index]
            last_error = 1. if len(predictor_errors) < 1 else predictor_errors[-1]
            this_error = 0.
            for example_index, (target, output) in enumerate(zip(target_values, output_values)):
                if rational:
                    this_error += sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                else:
                    this_error += float(output != target)

            predictor_errors.append((last_error * each_step + this_error / len(input_values)) / (each_step + 1))

            if Timer.time_passed(2000):
                print("{:05.2f}% finished".format(100. * each_step / iterations))

    # fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
    for predictor_index, each_predictor in enumerate(predictors):
        print(each_predictor.get_structure())

        predictor_errors = errors[predictor_index]
        pyplot.plot(time_axis, predictor_errors, label="{:s} {:d}".format(each_predictor.__class__.__name__, each_predictor.no_examples))
        pyplot.draw()
        pyplot.pause(.001)

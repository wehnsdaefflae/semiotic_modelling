# coding=utf-8
import time
from math import sqrt
from typing import Iterable, Tuple, TypeVar

from matplotlib import pyplot

from data.data_types import CONCURRENT_EXAMPLES
from data.example_generation import EXAMPLE
from modelling.predictors.abstract_predictor import Predictor
from tools.timer import Timer


# TODO: define TypeVars here and replace Any

IN_TYPE = TypeVar("IN_TYPE")
OUT_TYPE = TypeVar("OUT_TYPE")


# TODO: concurrent_examples
def experiment(examples: CONCURRENT_EXAMPLES[IN_TYPE, OUT_TYPE], predictors: Tuple[Predictor[IN_TYPE, OUT_TYPE], ...],
               rational: bool, iterations: int):
    time_axis = []
    errors = None

    # TODO: add time plot and print functionality
    """
    
    for _i, each_sequence in enumerate(sequences):
        if rational:
            print("not implemented yet")
        else:
            f = generic_functionality(each_sequence, iterations, rational=rational)
            print("sequence {:02d} functionality: {:05.3f}".format(_i, f))
    """

    step_duration = 0.
    # acc_error = [0. for
    for each_step, current_examples in enumerate(examples):
        if each_step >= iterations:
            break

        if errors is None:
            errors = tuple(tuple([] for _ in current_examples) for _ in predictors)

        time_axis.append(each_step)
        input_values, target_values = zip(*current_examples)

        for predictor_index, each_predictor in enumerate(predictors):
            last_time = time.time()
            output_values = each_predictor.predict(input_values)
            each_predictor.fit(input_values, target_values)
            step_duration += time.time() - last_time

            predictor_errors = errors[predictor_index]
            for example_index, (target, output) in enumerate(zip(target_values, output_values)):
                sequence_errors = predictor_errors[example_index]
                last_error = 1. if len(sequence_errors) < 1 else sequence_errors[-1]

                if rational:
                    this_error = sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                else:
                    this_error = float(output != target)

                sequence_errors.append((last_error * each_step + this_error) / (each_step + 1))

            if Timer.time_passed(2000):
                print("{:05.2f}% finished".format(100. * each_step / iterations))

    # fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
    for predictor_index, each_predictor in enumerate(predictors):
        print(each_predictor.get_structure())

        predictor_errors = errors[predictor_index]
        for example_index, each_error in enumerate(predictor_errors):
            pyplot.plot(time_axis, each_error, label="{:02d}/{:02d} {:s}".format(example_index + 1, len(predictor_errors), each_predictor.__class__.__name__))
            pyplot.draw()
            pyplot.pause(.001)

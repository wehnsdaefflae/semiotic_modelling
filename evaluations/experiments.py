# coding=utf-8
import time
from math import sqrt
from typing import Tuple, TypeVar

from matplotlib import pyplot

from data.data_types import CONCURRENT_EXAMPLES
from modelling.predictors.abstract_predictor import Predictor
from tools.timer import Timer
from visualization.visualization import Canvas

IN_TYPE = TypeVar("IN_TYPE")
OUT_TYPE = TypeVar("OUT_TYPE")


def experiment(examples: CONCURRENT_EXAMPLES[IN_TYPE, OUT_TYPE], predictors: Tuple[Predictor[IN_TYPE, OUT_TYPE], ...],
               rational: bool, iterations: int, steps: int = 100):
    assert steps >= 1
    time_axis = []

    errors = tuple([] for _ in predictors)
    acc_errors = [0. for _ in predictors]

    durations = tuple([] for _ in predictors)
    acc_durations = [0. for _ in predictors]

    # TODO: add time complexity plot and print functionality
    """
    
    for _i, each_sequence in enumerate(sequences):
        if rational:
            print("not implemented yet")
        else:
            f = generic_functionality(each_sequence, iterations, rational=rational)
            print("sequence {:02d} functionality: {:05.3f}".format(_i, f))
    """

    for each_step, current_examples in enumerate(examples):
        if each_step >= iterations:
            break

        if (each_step + 1) % steps == 0:
            time_axis.append(each_step)

        input_values, target_values = zip(*current_examples)

        for predictor_index, each_predictor in enumerate(predictors):
            # test and training
            last_time = time.time()
            output_values = each_predictor.predict(input_values)
            each_predictor.fit(input_values, target_values)
            this_duration = (time.time() - last_time) * 1000.

            # determine error
            this_error = 0.
            for example_index, (target, output) in enumerate(zip(target_values, output_values)):
                if rational:
                    this_error += sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                else:
                    this_error += float(output != target)

            # memorize error and duration
            acc_error = acc_errors[predictor_index] + this_error / len(output_values)
            acc_errors[predictor_index] = acc_error
            acc_duration = acc_durations[predictor_index] + this_duration
            acc_durations[predictor_index] = acc_duration

            # log error
            if (each_step + 1) % steps == 0:
                predictor_durations = durations[predictor_index]
                predictor_durations.append(acc_duration / steps)

                predictor_errors = errors[predictor_index]
                if len(predictor_errors) < 1:
                    predictor_errors.append(acc_error / steps)
                else:
                    last_error = predictor_errors[-1]
                    predictor_errors.append((last_error * (each_step - steps) + acc_error) / each_step)

                acc_durations[predictor_index] = 0.
                acc_errors[predictor_index] = 0.

            if Timer.time_passed(2000):
                print("{:05.2f}% finished".format(100. * each_step / iterations))

    for predictor_index, each_predictor in enumerate(predictors):
        print(each_predictor.get_structure())

        Canvas.ax1.set_ylabel("average total error")
        Canvas.ax1.plot(time_axis,
                        errors[predictor_index],
                        label="{:s} {:d}".format(each_predictor.__class__.__name__, each_predictor.no_examples))
        Canvas.ax1.legend()

        Canvas.ax2.set_ylabel("iteration time (ms)")
        Canvas.ax2.plot(time_axis,
                        durations[predictor_index],
                        label="{:s} {:d}".format(each_predictor.__class__.__name__, each_predictor.no_examples))
        Canvas.ax2.legend()
        pyplot.draw()
        pyplot.pause(.001)

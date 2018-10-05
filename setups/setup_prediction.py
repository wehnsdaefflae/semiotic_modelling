# coding=utf-8
import time
from math import sqrt

from modelling.predictors.abstract_predictor import Predictor
from tools.timer import Timer
from visualization.visualization import Visualize


def setup(predictor: Predictor, example_generator, visualization_steps: int, iterations: int = 500000):
    print("Starting experiment with {:s} for {:d} iterations...".format(predictor.name(), iterations))

    average_error = 0.
    average_duration = 0.

    # exchange rate adaptation
    # error_list = []

    for t in range(iterations):
        # get concurrent examples
        concurrent_examples = next(example_generator)
        concurrent_inputs, concurrent_targets = zip(*concurrent_examples)

        # perform prediction and fit
        this_time = time.time()
        concurrent_outputs = predictor.predict(concurrent_inputs)
        predictor.fit(concurrent_examples)

        # update plot
        duration = time.time() - this_time
        try:
            # error = sum(abs(__o - __t) for _o, _t in zip(concurrent_outputs, concurrent_targets) for __o, __t in zip(_o, _t)) / len(concurrent_targets)
            error = sum(sqrt(sum((__o - __t) ** 2 for __o, __t in zip(_o, _t))) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)

        except TypeError:
            error = sum(float(_o != _t) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)

        # exchange rate adaptation
        # if .5 < concurrent_outputs[0][0]:
        #     error_list.append(error)

        average_error = (average_error * t + error) / (t + 1)
        average_duration = (average_duration * t + duration) / (t + 1)
        if (t + 1) % visualization_steps == 0:
            # exchange rate adaptation
            # Visualize.append("error", predictor.__class__.__name__, 1. if len(error_list) < 1 else sum(error_list) / len(error_list))
            Visualize.append("error", predictor.__class__.__name__, average_error)
            Visualize.append("duration", predictor.__class__.__name__, average_duration)

            try:
                for _e, (each_output, each_target) in enumerate(zip(concurrent_outputs, concurrent_targets)):
                    for _o, (output_value, target_value) in enumerate(zip(each_output, each_target)):
                        axis_key = f"output {_o:02d}/{_e:02d}"
                        Visualize.append(axis_key, predictor.__class__.__name__, output_value)
                        Visualize.append(axis_key, "target", target_value)
            except TypeError:
                pass

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    Visualize.finalize("error", predictor.__class__.__name__)
    Visualize.finalize("duration", predictor.__class__.__name__)
    # todo: finalize outputs?

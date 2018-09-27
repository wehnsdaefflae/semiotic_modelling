# coding=utf-8
import time

from modelling.predictors.abstract_predictor import Predictor
from tools.timer import Timer
from visualization.visualization import VisualizeSingle


def setup(predictor: Predictor, example_generator, rational: bool, iterations: int = 500000):
    print("Starting experiment with {:s} for {:d} iterations...".format(predictor.name(), iterations))

    visualization_steps = iterations // 1000
    average_error = 0.
    average_duration = 0.

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
        if rational:
            error = sum(abs(__o - __t) for _o, _t in zip(concurrent_outputs, concurrent_targets) for __o, __t in zip(_o, _t)) / len(concurrent_targets)
        else:
            error = sum(float(_o != _t) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)

        average_error = (average_error * t + error) / (t + 1)
        average_duration = (average_duration * t + duration) / (t + 1)
        if (t + 1) % visualization_steps == 0:
            VisualizeSingle.update("error", predictor.__class__.__name__, average_error)
            VisualizeSingle.update("output", predictor.__class__.__name__, 0. if not rational else concurrent_outputs[0][0])
            VisualizeSingle.update("output", "target", 0. if not rational else concurrent_targets[0][0])
            VisualizeSingle.update("duration", predictor.__class__.__name__, average_duration)

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    VisualizeSingle.plot("error", predictor.__class__.__name__)
    VisualizeSingle.plot("output", predictor.__class__.__name__)
    VisualizeSingle.plot("output", "target")
    VisualizeSingle.plot("duration", predictor.__class__.__name__)
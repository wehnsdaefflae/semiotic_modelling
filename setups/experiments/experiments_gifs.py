# coding=utf-8
import time
from math import sqrt

from data_generation.data_sources.sequences.read_gif import generate_rbg_pixels, generate_pixel_examples
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.rational.semiotic import RationalSemioticModel
from tools.load_configs import Config
from visualization.visualization import VisualizeSingle


def experiment():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__},
            "duration": {NominalSemioticModel.__name__}
        },
        "gif"
    )
    config = Config("../configs/config.json")

    size = 5
    # size = 200
    # size = 100

    pixel_generator = generate_rbg_pixels(config["data_dir"] + "gifs/tenor.gif", window_size=size)
    predictor = RationalSemioticModel(
        input_dimension=3,
        output_dimension=3,

        no_examples=3072,
        # no_examples=1,
        # no_examples=6,

        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)

    average_error = 0.
    average_duration = 0.
    example_sequence = generate_pixel_examples(pixel_generator)

    for _t in range(1000000):
        concurrent_examples = next(example_sequence)
        input_values, target_values = zip(*concurrent_examples)

        now = time.time()
        output_values = predictor.predict(input_values)
        predictor.fit(concurrent_examples)

        duration = time.time() - now
        error = sum(sqrt(sum((_o - _t) ** 2 for _o, _t in zip(each_output, each_target))) for each_output, each_target in zip(output_values, target_values)) / len(target_values)

        average_duration = (average_duration * _t + duration) / (_t + 1)
        average_error = (average_error * _t + error) / (_t + 1)

        if (_t + 1) % 10 == 0:
            print("frame {:05d}, error {:5.2f}, structure {:s}".format(_t, average_error, str(predictor.get_structure())))
            VisualizeSingle.update("duration", NominalSemioticModel.__name__, average_duration)
            VisualizeSingle.update("error", NominalSemioticModel.__name__, average_error)

    VisualizeSingle.plot("duration", NominalSemioticModel.__name__)
    VisualizeSingle.plot("error", NominalSemioticModel.__name__)

    VisualizeSingle.finish()

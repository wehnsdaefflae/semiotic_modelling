# coding=utf-8
import time
from math import sqrt

from data_generation.data_sources.sequences.read_gif import generate_rbg_pixels, generate_pixel_examples, generate_grayscale_pixels
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.rational.baseline import MovingAverage, Regression
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.setup_prediction import setup
from tools.load_configs import Config
from visualization.visualization import VisualizeSingle, Visualize


def experiment(iterations: int = 500):
    size = 120
    out_dim = 1
    no_ex = 4

    plots = {
        "error": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
        "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
    }
    outputs = {f"output {_o:02d}/{_e:02d}": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__, "target"} for _o in range(out_dim) for _e in range(no_ex)}
    plots.update(outputs)

    Visualize.init(
        "gif",
        plots,
        x_range=iterations,
        refresh_rate=40
    )
    config = Config("../configs/config.json")

    predictor = RationalSemioticModel(
        input_dimension=1,
        output_dimension=out_dim,

        no_examples=no_ex,

        alpha=100,
        sigma=.5,
        drag=100,
        trace_length=1)
    pixel_generator = generate_grayscale_pixels(generate_rbg_pixels(config["data_dir"] + "gifs/tenor.gif", window_size=size))
    example_sequence = generate_pixel_examples(pixel_generator)
    setup(predictor, example_sequence, 1, iterations=iterations)
    for _each_output in outputs:
        Visualize.finalize(_each_output, "target")

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=1,
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    pixel_generator = generate_grayscale_pixels(generate_rbg_pixels(config["data_dir"] + "gifs/tenor.gif", window_size=size))
    example_sequence = generate_pixel_examples(pixel_generator)
    setup(predictor, example_sequence, 1, iterations=iterations)
    for _each_output in outputs:
        Visualize.finalize(_each_output, "target")

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    pixel_generator = generate_grayscale_pixels(generate_rbg_pixels(config["data_dir"] + "gifs/tenor.gif", window_size=size))
    example_sequence = generate_pixel_examples(pixel_generator)
    setup(predictor, example_sequence, 1, iterations=iterations)
    for _each_output in outputs:
        Visualize.finalize(_each_output, "target")

    print("done!")
    Visualize.show()
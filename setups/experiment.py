# coding=utf-8
import time
from math import sqrt

from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.read_gif import generate_rbg_pixels, generate_pixel_examples
from data_generation.data_sources.systems.controller_nominal import SarsaController
from data_generation.data_sources.systems.environments import GridWorldLocal
from data_generation.data_sources.sequences.non_interactive import examples_rational_trigonometric, alternating_examples, sequence_rational_crypto, sequence_nominal_text
from modelling.predictors.abstract_predictor import Predictor
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.rational.baseline import Regression, MovingAverage
from modelling.predictors.rational.semiotic import RationalSemioticModel
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from tools.timer import Timer
from visualization.visualization import VisualizeSingle


def exchange_rates():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "binance/"

    start_stamp = 1501113780
    end_stamp = 1532508240
    interval_seconds = 60

    in_cryptos = "eos", "snt"  # , "qtum", "bnt"
    out_crypto = "qtum"

    # inputs = tuple(ExchangeRates(data_dir + "{:s}ETH.csv".format(_c.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp) for _c in in_cryptos)
    inputs = tuple(sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(_c.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp) for _c in in_cryptos)

    input_sequence = merge_iterators(inputs)
    targets = sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(out_crypto.upper()), interval_seconds,
                                       start_val=start_stamp, end_val=end_stamp)
    target_sequence = ((_x, ) for _x in targets)

    example_sequences = (input_sequence, target_sequence),
    return from_sequences(example_sequences)


def text_sequence():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "Texts/"

    input_sequence = sequence_nominal_text(data_dir + "emma.txt")
    target_sequence = sequence_nominal_text(data_dir + "emma.txt")
    next(target_sequence)

    example_sequence = (input_sequence, target_sequence),
    return from_sequences(example_sequence)


def sequence_prediction(predictor: Predictor, example_generator, rational: bool, iterations: int = 500000):
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


def controlled_grid_interaction(predictor: Predictor, iterations: int = 500000):
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "grid_worlds/"

    grid_world = GridWorldLocal(data_dir + "square.txt", rotational=True)
    # grid_world = GridWorldLocal(data_dir + "simple.txt", rotational=False)
    # grid_world = GridWorldGlobal(data_dir + "sutton.txt", rotational=False)

    controller = SarsaController(grid_world.get_motor_range(), alpha=.8, gamma=.1, epsilon=.1)
    # controller = RandomController(grid_world.get_motor_range())

    last_sensor = None
    last_motor = None
    sensor, reward = grid_world.react_to(None)

    visualization_steps = iterations // 1000
    average_reward = .0
    average_error = .0
    average_duration = .0
    for t in range(iterations):
        # get data
        this_time = time.time()
        concurrent_inputs = (last_sensor, last_motor),
        concurrent_outputs = predictor.predict(concurrent_inputs)
        concurrent_targets = sensor,
        concurrent_examples = (concurrent_inputs[0], concurrent_targets[0]),
        predictor.fit(concurrent_examples)
        d = time.time() - this_time

        # query controller
        perception = predictor.get_state(), last_sensor
        motor = controller.react_to(perception, reward)

        error = sum(float(_o != _t) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)

        average_reward = (average_reward * t + reward) / (t + 1)
        average_error = (average_error * t + error) / (t + 1)
        average_duration = (average_duration * t + d) / (t + 1)

        if (t + 1) % visualization_steps == 0:
            VisualizeSingle.update("reward", predictor.__class__.__name__, average_reward)
            VisualizeSingle.update("error", predictor.__class__.__name__, average_error)
            VisualizeSingle.update("duration", predictor.__class__.__name__, average_duration)

        last_sensor = sensor
        last_motor = motor

        sensor, reward = grid_world.react_to(motor)

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    VisualizeSingle.plot("reward", predictor.__class__.__name__)
    VisualizeSingle.plot("error", predictor.__class__.__name__)
    VisualizeSingle.plot("duration", predictor.__class__.__name__)


def nominal_sequence():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "output": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        }, "nominal sequence"
    )

    print("Generating semiotic model...")
    predictor = NominalSemioticModel(
        no_examples=1,
        alpha=100,
        sigma=.2,
        trace_length=1)
    sequence = text_sequence()

    sequence_prediction(predictor, sequence, False, iterations=500000)

    print("Generating Markov model...")
    predictor = NominalMarkovModel(no_examples=1)
    sequence = text_sequence()
    sequence_prediction(predictor, sequence, False, iterations=500000)

    VisualizeSingle.finish()


def artificial_nominal_sequence():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "output": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        }, "nominal sequence"
    )

    for _i in range(20):
        print("Generating semiotic model...")
        predictor = NominalSemioticModel(
            no_examples=1,
            alpha=100,
            sigma=.2,
            trace_length=1)
        sequence = ((_x,) for _x in alternating_examples())
        sequence_prediction(predictor, sequence, False, iterations=500000)

        print("Generating Markov model...")
        predictor = NominalMarkovModel(no_examples=1)
        sequence = ((_x,) for _x in alternating_examples())
        sequence_prediction(predictor, sequence, False, iterations=500000)

    VisualizeSingle.finish()


def rational_sequence():
    VisualizeSingle.initialize(
        {
            "error": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "output": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
        }, "rational sequence"
    )

    print("Generating semiotic model...")
    predictor = RationalSemioticModel(
        input_dimension=2,
        output_dimension=1,
        no_examples=1,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    sequence = exchange_rates()
    sequence_prediction(predictor, sequence, True, iterations=500000)

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=2,
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = exchange_rates()
    sequence_prediction(predictor, sequence, True, iterations=500000)

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = exchange_rates()
    sequence_prediction(predictor, sequence, True, iterations=500000)

    print("done!")
    VisualizeSingle.finish()


def trigonometry_sequence():
    VisualizeSingle.initialize(
        {
            "error": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "output": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__, "target"},
            "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
        }, "rational sequence"
    )

    print("Generating semiotic model...")
    predictor = RationalSemioticModel(
        input_dimension=1,
        output_dimension=1,
        no_examples=1,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    sequence_prediction(predictor, sequence, True, iterations=500000)

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=1,
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    sequence_prediction(predictor, sequence, True, iterations=500000)

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    sequence_prediction(predictor, sequence, True, iterations=500000)

    print("done!")
    VisualizeSingle.finish()


def nominal_interaction(repeat: int = 10):
    VisualizeSingle.initialize(
        {
            "reward": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        },
        "grid world")

    for _i in range(repeat):
        print("Run {:d} of {:d}...".format(_i * 2 + 1, repeat * 2))
        predictor = NominalSemioticModel(
            no_examples=1,
            alpha=100,
            sigma=.2,
            trace_length=1)
        controlled_grid_interaction(predictor, iterations=500000)

        print("Run {:d} of {:d}...".format(_i * 2 + 2, repeat * 2))
        predictor = NominalMarkovModel(no_examples=1)
        controlled_grid_interaction(predictor, iterations=500000)

    print("done!")
    VisualizeSingle.finish()


def gif_segmentation():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__},
            "duration": {NominalSemioticModel.__name__}
        },
        "gif"
    )
    config = Config("../configs/config.json")
    size = 5
    pixel_generator = generate_rbg_pixels(config["data_dir"] + "gifs/tenor.gif", window_size=size)
    predictor = RationalSemioticModel(
        input_dimension=3,
        output_dimension=3,
        no_examples=3072,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)

    average_error = 0.
    average_duration = 0.
    for _t, concurrent_examples in enumerate(generate_pixel_examples(pixel_generator)):
        print("frame {:05d}, error {:5.2f}, structure {:s}".format(_t, average_error, str(predictor.get_structure())))
        input_values, target_values = zip(*concurrent_examples)

        now = time.time()
        output_values = predictor.predict(input_values)
        predictor.fit(concurrent_examples)

        duration = time.time() - now
        error = sum(sqrt(sum((_o - _t) ** 2 for _o, _t in zip(each_output, each_target))) for each_output, each_target in zip(output_values, target_values)) / len(target_values)

        average_duration = (average_duration * _t + duration) / (_t + 1)
        average_error = (average_error * _t + error) / (_t + 1)

        if _t + 1 % 1000 == 0:
            VisualizeSingle.update("duration", NominalSemioticModel.__name__, average_duration)
            VisualizeSingle.update("error", NominalSemioticModel.__name__, average_error)

            VisualizeSingle.plot("duration", NominalSemioticModel.__name__)
            VisualizeSingle.plot("error", NominalSemioticModel.__name__)

    VisualizeSingle.finish()


if __name__ == "__main__":
    # nominal_sequence()
    # nominal_interaction()
    # rational_sequence()
    # trigonometry_sequence()
    # artificial_nominal_sequence()
    gif_segmentation()

    # TODO: ascending descending nominal
    # TODO: rational pole cart
    # TODO: remove deprecated stuff

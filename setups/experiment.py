# coding=utf-8
import time

from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.sequences import ExchangeRates, Text
from data_generation.data_sources.systems.abstract_classes import Controller
from data_generation.data_sources.systems.controller_nominal import SarsaController, RandomController
from data_generation.data_sources.systems.environments import GridWorldLocal, GridWorldGlobal
from modelling.predictors.abstract_predictor import Predictor
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from tools.timer import Timer
from visualization_.visualization import VisualizationPyplot, VisualizeSingle


def rational_sequence():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "binance/"

    start_stamp = 1501113780
    end_stamp = 1532508240
    interval_seconds = 60

    in_cryptos = "eos", "snt"  # , "qtum", "bnt"
    out_crypto = "qtum"

    inputs = tuple(ExchangeRates(data_dir + "{:s}ETH.csv".format(_c.upper()), interval_seconds,
                                 start_val=start_stamp, end_val=end_stamp)
                   for _c in in_cryptos)

    input_sequence = merge_iterators(inputs)
    targets = ExchangeRates(data_dir + "{:s}ETH.csv".format(out_crypto.upper()), interval_seconds,
                            start_val=start_stamp, end_val=end_stamp)
    target_sequence = ((_x, ) for _x in targets)

    example_sequences = (input_sequence, target_sequence),
    return from_sequences(example_sequences)


def text_sequence():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "Texts/"

    inputs = Text(data_dir + "mansfield_park.txt"), Text(data_dir + "pride_prejudice.txt")
    # input_sequence = merge_iterators(inputs)
    input_sequence = Text(data_dir + "emma.txt")
    target_sequence = Text(data_dir + "emma.txt")
    next(target_sequence)

    example_sequence = (input_sequence, target_sequence),
    return from_sequences(example_sequence)


def experiment_sequence(predictor: Predictor, example_generator, iterations: int = 500000):
    print("Starting experiment with {:s} for {:d} iterations...".format(predictor.name(), iterations))
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
        error = sum(float(_o != _t) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)

        average_error = (average_error * t + error) / (t + 1)
        average_duration = (average_duration * t + duration) / (t + 1)
        VisualizeSingle.update("error", predictor.__class__.__name__, average_error)
        VisualizeSingle.update("output", predictor.__class__.__name__, 0.)
        VisualizeSingle.update("duration", predictor.__class__.__name__, average_duration)

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))


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

    visualization_steps = 1000
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
            VisualizeSingle.update("output", predictor.__class__.__name__, 0.)
            VisualizeSingle.update("error", predictor.__class__.__name__, average_error)
            VisualizeSingle.update("duration", predictor.__class__.__name__, average_duration)

        last_sensor = sensor
        last_motor = motor

        sensor, reward = grid_world.react_to(motor)

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))


def nominal_sequence():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            # "output": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        }, "sequence"
    )
    """
    predictor = RationalSemioticModel(
        input_dimension=2,
        output_dimension=1,
        no_examples=1,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    sequence = rational_sequence()
    """

    print("Generating semiotic model...")
    predictor = NominalSemioticModel(
        no_examples=1,
        alpha=100,
        sigma=.2,
        trace_length=1)
    sequence = text_sequence()
    # """
    experiment_sequence(predictor, sequence, iterations=500000)
    VisualizeSingle.plot()

    print("Generating Markov model...")
    predictor = NominalMarkovModel(no_examples=1)
    sequence = text_sequence()
    experiment_sequence(predictor, sequence, iterations=500000)
    VisualizeSingle.plot()

    VisualizeSingle.finish()


def nominal_interaction(repeat: int = 10):
    VisualizeSingle.initialize(
        {
            "reward": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            # "output": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
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
        VisualizeSingle.plot()

    print("done!")
    VisualizeSingle.finish()


if __name__ == "__main__":
    # nominal_sequence()
    nominal_interaction()

# coding=utf-8
import time

from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.sequences import ExchangeRates, Text
from data_generation.data_sources.systems.abstract_classes import Controller
from data_generation.data_sources.systems.environments import GridWorldLocal
from modelling.predictors.abstract_predictor import Predictor
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from tools.timer import Timer
from visualization_.visualization import VisualizationSemiotic, Visualization, VisualizationPyplot


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


def nominal_sequence():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "Texts/"

    inputs = Text(data_dir + "mansfield_park.txt"), Text(data_dir + "pride_prejudice.txt")
    input_sequence = merge_iterators(inputs)
    target_sequence = Text(data_dir + "emma.txt")

    example_sequence = (input_sequence, target_sequence),
    return from_sequences(example_sequence)


def experiment_sequence(predictor: Predictor, example_generator, visualization: VisualizationPyplot, iterations: int = 500000):
    for t in range(iterations):
        # get concurrent examples
        concurrent_examples = next(example_generator)
        concurrent_inputs, concurrent_targets = zip(*concurrent_examples)

        # perform prediction and fit
        this_time = time.time()
        concurrent_outputs = predictor.predict(concurrent_inputs)
        predictor.fit(concurrent_examples)

        # update plot
        visualization.update_duration(t, time.time() - this_time)
        error = sum(float(_o != _t) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)
        visualization.update_error(t, error)
        visualization.update_output(t, concurrent_outputs, concurrent_targets)

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    visualization.show()


def experiment_interaction(controller: Controller, predictor: Predictor, iterations: int):
    c = Config("../../../configs/config.json")
    data_dir = c["data_dir"] + "grid_worlds/"

    grid_world = GridWorldLocal(data_dir)

    last_sensor = None
    last_motor = None
    sensor, reward = grid_world.react_to(None)
    for t in range(iterations):
        concurrent_inputs = (last_sensor, last_motor),
        concurrent_outputs = predictor.predict(concurrent_inputs)
        concurrent_targets = sensor,
        predictor.fit(concurrent_targets)

        perception = predictor.get_state()
        motor = controller.react_to(perception, reward)

        # log error
        # log reward

        sensor, reward = grid_world.react_to(motor)

        last_sensor = sensor
        last_motor = motor


def main():
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

    predictor = NominalSemioticModel(
        no_examples=1,
        alpha=100,
        sigma=.2,
        trace_length=1)
    sequence = nominal_sequence()
    # """

    visualization = VisualizationPyplot("nominal", 1000)
    experiment_sequence(predictor, sequence, visualization, iterations=500000)

    visualization.finish()


if __name__ == "__main__":
    main()

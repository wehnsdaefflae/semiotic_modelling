# coding=utf-8
from typing import Tuple

from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.sequences import ExchangeRates, Text
from data_generation.data_sources.systems.abstract_classes import Controller
from data_generation.data_sources.systems.environments import GridWorldLocal
from modelling.predictors.abstract_predictor import Predictor
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.rational.semiotic import RationalSemioticModel
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from tools.timer import Timer


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
    target_sequence = ExchangeRates(data_dir + "{:s}ETH.csv".format(out_crypto.upper()), interval_seconds,
                                    start_val=start_stamp, end_val=end_stamp)

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


def experiment_sequence(predictor: Predictor, example_generator, iterations: int = 500000):
    # plot each line between points on the fly
    for t in range(iterations):
        concurrent_examples = next(example_generator)

        concurrent_inputs, concurrent_targets = zip(*concurrent_examples)
        concurrent_outputs = predictor.predict(concurrent_inputs)

        predictor.fit(concurrent_examples)

        # compare concurrent_outputs to concurrent_targets

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    # plot it


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
    # """
    predictor = RationalSemioticModel(
        input_dimension=2,
        output_dimension=1,
        no_examples=1,
        alpha=100,
        sigma=.2,
        drag=100,
        history_length=1)
    sequence = rational_sequence()
    """

    predictor = NominalSemioticModel(
        no_examples=1,
        alpha=100,
        sigma=.2,
        history_length=1)
    sequence = nominal_sequence()
    """

    experiment_sequence(predictor, sequence)


if __name__ == "__main__":
    main()

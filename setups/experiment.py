# coding=utf-8
from typing import Tuple

from data_generation.conversion import from_sequences
from data_generation.data_processing import trail
from data_generation.data_sources.sequences.sequences import ExchangeRates
from data_generation.data_sources.systems.abstract_classes import Environment, Controller
from data_generation.data_sources.systems.controller_nominal import RandomController
from data_generation.data_sources.systems.environments import GridWorldLocal
from modelling.predictors.abstract_predictor import Predictor
from tools.load_configs import Config
from tools.split_merge import merge_iterators


def experiment_sequence(predictor: Predictor[Tuple[float, ...], float], history_length: int = 1):
    c = Config("../../../configs/config.json")
    data_dir = c["data_dir"] + "binance/"

    start_stamp = 1501113780
    end_stamp = 1532508240
    interval_seconds = 60

    cryptos = "eos", "snt", "qtum", "bnt"

    inputs = tuple(ExchangeRates(data_dir + "{:s}ETH.csv".format(_c.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp) for _c in cryptos)
    input_merged = merge_iterators(inputs)
    input_sequence = trail(input_merged, history_length)
    target_sequence = ExchangeRates(data_dir + "EOSETH.csv", interval_seconds, start_val=start_stamp, end_val=end_stamp)

    example_sequences = (input_sequence, target_sequence),
    example_generator = from_sequences(example_sequences)

    steps = (end_stamp - start_stamp) // interval_seconds
    for t in range(steps):
        concurrent_examples = next(example_generator)

        concurrent_inputs, concurrent_targets = zip(*concurrent_examples)
        concurrent_outputs = predictor.predict(concurrent_inputs)

        # compare concurrent_outputs to concurrent_targets

    # plot it


def experiment_systems(predictor: Predictor, iterations: int, history_length: int = 1):
    _experiment_interaction(RandomController(), predictor, iterations, history_length=history_length)

    # plot it


def experiment_control(controller: Controller, predictor: Predictor, iterations: int, history_length: int = 1):
    _experiment_interaction(controller, predictor, history_length=history_length)
    # plot it


def _experiment_interaction(controller: Controller, predictor: Predictor, iterations: int, history_length: int = 1):
    c = Config("../../../configs/config.json")
    data_dir = c["data_dir"] + "grid_worlds/"

    history = []
    grid_world = GridWorldLocal(data_dir)

    sensor, reward = grid_world.react_to(None)
    for t in range(iterations):
        motor = controller.react_to(sensor)

        condition = sensor, motor
        history.append(condition)

        # TODO: dont store history here but in predictor and add predictor state to each peception
        # controller sensor _is_ predictor state (maybe plus actual sensor)

        sensor, reward = grid_world.react_to(motor)



def main():
    pass


if __name__ == "__main__":
    main()

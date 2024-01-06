# coding=utf-8
from typing import Sequence

from matplotlib import pyplot

from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.non_interactive import sequence_rational_crypto
from modelling.predictors.rational.baseline import Regression, MovingAverage
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.setup_prediction import setup
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from visualization.visualization import VisualizeSingle, Visualize


def experiment(iterations: int = 500000):
    no_ex = 1
    cryptos = "qtum", "bnt", "snt", "eos"

    train_in_cryptos = "qtum",
    train_out_crypto = "qtum"

    test_in_cryptos = "qtum",
    test_out_crypto = "qtum"

    in_dim = len(train_in_cryptos)
    out_dim = 1

    start_stamp = 1501113780
    end_stamp = 1532508240
    ahead = 60

    plots = {
            "error train": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "error test": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
        }

    outputs_train = {
        f"output train {_o:02d}/{_e:02d}": {
            RationalSemioticModel.__name__,
            Regression.__name__,
            MovingAverage.__name__,
            "target train"
        } for _o in range(out_dim) for _e in range(no_ex)}

    outputs_test = {
        f"output test {_o:02d}/{_e:02d}": {
            RationalSemioticModel.__name__,
            Regression.__name__,
            MovingAverage.__name__,
            "target test"
        } for _o in range(out_dim) for _e in range(no_ex)}

    plots.update(outputs_train)
    plots.update(outputs_test)

    Visualize.init(
        "exchange rates",
        plots,
        refresh_rate=100,
        x_range=1000
    )

    print("Generating semiotic model...")
    predictor = RationalSemioticModel(
        input_dimension=in_dim,
        output_dimension=out_dim,
        no_examples=no_ex,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    training_streams = exchange_rate_sequence(start_stamp, end_stamp - ahead, ahead, train_in_cryptos, train_out_crypto)
    test_streams = exchange_rate_sequence(start_stamp + ahead, end_stamp, ahead, test_in_cryptos, test_out_crypto)
    setup(predictor, training_streams, test_streams, iterations // 1000, iterations=iterations)
    for _each_output in outputs_train:
        Visualize.finalize(_each_output, "target train")
    for _each_output in outputs_test:
        Visualize.finalize(_each_output, "target test")

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=in_dim,
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    training_streams = exchange_rate_sequence(start_stamp, end_stamp - ahead, ahead, train_in_cryptos, train_out_crypto)
    test_streams = exchange_rate_sequence(start_stamp + ahead, end_stamp, ahead, test_in_cryptos, test_out_crypto)
    setup(predictor, training_streams, test_streams, iterations // 1000, iterations=iterations)
    for _each_output in outputs_train:
        Visualize.finalize(_each_output, "target train")
    for _each_output in outputs_test:
        Visualize.finalize(_each_output, "target test")

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    training_streams = exchange_rate_sequence(start_stamp, end_stamp - ahead, ahead, train_in_cryptos, train_out_crypto)
    test_streams = exchange_rate_sequence(start_stamp + ahead, end_stamp, ahead, test_in_cryptos, test_out_crypto)
    setup(predictor, training_streams, test_streams, iterations // 1000, iterations=iterations)
    for _each_output in outputs_train:
        Visualize.finalize(_each_output, "target train")
    for _each_output in outputs_test:
        Visualize.finalize(_each_output, "target test")

    print("done!")
    Visualize.show()


def greater_or_equal_than_before(generator, percentage_change: float, steps_ahead: int):
    assert steps_ahead >= 0
    window = [next(generator) for _ in range(steps_ahead + 1)]
    while True:
        ratio = window[-1] / window[0]
        if ratio >= 1. + percentage_change:
            yield 1.
        elif ratio < 1. - percentage_change:
            yield -1.
        else:
            yield 0.
        window.append(next(generator))
        window.pop(0)


def exchange_rate_sequence(start_stamp: int, end_stamp: int, look_ahead: int, in_cryptos: Sequence[str], out_crypto: str):
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "binance/"

    interval_seconds = 60

    inputs = tuple(sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(_c.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp) for _c in in_cryptos)

    input_sequence = merge_iterators(inputs)
    targets = greater_or_equal_than_before(
        sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(out_crypto.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp),
        .02,
        look_ahead)
    target_sequence = ((_x, ) for _x in targets)

    example_sequences = (input_sequence, target_sequence),
    return from_sequences(example_sequences)

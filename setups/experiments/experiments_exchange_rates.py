# coding=utf-8
from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.non_interactive import sequence_rational_crypto
from modelling.predictors.rational.baseline import Regression, MovingAverage
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.setup_prediction import setup
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from visualization.visualization import VisualizeSingle, Visualize


def experiment(iterations: int = 500000):
    out_dim = 1
    no_ex = 1

    plots = {
            "error": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
        }

    outputs = {f"output {_o:02d}/{_e:02d}": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__, "target"} for _o in range(out_dim) for _e in range(no_ex)}

    plots.update(outputs)

    Visualize.init(
        "exchange rates",
        plots,
        refresh_rate=100,
        x_range=1000
    )

    print("Generating semiotic model...")
    predictor = RationalSemioticModel(
        input_dimension=2,
        output_dimension=out_dim,
        no_examples=no_ex,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    sequence = exchange_rate_sequence()
    setup(predictor, sequence, iterations=iterations)

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=2,
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    sequence = exchange_rate_sequence()
    setup(predictor, sequence, iterations=iterations)

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    sequence = exchange_rate_sequence()
    setup(predictor, sequence, iterations=iterations)

    print("done!")
    Visualize.show()


def greater_or_equal_than_before(generator, steps_ahead: int):
    window = [next(generator) for _ in range(steps_ahead)]
    while True:
        yield float(window[-1] >= window[0]) * 2. - 1.
        window.append(next(generator))
        window.pop(0)


def exchange_rate_sequence():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "binance/"

    start_stamp = 1501113780
    end_stamp = 1532508240
    interval_seconds = 60

    in_cryptos = "eos", "snt"  # , "qtum", "bnt"
    out_crypto = "qtum"

    inputs = tuple(sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(_c.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp) for _c in in_cryptos)

    input_sequence = merge_iterators(inputs)
    targets = greater_or_equal_than_before(
        sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(out_crypto.upper()), interval_seconds, start_val=start_stamp, end_val=end_stamp),
        60)
    target_sequence = ((_x, ) for _x in targets)

    example_sequences = (input_sequence, target_sequence),
    return from_sequences(example_sequences)

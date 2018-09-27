# coding=utf-8
from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.non_interactive import sequence_rational_crypto
from modelling.predictors.rational.baseline import Regression, MovingAverage
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.setup_prediction import setup
from tools.load_configs import Config
from tools.split_merge import merge_iterators
from visualization.visualization import VisualizeSingle


def experiment():
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
    sequence = exchange_rate_sequence()
    setup(predictor, sequence, True, iterations=500000)

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=2,
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = exchange_rate_sequence()
    setup(predictor, sequence, True, iterations=500000)

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = exchange_rate_sequence()
    setup(predictor, sequence, True, iterations=500000)

    print("done!")
    VisualizeSingle.finish()


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
    targets = sequence_rational_crypto(data_dir + "{:s}ETH.csv".format(out_crypto.upper()), interval_seconds,
                                       start_val=start_stamp, end_val=end_stamp)
    target_sequence = ((_x, ) for _x in targets)

    example_sequences = (input_sequence, target_sequence),
    return from_sequences(example_sequences)
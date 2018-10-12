# coding=utf-8

# from setups.experiments.experiments_gifs import experiment
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.experiments.experiments_exchange_rates import experiment, exchange_rate_sequence
# from setups.experiments.experiments_interactive import experiment
# from setups.experiments.experiments_gifs import experiment
from setups.experiments.experiments_toy import experiment_nominal, experiment_rational
from setups.setup_prediction import SetupPrediction
from tools.timer import Timer


def new_setup():
    iterations = 500000

    no_ex = 1
    cryptos = "qtum", "bnt", "snt", "eos"

    train_in_cryptos = cryptos[:1]
    train_out_crypto = cryptos[0]

    test_in_cryptos = cryptos[:1]
    test_out_crypto = cryptos[0]

    in_dim = len(train_in_cryptos)
    out_dim = 1

    start_stamp = 1501113780
    end_stamp = 1532508240
    behind = 60

    predictor = RationalSemioticModel(
        input_dimension=in_dim,
        output_dimension=out_dim,
        no_examples=no_ex,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    training_streams = exchange_rate_sequence(start_stamp, end_stamp - behind, behind, train_in_cryptos, train_out_crypto)
    test_streams = exchange_rate_sequence(start_stamp + behind, end_stamp, behind, test_in_cryptos, test_out_crypto)

    setup = SetupPrediction("test", predictor, training_streams, test_streams, logging_steps=iterations // 1000)

    for _i in range(iterations):
        data = next(setup)
        if Timer.time_passed(2000):
            print(f"finished {(_i + 1) * 100 / iterations:5.2f}%...\n{str(data):s}\n")


if __name__ == "__main__":
    # experiment_nominal()
    # experiment_rational()
    # experiment()
    # nominal_interaction()
    # experiment()

    new_setup()

    # TODO: ascending descending nominal
    # TODO: rational pole cart
    # TODO: remove deprecated stuff

# coding=utf-8
from _framework.setup import Setup
from _framework.streams.linear.rational.implementations.exchange_rate_stream import ExchangeRateStream
from _framework.systems.predictors.rational.implementations.rational_average_predictor import RationalAverage
from _framework.systems.predictors.rational.implementations.rational_linear_regression_predictor import RationalLinearRegression
from _framework.systems.predictors.rational.implementations.rational_semiotic_predictor import RationalSemiotic
from tools.load_configs import Config


if __name__ == "__main__":
    config = Config("../../configs/config.json")
    path_dir = config["data_dir"] + "binance/"

    history_length = 1
    input_file_paths = path_dir + "EOSETH.csv", path_dir + "SNTETH.csv", path_dir + "QTUMETH.csv", path_dir + "BNTETH.csv"
    target_file_paths = path_dir + "BNBETH.csv",

    input_dimensions = history_length * len(input_file_paths)
    output_dimensions = len(target_file_paths)

    exchange_parameters = {
        "input_file_paths":     input_file_paths,
        "target_file_paths":    target_file_paths,
        "start_time":           "2017-08-09T09:00:00+00:00",
        "end_time":             "2018-07-25T08:30:00+00:00",
        "interval_seconds":     60,
        "offset_seconds":       60 * 60 * 1,
        "history_length":       history_length,
    }

    experiments = (
        {
            "predictor_def": (
                RationalAverage,
                {
                    "no_states": 1,
                    "input_dimensions": input_dimensions,
                    "output_dimensions": output_dimensions,
                    "drag": 100,
                }
            ),
            "streams_def": (
                ExchangeRateStream,
                exchange_parameters,
                exchange_parameters
            ),
        },
        {
            "predictor_def": (
                RationalLinearRegression,
                {
                    "no_states": 1,
                    "input_dimensions": input_dimensions,
                    "output_dimensions": output_dimensions,
                    "history_length": history_length,
                    "drag": 100,
                }
            ),
            "streams_def": (
                ExchangeRateStream,
                exchange_parameters,
                exchange_parameters
            ),
        },
        {
            "predictor_def": (
                RationalSemiotic,
                {
                    "no_states": 1,
                    "input_dimensions": input_dimensions,
                    "output_dimensions": output_dimensions,
                    "concrete_history_length": history_length,
                    "drag": 100,
                    "alpha": 100,
                    "sigma": .2,
                }
            ),
            "streams_def": (
                ExchangeRateStream,
                exchange_parameters,
                exchange_parameters
            ),
        },
    )

    setup = Setup(experiments, 1,  -50000, visualization_interval_secs=1., storage_interval_its=1000)
    setup.run_experiment()

# coding=utf-8
from _framework.setup import Setup
from _framework.streams.interactive.interaction_stream import InteractionStream
from _framework.streams.linear.nominal.text_stream import TextStream
from _framework.streams.linear.rational.trigonometric_stream import TrigonometricStream
from _framework.systems.controllers.nominal.implementations.nominal_random_controller import NominalRandomController
from _framework.systems.controllers.nominal.implementations.nominal_sarsa_controller import NominalSarsaController
from _framework.systems.predictors.nominal.implementations.nominal_markov_predictor import NominalMarkov
from _framework.systems.predictors.nominal.implementations.nominal_semiotic_predictor import NominalSemiotic
from _framework.systems.predictors.rational.implementations.rational_average_predictor import RationalAverage
from _framework.systems.predictors.rational.implementations.rational_linear_regression_predictor import RationalLinearRegression
from _framework.systems.predictors.rational.implementations.rational_semiotic_predictor import RationalSemiotic
from _framework.systems.tasks.nominal.implementations.grid_world.implementations.transitional_grid_world import TransitionalGridWorld
from tools.load_configs import Config


if __name__ == "__main__":
    config = Config("../../configs/config.json")
    file_dir = config["data_dir"]

    history_length = 1

    experiments = (
        {
            "predictor_def": (
                RationalAverage,
                {
                    "no_states": 1,
                    "input_dimensions": 1 * history_length,
                    "output_dimensions": 1,
                    "drag": 100,
                }
            ),
            "streams_def": (
                TrigonometricStream,
                {
                    "history_length": history_length,
                },
                {
                    "history_length": history_length,
                }
            ),
        },
        {
            "predictor_def": (
                RationalLinearRegression,
                {
                    "no_states": 1,
                    "input_dimensions": 1 * history_length,
                    "output_dimensions": 1,
                    "history_length": 1,
                    "drag": 100,
                }
            ),
            "streams_def": (
                TrigonometricStream,
                {
                    "history_length": history_length,
                },
                {
                    "history_length": history_length,
                }
            ),
        },
        {
            "predictor_def": (
                RationalSemiotic,
                {
                    "no_states": 1,
                    "input_dimensions": 1 * history_length,
                    "output_dimensions": 1,
                    "concrete_history_length": history_length,
                    "drag": 100,
                    "alpha": 100,
                    "sigma": .2,
                }
            ),
            "streams_def": (
                TrigonometricStream,
                {
                    "history_length": history_length,
                },
                {
                    "history_length": history_length,
                }
            ),
        },
    )

    setup = Setup(experiments, 1,  500000, visualization_interval_secs=1., storage_interval_its=1000)
    setup.run_experiment()

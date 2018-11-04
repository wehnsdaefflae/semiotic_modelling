# coding=utf-8
from _framework.setup import Setup
from _framework.streams.interactive.interaction_stream import InteractionStream
from _framework.systems.controllers.nominal.implementations.nominal_random_controller import NominalRandomController
from _framework.systems.controllers.nominal.implementations.nominal_sarsa_controller import NominalSarsaController
from _framework.systems.predictors.nominal.implementations.nominal_markov_predictor import NominalMarkov
from _framework.systems.predictors.nominal.implementations.nominal_semiotic_predictor import NominalSemiotic
from _framework.systems.tasks.nominal.implementations.grid_world.implementations.transitional_grid_world import TransitionalGridWorld
from tools.load_configs import Config


if __name__ == "__main__":
    config = Config("../../configs/config.json")
    file_path = config["data_dir"] + "grid_worlds/square.txt"

    experiments = (
        {
            "predictor_def": (
                NominalMarkov,
                {
                    "no_states": 1
                }
            ),
            "streams_def": (
                InteractionStream,
                {
                    "task_def": (
                        TransitionalGridWorld,
                        {
                            "local": True,
                            "file_path": file_path,
                        }
                    ),
                    "history_length": 1
                },
                {
                    "task_def": (
                        TransitionalGridWorld,
                        {
                            "local": True,
                            "file_path": file_path,
                        }
                    ),
                    "history_length": 1
                }
            ),
            "controller_def": (
                NominalSarsaController,
                {
                    "alpha": .1,
                    "gamma": .5,
                    "epsilon": .1
                }
            )
        }, {
            "predictor_def": (
                NominalSemiotic,
                {
                    "no_states": 1,
                    "alpha": 100,
                    "sigma": .2,
                    "drag": 1,
                }
            ),
            "streams_def": (
                InteractionStream,
                {
                    "task_def": (
                        TransitionalGridWorld,
                        {
                            "local": True,
                            "file_path": file_path,
                        }
                    ),
                    "history_length": 1
                },
                {
                    "task_def": (
                        TransitionalGridWorld,
                        {
                            "local": True,
                            "file_path": file_path,
                        }
                    ),
                    "history_length": 1
                }
            ),
            "controller_def": (
                NominalSarsaController,
                {
                    "alpha": .1,
                    "gamma": .5,
                    "epsilon": .1
                }
            )
        },
    )

    # todo: fix positive iterations
    setup = Setup(experiments, 10, -50000, visualization_interval_secs=1., storage_interval_its=1000)
    # setup = Setup(experiments, 10,  100000, visualization_interval_secs=1., storage_interval_its=1000)
    # setup = Setup(experiments, 10, 0, visualization_interval_secs=1., storage_interval_its=1000)
    setup.run_experiment()

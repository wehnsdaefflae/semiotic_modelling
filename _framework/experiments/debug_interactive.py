# coding=utf-8
from _framework.setup import ExperimentFactory, Setup
from _framework.streams.interactive.implementations import InteractionStream
from _framework.systems.controllers.nominal.implementations import NominalSarsaController
from _framework.systems.predictors.nominal.implementations import NominalMarkov
from _framework.systems.tasks.nominal.implementations import TransitionalGridWorld
from tools.load_configs import Config

if __name__ == "__main__":
    config = Config("../../configs/config.json")
    file_path = config["data_dir"] + "grid_worlds/square.txt"

    experiment_factories = (
        ExperimentFactory(
            (
                NominalMarkov,
                {"no_states": 1}
            ), (
                InteractionStream,
                {
                    "task_def": (
                        TransitionalGridWorld,
                        {
                            "local": False,
                            "file_path": file_path,
                        }
                    ),
                    "history_length": 0
                }, {
                    "task_def": (
                        TransitionalGridWorld,
                        {
                            "local": False,
                            "file_path": file_path,
                        }
                    ),
                    "history_length": 0
                }
            ), controller_def=(
                NominalSarsaController,
                {
                    "alpha": .1,
                    "gamma": .5,
                    "epsilon": .1
                }
            )
        ),
    )

    setup = Setup(experiment_factories, 10, 0, step_size=500)
    setup.run_experiment()

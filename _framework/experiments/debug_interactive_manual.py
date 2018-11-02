# coding=utf-8
from _framework.setup import Setup
from _framework.streams.interactive.implementations import InteractionStream
from _framework.systems.controllers.nominal.implementations import NominalManualController
from _framework.systems.predictors.nominal.implementations import NominalMarkov
from _framework.systems.tasks.nominal.implementations import TransitionalGridWorld
from tools.load_configs import Config

if __name__ == "__main__":
    config = Config("../../configs/config.json")
    file_path = config["data_dir"] + "grid_worlds/simple.txt"

    experiments = (
        {
            "predictor_def": (
                NominalMarkov, {
                    "no_states": 1
                }
            ),
            "streams_def": (
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
            ),
            "controller_def": (
                NominalManualController,
                dict()
            )
        },
    )

    setup = Setup(experiments, 1, 0, visualization_interval_secs=1., storage_interval_its=0)
    setup.run_experiment()

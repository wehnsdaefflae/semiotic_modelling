# coding=utf-8
from _framework.setup import ExperimentFactory, Setup
from _framework.streams.interactive.implementations import InteractionStream
from _framework.systems.controllers.nominal.implementations import NominalRandomController
from _framework.systems.predictors.nominal.implementations import NominalLastPredictor
from _framework.systems.tasks.nominal.implementations import NominalGridWorld

if __name__ == "__main__":
    experiment_factories = (
        ExperimentFactory(
            (
                NominalLastPredictor,
                dict()
            ), (
                InteractionStream,
                {
                    "task_def": (
                        NominalGridWorld,
                        dict()
                    ),
                    "history_length": 1
                }, {
                    "task_def": (
                        NominalGridWorld,
                        dict()
                    ),
                    "history_length": 1
                }
            ), controller_def=(
                NominalRandomController,
                dict()
            )
        ),
    )

    setup = Setup(experiment_factories, 2, 1000, step_size=100)
    setup.run_experiment()

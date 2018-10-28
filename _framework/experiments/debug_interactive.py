# coding=utf-8
from _framework.setup import ExperimentFactory, Setup
from _framework.streams.interactive.implementations import InteractionStream
from _framework.systems.controllers.nominal.implementations import NominalSarsaController
from _framework.systems.predictors.nominal.implementations import NominalMarkov
from _framework.systems.tasks.nominal.implementations import RotationalGridWorld

if __name__ == "__main__":
    experiment_factories = (
        ExperimentFactory(
            (
                NominalMarkov,
                {"no_states": 1}
            ), (
                InteractionStream,
                {
                    "task_def": (
                        RotationalGridWorld,
                        {"local": False}
                    ),
                    "history_length": 1
                }, {
                    "task_def": (
                        RotationalGridWorld,
                        {"local": False}
                    ),
                    "history_length": 1
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

    setup = Setup(experiment_factories, 10, 0, step_size=5000)
    setup.run_experiment()

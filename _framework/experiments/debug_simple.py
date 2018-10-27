# coding=utf-8
from _framework.setup import ExperimentFactory, Setup
from _framework.streams.linear.nominal.implementations import NominalAscendingDescending
from _framework.systems.predictors.nominal.implementations import NominalMarkov

if __name__ == "__main__":
    experiment_factories = (
        ExperimentFactory(
            (
                NominalMarkov,
                {"no_states": 1}
            ), (
                NominalAscendingDescending,
                {"history_length": 1},
                {"history_length": 1}
            )
        ),
        ExperimentFactory(
            (
                NominalMarkov,
                {"no_states": 1}
            ), (
                NominalAscendingDescending,
                {"history_length": 2},
                {"history_length": 2}
            )
        ),
    )

    setup = Setup(experiment_factories, 10, 0, step_size=1000, visualization=True)
    setup.run_experiment()

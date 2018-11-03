# coding=utf-8
from _framework.setup import Setup
from _framework.streams.linear.nominal.alternating_stream import NominalAscendingDescending
from _framework.systems.predictors.nominal.implementations import NominalMarkov

if __name__ == "__main__":
    experiments = (
        {
            "predictor_def": (
                NominalMarkov,
                {
                    "no_states": 1
                }
            ),
            "streams_def": (
                NominalAscendingDescending,
                {
                    "history_length": 1
                }, {
                    "history_length": 1
                }
            )
        }, {
            "predictor_def": (
                NominalMarkov,
                {
                    "no_states": 1
                }
            ),
            "streams_def": (
                NominalAscendingDescending,
                {
                    "history_length": 2
                }, {
                    "history_length": 2
                }
            )
        },
    )

    setup = Setup(experiments, 1, 0, visualization_interval_secs=1., storage_interval_its=0)
    setup.run_experiment()

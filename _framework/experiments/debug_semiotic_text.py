# coding=utf-8
from _framework.setup import Setup
from _framework.streams.linear.nominal.implementations.text_stream import TextStream
from _framework.systems.predictors.nominal.implementations.nominal_markov_predictor import NominalMarkov
from _framework.systems.predictors.nominal.implementations.nominal_semiotic_predictor import NominalSemiotic
from tools.load_configs import Config


if __name__ == "__main__":
    config = Config("../../configs/config.json")
    file_dir = config["data_dir"]

    experiments = (
        {
            "predictor_def": (
                NominalMarkov,
                {
                    "no_states": 1
                }
            ),
            "streams_def": (
                TextStream,
                {
                    "file_path": file_dir + "Texts/emma.txt",
                    "history_length": 1
                },
                {
                    "file_path": file_dir + "Texts/pride_prejudice.txt",
                    "history_length": 1
                }
            ),
        },
        {
            "predictor_def": (
                NominalSemiotic,
                {
                    "no_states": 1,
                    "alpha": 100,
                    "sigma": .2
                }
            ),
            "streams_def": (
                TextStream,
                {
                    "file_path": file_dir + "Texts/emma.txt",
                    "history_length": 1
                },
                {
                    "file_path": file_dir + "Texts/pride_prejudice.txt",
                    "history_length": 1
                }
            ),
        },
    )

    setup = Setup(experiments, 1,  500000, visualization_interval_secs=1., storage_interval_its=1000)
    setup.run_experiment()

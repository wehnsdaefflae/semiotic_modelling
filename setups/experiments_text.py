from data_generation.conversion import from_sequences
from data_generation.data_sources.sequences.non_interactive import sequence_nominal_text
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from setups.setup_prediction import setup
from tools.load_configs import Config
from visualization.visualization import VisualizeSingle


def experiment():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "output": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        }, "nominal sequence"
    )

    print("Generating semiotic model...")
    predictor = NominalSemioticModel(
        no_examples=1,
        alpha=100,
        sigma=.2,
        trace_length=1)
    sequence = text_sequence()

    setup(predictor, sequence, False, iterations=500000)

    print("Generating Markov model...")
    predictor = NominalMarkovModel(no_examples=1)
    sequence = text_sequence()
    setup(predictor, sequence, False, iterations=500000)

    VisualizeSingle.finish()


def text_sequence():
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "Texts/"

    input_sequence = sequence_nominal_text(data_dir + "emma.txt")
    target_sequence = sequence_nominal_text(data_dir + "emma.txt")
    next(target_sequence)

    example_sequence = (input_sequence, target_sequence),
    return from_sequences(example_sequence)
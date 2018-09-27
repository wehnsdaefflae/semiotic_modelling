from data_generation.data_sources.sequences.non_interactive import examples_rational_trigonometric, alternating_examples
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.rational.baseline import Regression, MovingAverage
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.setup_prediction import setup
from visualization.visualization import VisualizeSingle


def experiment_rational():
    VisualizeSingle.initialize(
        {
            "error": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "output": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__, "target"},
            "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
        }, "rational sequence"
    )

    print("Generating semiotic model...")
    predictor = RationalSemioticModel(
        input_dimension=1,
        output_dimension=1,
        no_examples=1,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    setup(predictor, sequence, True, iterations=500000)

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=1,
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    setup(predictor, sequence, True, iterations=500000)

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=1,
        drag=100,
        no_examples=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    setup(predictor, sequence, True, iterations=500000)

    print("done!")
    VisualizeSingle.finish()


def experiment_nominal():
    VisualizeSingle.initialize(
        {
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "output": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        }, "nominal sequence"
    )

    for _i in range(20):
        print("Generating semiotic model...")
        predictor = NominalSemioticModel(
            no_examples=1,
            alpha=100,
            sigma=.2,
            trace_length=1)
        sequence = ((_x,) for _x in alternating_examples())
        setup(predictor, sequence, False, iterations=500000)

        print("Generating Markov model...")
        predictor = NominalMarkovModel(no_examples=1)
        sequence = ((_x,) for _x in alternating_examples())
        setup(predictor, sequence, False, iterations=500000)

    VisualizeSingle.finish()
from data_generation.data_sources.sequences.non_interactive import examples_rational_trigonometric, alternating_examples
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.rational.baseline import Regression, MovingAverage
from modelling.predictors.rational.semiotic import RationalSemioticModel
from setups.setup_prediction import setup
from visualization.visualization import Visualize


def experiment_rational(iterations: int = 50000):
    out_dim = 1
    no_ex = 1

    plots = {
            "error": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__},
            "duration": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__}
        }

    outputs = {f"output {_o:02d}/{_e:02d}": {RationalSemioticModel.__name__, Regression.__name__, MovingAverage.__name__, "target"} for _o in range(out_dim) for _e in range(no_ex)}

    plots.update(outputs)

    Visualize.init(
        "trigonometric",
        plots,
        refresh_rate=100,
        x_range=1000
    )

    print("Generating semiotic model...")
    predictor = RationalSemioticModel(
        input_dimension=1,
        output_dimension=out_dim,
        no_examples=no_ex,
        alpha=100,
        sigma=.2,
        drag=100,
        trace_length=1)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    setup(predictor, sequence, iterations // 1000, iterations=iterations)

    print("Generating regression model...")
    predictor = Regression(
        input_dimension=1,
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    setup(predictor, sequence, iterations // 1000, iterations=iterations)

    print("Generating average model...")
    predictor = MovingAverage(
        output_dimension=out_dim,
        drag=100,
        no_examples=no_ex)
    sequence = ((_x, ) for _x in examples_rational_trigonometric())
    setup(predictor, sequence, iterations // 1000, iterations=iterations)

    print("done!")
    Visualize.show()


def experiment_nominal(iterations: int = 25000):
    Visualize.init(
        "nominal sequence",
        {
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        },
        refresh_rate=100,
        x_range=1000
    )

    for _i in range(20):
        print("Generating semiotic model...")
        predictor = NominalSemioticModel(
            no_examples=1,
            alpha=100,
            sigma=1.,
            trace_length=1)
        sequence = ((_x,) for _x in alternating_examples())
        setup(predictor, sequence, iterations // 1000, iterations=iterations)

        print("Generating Markov model...")
        predictor = NominalMarkovModel(no_examples=1)
        sequence = ((_x,) for _x in alternating_examples())
        setup(predictor, sequence, iterations // 1000, iterations=iterations)

    Visualize.show()

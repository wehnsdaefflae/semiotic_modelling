from typing import Hashable, Iterator, Iterable

from matplotlib import pyplot

from environments.functionality import nominal_prediction_functionality
from environments.non_interactive import env_text, env_ascending_descending_nominal
from modelling.model_types import NominalSemioticModel, NominalMarkovModel, Predictor
from tools.load_configs import Config
from tools.timer import Timer


def _multiple_nominal(sequences: Iterable[Iterator[Hashable]], model: Predictor[Hashable, Hashable], iterations: int, history_length: int = 1):
    time_axis = []
    total_success = [0 for _ in sequences]
    successes = [[] for _ in sequences]

    last_shapes = [[] for _ in sequences]
    for each_step in range(iterations):
        this_shape = [next(sequence) for sequence in sequences]
        if all(len(each_last) >= history_length for each_last in last_shapes):
            input_values = tuple(tuple(each_shape) for each_shape in last_shapes)
            target_values = tuple(this_shape)

            output_values = model.predict(input_values)

            time_axis.append(each_step)
            for _i, (target, output) in enumerate(zip(target_values, output_values)):
                total_success[_i] += float(target == output)
                successes[_i].append(total_success[_i] / (each_step + 1))

            model.fit(input_values, target_values)

        if Timer.time_passed(2000):
            print("{:05.2f}% finished: {:s}".format(100. * each_step / iterations, str(model.get_structure())))

        for each_last, each_this in zip(last_shapes, this_shape):
            each_last.append(each_this)
            while history_length < len(each_last):
                each_last.pop(0)

    print(model.get_structure())
    for _i, each_success in enumerate(successes):
        pyplot.plot(time_axis, each_success, label="{:s} {:02d}/{:02d}".format(model.__class__.__name__, _i, len(successes)))
    pyplot.draw()
    pyplot.pause(.001)


def single_natural_sequence_comparison(iterations: int):
    c = Config("configs/config.json")

    f = nominal_prediction_functionality(env_text(c["data_dir"] + "Texts/pride_prejudice.txt"), iterations)
    print("example sequence functionality: {:05.3f}".format(f))

    environments = [env_text(c["data_dir"] + "Texts/pride_prejudice.txt")]
    predictor = NominalMarkovModel(no_examples=len(environments))
    _multiple_nominal(environments, predictor, iterations=iterations, history_length=1)

    environments = [env_text(c["data_dir"] + "Texts/pride_prejudice.txt")]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)
    _multiple_nominal(environments, predictor, iterations=iterations, history_length=1)


def single_artificial_sequence_comparison(iterations: int):
    f = nominal_prediction_functionality(env_ascending_descending_nominal(), iterations)
    print("example sequence functionality: {:05.3f}".format(f))

    environments = [env_ascending_descending_nominal()]
    predictor = NominalMarkovModel(no_examples=len(environments))
    _multiple_nominal(environments, predictor, iterations=iterations)

    environments = [env_ascending_descending_nominal()]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)
    _multiple_nominal(environments, predictor, iterations=iterations)


def multiple_natural_sequences_transfer(iterations: int):
    c = Config("configs/config.json")
    environment_a = env_text(c["data_dir"] + "Texts/pride_prejudice.txt")
    environment_b = env_text(c["data_dir"] + "Texts/mansfield_park.txt")
    environments = [environment_a, environment_b]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)

    _multiple_nominal(environments, predictor, iterations)


if __name__ == "__main__":
    #multiple_natural_sequences_transfer(500000)
    #single_natural_sequence_comparison(500000)

    # multiple_natural_sequences_transfer(600000)
    single_natural_sequence_comparison(600000)
    # single_artificial_sequence_comparison(50000)

    pyplot.legend()
    pyplot.show()

from typing import Iterable, Iterator, Any, Sequence, Tuple

from matplotlib import pyplot

from modelling.model_types import Predictor
from tools.timer import Timer


def one_step_prediction(sequences: Iterable[Iterator[Any]], model: Predictor[Any, Any], rational: bool, iterations: int, history_length: int = 1):
    time_axis = []
    total_errors = [1. for _ in sequences]
    errors = [[] for _ in sequences]

    last_shapes = [[] for _ in sequences]
    for each_step in range(iterations):
        this_shape = [next(sequence) for sequence in sequences]
        if all(len(each_last) == history_length for each_last in last_shapes):
            input_values = tuple(tuple(each_shape) for each_shape in last_shapes)
            target_values = tuple(this_shape)

            output_values = model.predict(input_values)
            time_axis.append(each_step)
            for _i, (target, output) in enumerate(zip(target_values, output_values)):
                if rational:
                    total_errors[_i] += output[0] - target
                else:
                    total_errors[_i] += float(output != target)
                errors[_i].append(total_errors[_i] / (each_step + 1))

            model.fit(input_values, tuple((_x,) for _x in target_values))

        if Timer.time_passed(2000):
            print("{:05.2f}% finished: {:s}".format(100. * each_step / iterations, str(model.get_structure())))

        for each_last, each_this in zip(last_shapes, this_shape):
            each_last.append(each_this)
            while history_length < len(each_last):
                each_last.pop(0)

    print(model.get_structure())
    for _i, each_success in enumerate(errors):
        pyplot.plot(time_axis, each_success, label="{:s} {:02d}/{:02d}".format(model.__class__.__name__, _i + 1, len(errors)))
    pyplot.draw()
    pyplot.pause(.001)


def example_prediction(example_sequences: Iterable[Iterator[Tuple[Any, Any]]], model: Predictor[Any, Any],
                       rational: bool, iterations: int, history_length: int = 1):
    time_axis = []
    total_errors = [1. for _ in example_sequences]
    errors = tuple([] for _ in example_sequences)

    input_histories = tuple([] for _ in example_sequences)
    for each_step in range(iterations):
        this_examples = tuple(next(each_sequence) for each_sequence in example_sequences)
        this_inputs, target_values = zip(*this_examples)

        for each_history, each_input in zip(input_histories, this_inputs):
            each_history.append(each_input)
            while history_length < len(each_history):
                each_history.pop(0)

        input_values = tuple(tuple(each_input) for each_input in input_histories)
        if all(len(each_input) == history_length for each_input in input_values):
            output_values = model.predict(input_values)
            time_axis.append(each_step)
            for _i, (target, output) in enumerate(zip(target_values, output_values)):
                if rational:
                    total_errors[_i] += output[0] - target
                else:
                    total_errors[_i] += float(output[0] != target)
                errors[_i].append(total_errors[_i] / (each_step + 1))

            model.fit(input_values, tuple((_x,) for _x in target_values))

            if Timer.time_passed(2000):
                print("{:05.2f}% finished: {:s}".format(100. * each_step / iterations, str(model.get_structure())))

    print(model.get_structure())
    for _i, each_success in enumerate(errors):
        pyplot.plot(time_axis, each_success, label="{:s} {:02d}/{:02d}".format(model.__class__.__name__, _i + 1, len(errors)))
    pyplot.draw()
    pyplot.pause(.001)

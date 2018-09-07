# coding=utf-8
import time
from math import sqrt
from typing import Tuple, TypeVar, Generator, Optional

from matplotlib import pyplot

from data.data_types import CONCURRENT_EXAMPLES
from modelling.predictors.abstract_predictor import Predictor
from tools.timer import Timer
from visualization.visualization import Canvas

IN_TYPE = TypeVar("IN_TYPE")
OUT_TYPE = TypeVar("OUT_TYPE")


def prediction(examples: CONCURRENT_EXAMPLES[IN_TYPE, OUT_TYPE], predictors: Tuple[Predictor[IN_TYPE, OUT_TYPE], ...],
               rational: bool, iterations: int, steps: int = 100):
    # TODO: how to combine with example_generation?!
    assert steps >= 1
    time_axis = []

    errors = tuple([] for _ in predictors)
    acc_errors = [0. for _ in predictors]

    durations = tuple([] for _ in predictors)
    acc_durations = [0. for _ in predictors]

    # TODO: print functionality
    """
    
    for _i, each_sequence in enumerate(sequences):
        if rational:
            print("not implemented yet")
        else:
            f = generic_functionality(each_sequence, iterations, rational=rational)
            print("sequence {:02d} functionality: {:05.3f}".format(_i, f))
    """

    for each_step, current_examples in enumerate(examples):
        if each_step >= iterations:
            break

        if (each_step + 1) % steps == 0:
            time_axis.append(each_step)

        input_values, target_values = zip(*current_examples)

        for predictor_index, each_predictor in enumerate(predictors):
            # test and training
            last_time = time.time()
            output_values = each_predictor.predict(input_values)
            each_predictor.fit(input_values, target_values)
            this_duration = (time.time() - last_time) * 1000.

            # determine error
            this_error = 0.
            for example_index, (target, output) in enumerate(zip(target_values, output_values)):
                if rational:
                    this_error += sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                else:
                    this_error += float(output != target)

            # memorize error and duration
            acc_error = acc_errors[predictor_index] + this_error / len(output_values)
            acc_errors[predictor_index] = acc_error
            acc_duration = acc_durations[predictor_index] + this_duration
            acc_durations[predictor_index] = acc_duration

            # log error
            if (each_step + 1) % steps == 0:
                predictor_durations = durations[predictor_index]
                predictor_durations.append(acc_duration / steps)

                predictor_errors = errors[predictor_index]
                if len(predictor_errors) < 1:
                    predictor_errors.append(acc_error / steps)
                else:
                    last_error = predictor_errors[-1]
                    predictor_errors.append((last_error * (each_step - steps) + acc_error) / each_step)

                acc_durations[predictor_index] = 0.
                acc_errors[predictor_index] = 0.

            if Timer.time_passed(2000):
                print("{:05.2f}% finished".format(100. * each_step / iterations))

    for predictor_index, each_predictor in enumerate(predictors):
        print(each_predictor.get_structure())

        Canvas.ax1.set_ylabel("average total error")
        Canvas.ax1.plot(time_axis,
                        errors[predictor_index],
                        label="{:s} {:d}".format(each_predictor.__class__.__name__, each_predictor.no_examples))
        Canvas.ax1.legend()

        Canvas.ax2.set_ylabel("iteration time (ms)")
        Canvas.ax2.plot(time_axis,
                        durations[predictor_index],
                        label="{:s} {:d}".format(each_predictor.__class__.__name__, each_predictor.no_examples))
        Canvas.ax2.legend()
        pyplot.draw()
        pyplot.pause(.001)


SENSOR = TypeVar("SENSOR")
MOTOR = TypeVar("MOTOR")

CONDITION = Tuple[Tuple[SENSOR, MOTOR], ...]


def interaction(environment: Generator[Tuple[SENSOR, float], Optional[MOTOR], None],
                controller: Generator[MOTOR, Optional[Tuple[SENSOR, float]], None],
                predictor: Predictor[CONDITION, SENSOR],
                rational: bool, iterations: int, history_length: int = 1, steps: int = 100):
    assert steps >= 1
    time_axis = []

    rewards = []
    acc_rewards = 0.

    errors = []
    acc_errors = 0.

    durations = []
    acc_durations = 0.

    history = []

    sensor, reward = environment.send(None)     # type: SENSOR, float
    motor = controller.send(None)               # type: MOTOR

    for each_step in range(iterations):
        if (each_step + 1) % steps == 0:
            time_axis.append(each_step)

        if len(history) == history_length:
            input_value = tuple(history),
            target_value = sensor,

            # test and training
            last_time = time.time()
            output_value = predictor.predict(input_value)
            predictor.fit(input_value, target_value)
            this_duration = (time.time() - last_time) * 1000.

            # TODO: check position in iteration, log reward
            feedback = sensor + predictor.get_state(), reward
            motor = controller.send(feedback)
            sensor, reward = environment.send(motor)

            # determine reward
            acc_rewards += reward
            # determine error
            this_error = 0.
            for example_index, (target, output) in enumerate(zip(target_value, output_value)):
                if rational:
                    this_error += sqrt(sum((each_output - each_target) ** 2. for each_output, each_target in zip(output, target)))
                else:
                    this_error += float(output != target)

            # memorize error and duration
            acc_errors += this_error / len(output_value)
            acc_durations += this_duration

            # log error
            if (each_step + 1) % steps == 0:
                rewards.append(acc_rewards / steps)
                durations.append(acc_durations / steps)

                if len(errors) < 1:
                    errors.append(acc_errors / steps)
                else:
                    last_error = errors[-1]
                    errors.append((last_error * (each_step - steps) + acc_errors) / each_step)

                acc_durations = 0.
                acc_errors = 0.

        condition = sensor, motor
        history.append(condition)
        while history_length < len(history):
            history.pop(0)

        if Timer.time_passed(2000):
            print("{:05.2f}% finished".format(100. * each_step / iterations))

    print(predictor.get_structure())

    Canvas.ax1.set_ylabel("error")
    Canvas.ax1.plot(time_axis, errors, label="error {:s} {:d}".format(predictor.__class__.__name__, predictor.no_examples))
    Canvas.ax1.legend()
    ax11 = Canvas.ax1.twinx()
    ax11.set_ylabel("reward")
    ax11.plot(time_axis, rewards, color="orange", label="reward {:s} {:d}".format(predictor.__class__.__name__, predictor.no_examples))
    ax11.legend()

    Canvas.ax2.set_ylabel("iteration time (ms)")
    Canvas.ax2.plot(time_axis, durations, label="{:s} {:d}".format(predictor.__class__.__name__, predictor.no_examples))
    Canvas.ax2.legend()
    pyplot.draw()
    pyplot.pause(.001)

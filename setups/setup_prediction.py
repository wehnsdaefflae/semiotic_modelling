# coding=utf-8
import datetime
import time
from math import sqrt
from typing import Tuple, Generator, Any, Union, Hashable, Iterator, TypeVar

from modelling.predictors.abstract_predictor import Predictor
from tools.logger import DataLogger
from tools.timer import Timer
from visualization.visualization import Visualize


VALUES = Union[Tuple[float, ...], Hashable]
EXAMPLE = Tuple[VALUES, VALUES]
CONCURRENT_EXAMPLES = Tuple[EXAMPLE, ...]


TYPE_A = TypeVar("TYPE_A")
TYPE_B = TypeVar("TYPE_B")


class SetupPrediction(Iterator[Tuple[Any, ...]]):
    def __init__(self,
                 name: str,
                 predictor: Predictor,
                 stream_train: Generator[CONCURRENT_EXAMPLES, None, None],
                 stream_test: Generator[CONCURRENT_EXAMPLES, None, None],
                 logging_steps: int = 1):
        self.name = name
        self.predictor = predictor
        self.stream_train = stream_train
        self.stream_test = stream_test

        self.average_train_error = []
        self.average_test_error = []
        self.average_duration = 0.

        self.iterations = 1

        _time = datetime.datetime.now()
        _time_str = _time.strftime("%Y-%m-%d_%H-%M-%S")
        self.stats_file_path = _time_str + "_" + name + ".log"
        self.logging_steps = logging_steps
        self.header = []

    def __log_data(self, header, duration, errors_train, errors_test):
        if len(self.header) < 1:
            self.header.append("iteration")
            self.header.append("duration")
            for _i in range(len(errors_train)):
                self.header.append(f"error_train_{_i:05d}")
            self.header.append("error_train_all")
            for _i in range(len(errors_test)):
                self.header.append(f"error_test_{_i:05d}")
            self.header.append("error_test_all")

        if self.logging_steps >= self.iterations:
            self.average_duration = duration
            self.average_train_error.extend(errors_train)
            self.average_test_error.extend(errors_test)

        else:
            self.average_duration = (self.average_duration * self.iterations + duration) / (self.iterations + 1)
            for _i, (_a, _e) in enumerate(zip(self.average_train_error, errors_train)):
                self.average_train_error[_i] = (_a * self.iterations + _e) / (self.iterations + 1)
            for _i, (_a, _e) in enumerate(zip(self.average_test_error, errors_test)):
                self.average_test_error[_i] = (_a * self.iterations + _e) / (self.iterations + 1)

        complete_train_error = sum(self.average_train_error) / len(self.average_train_error)
        complete_test_error = sum(self.average_test_error) / len(self.average_test_error)
        data_floats = [self.average_duration] + self.average_train_error + [complete_train_error] + self.average_test_error + [complete_test_error]
        data_str = (f"{self.iterations:010d}", ) + tuple(f"{_x:.5f}" for _x in data_floats)

        DataLogger.log_to(self.stats_file_path, header, data_str)

    @staticmethod
    def __get_error__(outputs, targets) -> float:
        try:
            error = sqrt(sum((_o - _t) ** 2 for _o, _t in zip(outputs, targets)))

        except TypeError:
            error = float(outputs != targets)

        return error

    def __get_data(self):
        examples_train = next(self.stream_train)
        inputs_train, targets_train = zip(*examples_train)

        examples_test = next(self.stream_test)
        inputs_test, targets_test = zip(*examples_test)

        # perform predictors and fit
        this_time = time.time()
        outputs_train = self.predictor.predict(inputs_train)
        outputs_test = self.predictor.predict(inputs_test)
        self.predictor.fit(examples_train)
        duration = time.time() - this_time

        errors_train = tuple(SetupPrediction.__get_error__(_output, _target) for _output, _target in zip(outputs_train, targets_train))
        errors_test = tuple(SetupPrediction.__get_error__(_output, _target) for _output, _target in zip(outputs_test, targets_test))

        return duration, errors_train, errors_test

    def __next__(self):
        data = self.__get_data()

        if self.iterations % self.logging_steps == 0:
            self.__log_data(self.header, *data)

        self.iterations += 1
        return data


def setup(predictor: Predictor, train_generator, test_generator, visualization_steps: int, iterations: int = 500000):
    print("Starting experiment with {:s} for {:d} iterations...".format(predictor.name(), iterations))

    average_train_error = 0.
    average_test_error = 0.
    average_duration = 0.

    # exchange rate adaptation
    # error_list = []

    for t in range(iterations):
        # get concurrent examples
        examples_train = next(train_generator)
        inputs_train, targets_train = zip(*examples_train)

        examples_test = next(test_generator)
        inputs_test, targets_test = zip(*examples_test)

        # perform predictors and fit
        this_time = time.time()
        outputs_train = predictor.predict(inputs_train)
        outputs_test = predictor.predict(inputs_test)

        predictor.fit(examples_train)

        duration = time.time() - this_time

        # todo: continue from here

        # update plot
        try:
            train_error = sum(sqrt(sum((__o - __t) ** 2 for __o, __t in zip(_o, _t))) for _o, _t in zip(outputs_train, targets_train)) / len(targets_train)
        except TypeError:
            train_error = sum(float(_o != _t) for _o, _t in zip(outputs_train, targets_train)) / len(targets_train)

        try:
            test_error = sum(sqrt(sum((__o - __t) ** 2 for __o, __t in zip(_o, _t))) for _o, _t in zip(outputs_test, targets_test)) / len(targets_test)
        except TypeError:
            test_error = sum(float(_o != _t) for _o, _t in zip(outputs_test, targets_test)) / len(targets_test)

        # exchange rate adaptation
        # if .5 < concurrent_outputs[0][0]:
        #     error_list.append(error)

        average_train_error = (average_train_error * t + train_error) / (t + 1)
        average_test_error = (average_test_error * t + test_error) / (t + 1)

        average_duration = (average_duration * t + duration) / (t + 1)
        if (t + 1) % visualization_steps == 0:
            # exchange rate adaptation
            Visualize.append("error train", predictor.__class__.__name__, average_train_error)
            Visualize.append("error test", predictor.__class__.__name__, average_test_error)

            Visualize.append("duration", predictor.__class__.__name__, average_duration)

            try:
                for _e, (each_train_output, each_train_target) in enumerate(zip(outputs_train, targets_train)):
                    for _o, (train_output_value, train_target_value) in enumerate(zip(each_train_output, each_train_target)):
                        axis_key = f"output train {_o:02d}/{_e:02d}"
                        Visualize.append(axis_key, predictor.__class__.__name__, train_output_value)
                        Visualize.append(axis_key, "target train", train_target_value)

            except TypeError:
                pass

            try:
                for _e, (each_test_output, each_test_target) in enumerate(zip(outputs_test, targets_test)):
                    for _o, (test_output_value, test_target_value) in enumerate(zip(each_test_output, each_test_target)):
                        axis_key = f"output test {_o:02d}/{_e:02d}"
                        Visualize.append(axis_key, predictor.__class__.__name__, test_output_value)
                        Visualize.append(axis_key, "target test", test_target_value)

            except TypeError:
                pass

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    Visualize.finalize("error train", predictor.__class__.__name__)
    Visualize.finalize("error test", predictor.__class__.__name__)
    Visualize.finalize("duration", predictor.__class__.__name__)
    # todo: finalize outputs?

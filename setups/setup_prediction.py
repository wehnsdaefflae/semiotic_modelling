# coding=utf-8
import datetime
import os
import sys
import time
from math import sqrt
from typing import Tuple, Generator, Any, Union, Hashable, Iterator, TypeVar, Generic, Dict, Collection, Sequence, Type

from _live_dash_plotly.send_data import SemioticVisualization
from _live_dash_plotly.visualization_server import VisualizationView
from data_generation.data_sources.systems.abstract_classes import System, Controller
from modelling.predictors.abstract_predictor import Predictor
from tools.logger import Logger, DataLogger
from tools.timer import Timer
from visualization.visualization import Visualize


VALUES = Union[Tuple[float, ...], Hashable]
EXAMPLE = Tuple[VALUES, VALUES]
CONCURRENT_EXAMPLES = Tuple[EXAMPLE, ...]


TYPE_A = TypeVar("TYPE_A")
TYPE_B = TypeVar("TYPE_B")


class Experiment(Generic[TYPE_A, TYPE_B]):
    def __init__(self,
                 predictor: Predictor[Tuple[TYPE_A, TYPE_B], TYPE_A],
                 controller: Controller[TYPE_A, TYPE_B],
                 train_system: System[TYPE_B, TYPE_A],
                 test_system: System[TYPE_B, TYPE_A]):

        self.predictor = predictor
        self.controller = controller
        self.train_system = train_system
        self.test_system = test_system

        self.train_error = 0.
        self.train_reward = 0.
        self.train_duration = 0.

        self.test_error = 0.
        self.test_reward = 0.
        self.test_duration = 0.

        self.iterations = 0

    def __str__(self):
        return ", ".join([str(_x) for _x in [self.predictor, self.controller, self.train_system, self.test_system]])

    def _adapt_average(self, previous_average: float, new_value: float) -> float:
        if self.iterations < 1:
            return new_value
        return (previous_average * self.iterations + new_value) / (self.iterations + 1)

    def step(self, steps: int) -> Dict[str, float]:
        for _ in range(steps):
            train_error, test_error = 0., 0.
            train_reward, test_reward = 0., 0.
            train_duration, test_duration = 0., 0.

            pass    # this is where the magic happens

            self.train_error = self._adapt_average(self.train_error, train_error)
            self.train_reward = self._adapt_average(self.train_reward, train_reward)
            self.train_duration = self._adapt_average(self.train_duration, train_duration)

            self.test_error = self._adapt_average(self.test_error, test_error)
            self.test_reward = self._adapt_average(self.test_reward, test_reward)
            self.test_duration = self._adapt_average(self.test_duration, test_duration)

            self.iterations += 1

        data = {
            "train error": self.train_error,
            "train reward": self.train_reward,
            "train duration": self.train_duration,
            "test error": self.test_error,
            "test reward": self.test_reward,
            "test duration": self.test_duration
        }

        return data


class ExperimentFactory(Generic[TYPE_A, TYPE_B]):
    def __init__(self,
                 predictor_class: Type[Predictor[Tuple[TYPE_A, TYPE_B], TYPE_A]], predictor_args: Dict[str, Any],
                 controller_class: Type[Controller[TYPE_B, TYPE_A]], controller_args: Dict[str, Any],
                 train_system_class: Type[System[TYPE_B, TYPE_A]], train_system_args: Dict[str, Any],
                 test_system_class: Type[System[TYPE_B, TYPE_A]], test_system_args: Dict[str, Any]):

        self.predictor_class, self.predictor_args = predictor_class, predictor_args
        self.controller_class, self.controller_args = controller_class, controller_args
        self.train_system_class, self.train_system_args = train_system_class, train_system_args
        self.test_system_class, self.test_system_args = test_system_class, test_system_args

    def __str__(self):
        return ", ".join([self.predictor_class.__name__, self.controller_class.__name__, self.train_system_class.__name__, self.test_system_class.__name__])

    def create(self) -> Experiment[TYPE_A, TYPE_B]:
        predictor = self.predictor_class(**self.predictor_args)
        controller = self.controller_class(**self.controller_args)
        train_system = self.train_system_class(**self.train_system_args)
        test_system = self.test_system_class(**self.test_system_args)
        return Experiment(predictor, controller, train_system, test_system)


class BasicResults:
    def __init__(self, experiment_names: Sequence[str]):
        self.result_names = "train reward", "test reward", "train error", "test error", "train duration", "test duration"
        self.results = {_x: tuple([] for _ in experiment_names) for _x in self.result_names}
        self.experiment_names = experiment_names

    def update(self, experiment_index: int, result: Dict[str, float]):
        for key in self.result_names:
            result_values = self.results[key]
            each_experiment = result_values[experiment_index]
            each_result = result[key]
            each_experiment.append(each_result)

    def plot(self):
        for _i, each_experiment in enumerate(self.experiment_names):
            for key in self.result_names:
                axis_name = key.split()[-1]     # ugly
                plot_name = each_experiment + " " + key
                result_values = self.results[key]
                SemioticVisualization.plot(axis_name, plot_name, result_values[_i])


class Setup(Generic[TYPE_A, TYPE_B]):
    def __init__(self, factories: Collection[ExperimentFactory[TYPE_A, TYPE_B]], repetitions: int, iterations: int):
        self.repetitions = repetitions
        self.iterations = iterations
        self.step_size = 1000

        self.factories = factories
        self.experiments = tuple((_f.create() for _f in self.factories) for _ in range(self.repetitions))

        self.axes = "reward", "error", "duration"
        SemioticVisualization.initialize(self.axes, repetitions, length=iterations)

    def _batch(self):
        collected = BasicResults(tuple(str(_f) for _f in self.factories))
        for each_comparison in self.experiments:
            for _i, each_experiment in enumerate(each_comparison):
                result = each_experiment.step(self.step_size)
                collected.update(_i, result)
        collected.plot()

    def run_experiment(self):
        for _ in range(self.iterations // self.step_size):
            self._batch()

        for _ in range(self.iterations % self.step_size):
            self._batch()


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

        # perform prediction and fit
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

        # perform prediction and fit
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

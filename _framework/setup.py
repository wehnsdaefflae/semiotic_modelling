# coding=utf-8
# !/usr/bin/env python3

import time
from typing import Tuple, Any, TypeVar, Generic, Dict, Collection, Sequence, Type, Optional

import tqdm

from _framework.streams_interaction import InteractionStream
from _framework.streams_linear import NominalAscendingDescending
from _framework.systems_abstract import Predictor, Controller, Task
from _framework.streams_abstract import ExampleStream
from _framework.systems_control import NominalRandomController
from _framework.systems_prediction import NominalLastPredictor
from _framework.systems_task import NominalGridWorld
from _live_dash_plotly.send_data import SemioticVisualization
from tools.functionality import DictList, smear
from tools.logger import Logger, DataLogger
from tools.timer import Timer

TYPE_A = TypeVar("TYPE_A")
TYPE_B = TypeVar("TYPE_B")


class Experiment(Generic[TYPE_A, TYPE_B]):
    def __init__(self,
                 name: str,
                 predictor: Predictor[TYPE_B, TYPE_A],
                 stream_train: ExampleStream[TYPE_B, TYPE_A],
                 stream_test: ExampleStream[TYPE_B, TYPE_A]):

        self._name = name

        self._predictor = predictor
        self._stream_train = stream_train
        self._stream_test = stream_test

        self._data = dict()

        self._iterations = 0

    def __str__(self):
        return self._name

    def _single_step(self):
        examples_train = self._stream_train.next()
        inputs_train, targets_train = zip(*examples_train)

        examples_test = self._stream_test.next()
        inputs_test, targets_test = zip(*examples_test)

        reward_train = self._stream_test.get_last_reward()
        reward_test = self._stream_train.get_last_reward()

        this_time = time.time()

        outputs_train = self._predictor.predict(inputs_train)
        outputs_test = self._predictor.predict(inputs_test)
        self._predictor.fit(inputs_train, targets_train)

        duration = time.time() - this_time
        errors_train = self._stream_train.error(outputs_train, targets_train)
        errors_test = self._stream_train.error(outputs_test, targets_test)

        return duration, errors_train, errors_test, reward_train, reward_test

    def step(self, steps: int) -> Dict[str, float]:
        avrg_train_error, avrg_test_error = self._data.get("error train", 0.), self._data.get("error test", 0.)
        avrg_train_reward, avrg_test_reward = self._data.get("reward train", 0.), self._data.get("reward test", 0.)
        avrg_duration = self._data.get("duration", 0.)

        for _ in range(steps):
            duration, train_error, test_error, train_reward, test_reward = self._single_step()

            avrg_train_error = smear(avrg_train_error, train_error, self._iterations)
            avrg_train_reward = smear(avrg_train_reward, train_reward, self._iterations)

            avrg_test_error = smear(avrg_test_error, test_error, self._iterations)
            avrg_test_reward = smear(avrg_test_reward, test_reward, self._iterations)

            avrg_duration = smear(avrg_duration, duration, self._iterations)

            self._iterations += 1

        self._data["error train"], self._data["error test"] = avrg_train_error, avrg_test_error
        self._data["reward train"], self._data["reward test"] = avrg_train_reward, avrg_test_reward
        self._data["duration"] = avrg_duration

        return dict(self._data)


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class ExperimentFactory(Generic[TYPE_A, TYPE_B]):
    def __init__(self,
                 predictor_def: Tuple[Type[Predictor[Tuple[TYPE_A, TYPE_B], TYPE_A]], Dict[str, Any]],
                 streams_def: Tuple[Type[ExampleStream[TYPE_B, TYPE_A]], Dict[str, Any], Dict[str, Any]],
                 controller_def: Optional[Tuple[Type[Controller], Dict[str, Any]]] = None):

        self._predictor_class, self._predictor_args = predictor_def
        self._stream_class, self._train_stream_args, self._test_stream_args = streams_def

        if controller_def is None:
            self._controller_class, self._controller_args = None, None
            self.is_interactive = False

        else:
            task_train_class = self._train_stream_args["task_class"]
            assert isinstance(task_train_class, Task)
            self._controller_args["motor_space"] = task_train_class.motor_space()

            self._train_stream_args["learn_control"] = True
            self._test_stream_args["learn_control"] = False

            self._controller_class, self._controller_args = controller_def
            self.is_interactive = True

        self._no_experiment = 0

    def create(self) -> Experiment[TYPE_A, TYPE_B]:
        predictor = self._predictor_class(**self._predictor_args)

        if self.is_interactive:
            controller = self._controller_class(**self._controller_args)
            self._train_stream_args["controller"] = controller
            self._test_stream_args["controller"] = controller

        train_system = self._stream_class(**self._train_stream_args)
        test_system = self._stream_class(**self._test_stream_args)

        name = f"({str(predictor):s}, {str(train_system):s}, {str(test_system):s}) #{self._no_experiment:03d}"
        experiment = Experiment(name, predictor, train_system, test_system)

        self._no_experiment += 1
        return experiment


class Setup(Generic[TYPE_A, TYPE_B]):
    def __init__(self, factories: Collection[ExperimentFactory[TYPE_A, TYPE_B]], no_instances: int, iterations: int, step_size: int = 1000, visualization: bool = True):
        self._no_instances = no_instances
        self._iterations = iterations
        self._step_size = step_size

        self._factories = factories
        self._experiments = tuple((_f.create() for _ in range(self._no_instances)) for _f in self._factories)

        self._finished_batches = 0

        self._axes = "reward", "error", "duration"
        self._visualization = visualization
        if self._visualization:
            SemioticVisualization.initialize(self._axes, no_instances, length=iterations)

    @staticmethod
    def _log(name: str, result: DictList[str, Sequence[float]]):
        header = []
        values_str = []
        for _plot_name, _value_list in sorted(result.items(), key=lambda _x: _x[0]):
            column_prefix = name + " " + _plot_name
            for _i, _v in enumerate(_value_list):
                header.append(column_prefix + f"{_i:03d}")
                values_str.append(f"{_v:.5f}")
        DataLogger.log_to(header, values_str, dir_path="results/")

    @staticmethod
    def _plot(name: str, axes: Sequence[str], each_result: DictList[str, Sequence[float]]):
        for _plot_name, _values in each_result.items():
            for _axis_name in axes:
                if _axis_name in _plot_name:
                    label = name + " " + _plot_name
                    SemioticVisualization.plot(_axis_name, label, _values)

    def _batch(self, no_steps: int):
        result_array = DictList()

        for _i, each_array in enumerate(self._experiments):
            full_name = ""
            for each_experiment in each_array:
                if 0 >= len(full_name):
                    full_name = str(each_experiment)
                result_single = each_experiment.step(no_steps)
                result_array.update_lists(result_single)

            name = full_name.split(" #")[0]

            Setup._log(name, result_array)
            if self._visualization:
                Setup._plot(name, self._axes, result_array)

            result_array.clear()

        self._finished_batches += 1

    def run_experiment(self):
        if self._iterations < 0:
            while True:
                self._batch(self._step_size)
                if Timer.time_passed(2000):
                    Logger.log(f"finished iteration #{self._finished_batches * self._step_size:d}")

        with tqdm.tqdm(total=self._iterations) as progress_bar:
            for _ in range(self._iterations // self._step_size):
                self._batch(self._step_size)
                progress_bar.update(self._step_size)

            remainder = self._iterations % self._step_size
            self._batch(remainder)
            progress_bar.update(remainder)


def interactive():
    experiment_factories = (
        ExperimentFactory(
            (
                NominalLastPredictor,
                dict()
            ), (
                InteractionStream,
                {
                    "task_class": NominalGridWorld,
                    "task_args": dict(),
                    "history_length": 1
                }, {
                    "task_class": NominalGridWorld,
                    "task_args": dict(),
                    "history_length": 1
                }
            ), controller_def=(
                NominalRandomController,
                dict()
            )
        ),
    )

    setup = Setup(experiment_factories, 2, 1000, step_size=100)
    setup.run_experiment()


def simple():
    experiment_factories = (
        ExperimentFactory(
            (
                NominalLastPredictor,
                {"no_states": 1}
            ), (
                NominalAscendingDescending,
                dict(),
                dict()
            )
        ),
    )

    setup = Setup(experiment_factories, 2, 1000, step_size=100, visualization=False)
    setup.run_experiment()


if __name__ == "__main__":
    simple()

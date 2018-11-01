# coding=utf-8
# !/usr/bin/env python3

import time
from typing import Tuple, Any, TypeVar, Generic, Dict, Collection, Sequence, Type, Optional

import tqdm

from _framework.systems.controllers.abstract import Controller
from _framework.systems.predictors.abstract import Predictor
from _framework.streams.abstract import ExampleStream
from _framework.systems.tasks.nominal.abstract import NominalTask
from _framework.systems.tasks.rational.abstract import RationalTask
from _live_dash_plotly.send_data import SemioticVisualization
from tools.functionality import DictList, smear
from tools.logger import Logger, DataLogger, get_time_string, get_main_script_name

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

        self.error_train = 0.
        self.error_test = 0.
        self.reward_train = 0.
        self.reward_test = 0.
        self.duration = 0.

        self._iterations = 0

    def __str__(self):
        return self._name

    def step(self) -> Tuple[float, float, float, float, float]:
        examples_test = self._stream_test.next()
        inputs_test, targets_test = zip(*examples_test)
        self.reward_test = smear(self.reward_test, self._stream_test.get_reward(), self._iterations)

        examples_train = self._stream_train.next()
        inputs_train, targets_train = zip(*examples_train)
        self.reward_train = smear(self.reward_train, self._stream_train.get_reward(), self._iterations)

        this_time = time.time()

        outputs_train = self._predictor.predict(inputs_train)
        self._predictor.fit(inputs_train, targets_train)

        self.duration = smear(self.duration, (time.time() - this_time) * 1000., self._iterations)

        outputs_test = self._predictor.predict(inputs_test)
        self.error_train = smear(self.error_train, self._stream_train.error(outputs_train, targets_train), self._iterations)
        self.error_test = smear(self.error_test, self._stream_train.error(outputs_test, targets_test), self._iterations)

        self._iterations += 1

        return self.duration, self.error_train, self.error_test, self.reward_train, self.reward_test


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
            self._controller_class, self._controller_args = controller_def
            self.is_interactive = True

            task_train_class, task_train_args = self._train_stream_args["task_def"]
            if issubclass(task_train_class, NominalTask):
                self._controller_args["motor_space"] = task_train_class.motor_space()

            elif issubclass(task_train_class, RationalTask):
                self._controller_args["motor_range"] = task_train_class.motor_range()

            else:
                raise ValueError(f"unknown task class '{task_train_class.__name__:s}'")

            self._train_stream_args["learn_control"] = True
            self._test_stream_args["learn_control"] = False

        self._no_experiment = 0

    def create(self) -> Experiment[TYPE_A, TYPE_B]:
        predictor = self._predictor_class(**self._predictor_args)

        if self.is_interactive:
            controller = self._controller_class(**self._controller_args)
            self._train_stream_args["controller"] = controller
            self._test_stream_args["controller"] = controller
            self._train_stream_args["predictor"] = predictor
            self._test_stream_args["predictor"] = predictor

        train_system = self._stream_class(**self._train_stream_args)
        test_system = self._stream_class(**self._test_stream_args)

        name = f"({str(predictor):s}, {str(train_system):s}, {str(test_system):s}) #{self._no_experiment:03d}"
        experiment = Experiment(name, predictor, train_system, test_system)

        self._no_experiment += 1
        return experiment


class Setup(Generic[TYPE_A, TYPE_B]):
    Logger.file_name = get_main_script_name() + ".log"
    Logger.dir_path = f"results/{get_time_string():s}/"

    def __init__(self, factory_args: Collection[Dict[str, Any]], no_instances: int, max_iterations: int, interval: float = 1., visualization: bool = True):
        self._no_instances = no_instances
        self._max_iterations = max_iterations
        self._interval = interval

        factories = tuple(ExperimentFactory[TYPE_A, TYPE_B](**each_args) for each_args in factory_args)
        self._experiments = tuple(tuple(_f.create() for _ in range(self._no_instances)) for _f in factories)

        self._visualization = visualization
        if self._visualization:
            SemioticVisualization.initialize(("reward", "error", "duration"), no_instances, length=max_iterations)

        self._iteration = 0

    @staticmethod
    def _save_results_batch(iteration: int, result: Dict[str, DictList[str, Sequence[float]]]):
        for each_experiment_name, each_experiment_result in result.items():
            header = ["iteration"]
            values_str = [f"{iteration:d}"]
            for _plot_name, _value_list in sorted(each_experiment_result.items(), key=lambda _x: _x[0]):
                for _i, _v in enumerate(_value_list):
                    header.append(_plot_name + f"_{_i:03d}")
                    values_str.append(f"{_v:.5f}")

            DataLogger.log_to(header, values_str, dir_path=Logger.dir_path, file_name=each_experiment_name + ".tsv")

    @staticmethod
    def _plot_batch(iteration: int, result: Dict[str, DictList[str, Sequence[float]]]):
        plot_data = tuple((_a, _p, _v) for _a, _sd in result.items() for _p, _v in _sd.items())
        SemioticVisualization.plot_batch(iteration, plot_data)

    def _batch(self, interval_sec: float):
        start_time = time.time()
        while time.time() < start_time + interval_sec:
            for _i, each_array in enumerate(self._experiments):
                for each_instance in each_array:
                    each_instance.step()
            self._iteration += 1

        duration_data = DictList()
        error_data = DictList()
        reward_data = DictList()
        file_data = {f"experiment_{_i:02d}": DictList() for _i in range(len(self._experiments))}

        for _i, each_array in enumerate(self._experiments):
            name = f"experiment_{_i:02d}"
            file_dict = file_data[name]
            for each_instance in each_array:
                duration_data.add(name + " duration", each_instance.duration)
                error_data.add(name + " train", each_instance.error_train)
                error_data.add(name + " test", each_instance.error_test)
                reward_data.add(name + " train", each_instance.reward_train)
                reward_data.add(name + " test", each_instance.reward_test)

                file_dict.add("duration", each_instance.duration)
                file_dict.add("error train", each_instance.error_train)
                file_dict.add("error test", each_instance.error_test)
                file_dict.add("reward train", each_instance.reward_train)
                file_dict.add("reward test", each_instance.reward_test)

        Setup._save_results_batch(self._iteration, file_data)
        if self._visualization:
            Setup._plot_batch(self._iteration, {"error": error_data, "reward": reward_data, "duration": duration_data})
            SemioticVisualization.update()

    def run_experiment(self):
        if 0 >= self._max_iterations:
            while True:
                self._batch(self._interval)

        with tqdm.tqdm(total=self._max_iterations) as progress_bar:
            while self._iteration < self._max_iterations:
                last_iteration = self._iteration
                self._batch(self._interval)
                progress_bar.update(self._iteration - last_iteration)


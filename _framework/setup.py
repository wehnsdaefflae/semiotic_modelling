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

        outputs_test = self._predictor.predict(inputs_test)

        this_time = time.time()

        outputs_train = self._predictor.predict(inputs_train)
        self._predictor.fit(inputs_train, targets_train)

        duration = time.time() - this_time
        errors_train = self._stream_train.error(outputs_train, targets_train)
        errors_test = self._stream_train.error(outputs_test, targets_test)

        return duration * 1000., errors_train, errors_test, reward_train, reward_test

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
            self._controller_class, self._controller_args = controller_def
            self.is_interactive = True

            task_train_class = self._train_stream_args["task_class"]
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

        train_system = self._stream_class(**self._train_stream_args)
        test_system = self._stream_class(**self._test_stream_args)

        name = f"({str(predictor):s}, {str(train_system):s}, {str(test_system):s}) #{self._no_experiment:03d}"
        experiment = Experiment(name, predictor, train_system, test_system)

        self._no_experiment += 1
        return experiment


class Setup(Generic[TYPE_A, TYPE_B]):
    Logger.file_name = get_main_script_name() + ".log"
    Logger.dir_path = f"results/{get_time_string():s}/"

    def __init__(self, factories: Collection[ExperimentFactory[TYPE_A, TYPE_B]], no_instances: int, iterations: int, step_size: int = 1000, visualization: bool = True):
        self._no_instances = no_instances
        self._iterations = iterations
        self._step_size = step_size

        self._factories = factories
        self._experiments = tuple(tuple(_f.create() for _ in range(self._no_instances)) for _f in self._factories)

        self._finished_batches = 0

        self._axes = "reward", "error", "duration"
        self._visualization = visualization
        if self._visualization:
            SemioticVisualization.initialize(self._axes, no_instances, length=iterations)
            # SemioticVisualization.initialize(self._axes, no_instances, length=-500)

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
    def _plot_batch(result: Dict[str, DictList[str, Sequence[float]]]):
        plot_data = tuple((_a, _p, _v) for _a, _sd in result.items() for _p, _v in _sd.items())
        SemioticVisualization.plot_batch(plot_data)

    def _batch(self, no_steps: int):
        plot_data = {"error": DictList(), "reward": DictList(), "duration": DictList()}
        file_data = dict()

        for _i, each_array in enumerate(self._experiments):
            name = f"experiment_{_i:02d}"

            file_data_dict = DictList()
            file_data[name] = file_data_dict

            for each_instance in each_array:
                result_single = each_instance.step(no_steps)
                file_data_dict.update_lists(result_single)

                for _name, _value in result_single.items():
                    if "error" in _name:
                        plot_data_dict = plot_data["error"]
                    elif "reward" in _name:
                        plot_data_dict = plot_data["reward"]
                    elif "duration" in _name:
                        plot_data_dict = plot_data["duration"]
                    else:
                        Logger.log(f"unknown value name {_name:s}")
                        continue

                    plot_data_dict.add(name + " " + _name, _value)

        Setup._save_results_batch(self._finished_batches * self._step_size, file_data)
        if self._visualization:
            Setup._plot_batch(plot_data)
            SemioticVisualization.update()
            # SemioticVisualization.update(steps=self._step_size)

        self._finished_batches += 1

    def run_experiment(self):
        if 0 >= self._iterations:
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

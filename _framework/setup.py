# coding=utf-8

from typing import Tuple, Any, Union, Hashable, TypeVar, Generic, Dict, Collection, Sequence, Type, Optional

from _framework.abstract_systems import Predictor, Controller, System, Task
from _live_dash_plotly.send_data import SemioticVisualization
from tools.functionality import DictList


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Process(Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, controller: Controller[SENSOR_TYPE, MOTOR_TYPE], predictor: Predictor[MOTOR_TYPE, SENSOR_TYPE]):
        self.controller = controller
        self.predictor = predictor

        self.data = dict()

    def survey(self, task_train: Task[MOTOR_TYPE, SENSOR_TYPE], iterations: int, tasks_test: None = Optional[Collection[Task[MOTOR_TYPE, SENSOR_TYPE]]]):
        survey_data = {
            ""
        }
        for _ in range(iterations):
            pass


VALUES = Union[Tuple[float, ...], Hashable]
EXAMPLE = Tuple[VALUES, VALUES]
CONCURRENT_EXAMPLES = Tuple[EXAMPLE, ...]


TYPE_A = TypeVar("TYPE_A")
TYPE_B = TypeVar("TYPE_B")


class Experiment(Generic[TYPE_A, TYPE_B]):
    def __init__(self,
                 name: str,
                 predictor: Predictor[Tuple[TYPE_A, TYPE_B], TYPE_A],
                 controller: Controller[TYPE_A, TYPE_B],
                 train_system: System[TYPE_B, TYPE_A],
                 test_systems: Optional[Collection[System[TYPE_B, TYPE_A]]] = None):

        self.name = name

        self.predictor = predictor
        self.controller = controller
        self.train_system = train_system
        self.test_systems = test_systems

        self.data_train = dict()
        self.data_test = tuple() if test_systems is None else tuple([] for _ in test_systems)

        self.train_error = 0.
        self.train_reward = 0.
        self.train_duration = 0.

        self.test_error = 0.
        self.test_reward = 0.
        self.test_duration = 0.

        self.iterations = 0

    def __str__(self):
        return self.name

    def _adapt_average(self, previous_average: float, new_value: float) -> float:
        if self.iterations < 1:
            return new_value
        return (previous_average * self.iterations + new_value) / (self.iterations + 1)

    def step(self, steps: int) -> Dict[str, float]:
        for _ in range(steps):
            train_error, test_error = 0., 0.
            train_reward, test_reward = 0., 0.
            train_duration, test_duration = 0., 0.

            raise NotImplementedError()

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


class Setup(Generic[TYPE_A, TYPE_B]):
    def __init__(self, factories: Collection[ExperimentFactory[TYPE_A, TYPE_B]], repetitions: int, iterations: int):
        self.repetitions = repetitions
        self.iterations = iterations
        self.step_size = 1000

        self.factories = factories
        # self.experiments = tuple((_f.create() for _f in self.factories) for _ in range(self.repetitions))
        self.experiments = tuple((_f.create() for _ in range(self.repetitions)) for _f in self.factories)
        self.names_experiments = tuple(str(_f) for _f in self.factories)

        self.axes = "reward", "error", "duration"
        SemioticVisualization.initialize(self.axes, repetitions, length=iterations)

    def _plot(self, name: str, each_result: DictList[str, Sequence[float]]):
        for _plot_name, _values in each_result.items():
            for _axis_name in self.axes:
                if _axis_name in _plot_name:
                    label = name + " " + _plot_name.replace(_axis_name, "")
                    SemioticVisualization.plot(_axis_name, label, _values)

    def _batch(self, no_steps: int):
        result_array = DictList()

        for _i, each_array in enumerate(self.experiments):
            for each_experiment in each_array:
                result_single = each_experiment.step(no_steps)
                result_array.update_lists(result_single)

            self._plot(self.names_experiments[_i], result_array)
            result_array.clear()

    def run_experiment(self):
        for _ in range(self.iterations // self.step_size):
            self._batch(self.step_size)

        self._batch(self.iterations % self.step_size)


if __name__ == "__main__":


    setup = Setup()

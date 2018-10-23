# coding=utf-8

from typing import Tuple, Any, Union, Hashable, TypeVar, Generic, Dict, Collection, Sequence, Type

from _framework.abstract_systems import Predictor
from _framework.abstract_trajectories import Controller, Task, ExampleStream, InteractiveStream
from _live_dash_plotly.send_data import SemioticVisualization
from tools.functionality import DictList, smear

VALUES = Union[Tuple[float, ...], Hashable]
EXAMPLE = Tuple[VALUES, VALUES]
CONCURRENT_EXAMPLES = Tuple[EXAMPLE, ...]


TYPE_A = TypeVar("TYPE_A")
TYPE_B = TypeVar("TYPE_B")


class Experiment(Generic[TYPE_A, TYPE_B]):
    def __init__(self,
                 name: str,
                 predictor: Predictor[Tuple[TYPE_A, TYPE_B], TYPE_A],
                 stream_train: ExampleStream[TYPE_B, TYPE_A],
                 stream_test: ExampleStream[TYPE_B, TYPE_A]):

        self.name = name

        self.predictor = predictor
        self.stream_train = stream_train
        self.stream_test = stream_test

        self.data = dict()

        self.iterations = 0

    def __str__(self):
        return self.name

    def _step(self):
        examples_train = self.stream_train.next()
        inputs_train, targets_train = zip(*examples_train)

        examples_test = self.stream_test.next()
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

    def step(self, steps: int) -> Dict[str, float]:
        avrg_train_error, avrg_test_error = self.data.get("train_error", 0.), self.data.get("test_error", 0.)
        avrg_train_reward, avrg_test_reward = self.data.get("train_reward", 0.), self.data.get("test_reward", 0.)
        avrg_train_duration, avrg_test_duration = self.data.get("train_duration", 0.), self.data.get("test_duration", 0.)

        for _ in range(steps):
            train_error, train_reward, train_duration, test_error, test_reward, test_duration = self._step()

            avrg_train_error = smear(avrg_train_error, train_error, self.iterations)
            avrg_train_reward = smear(avrg_train_reward, train_reward, self.iterations)
            avrg_train_duration = smear(avrg_train_duration, train_duration, self.iterations)

            avrg_test_error = smear(avrg_test_error, test_error, self.iterations)
            avrg_test_reward = smear(avrg_test_reward, test_reward, self.iterations)
            avrg_test_duration = smear(avrg_test_duration, test_duration, self.iterations)

            self.iterations += 1

        self.data["train_error"] = avrg_train_error
        self.data["train_duration"] = avrg_train_duration
        self.data["test_error"] = avrg_test_error
        self.data["test_duration"] = avrg_test_duration

        return dict(self.data)


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class InteractiveExperiment(Experiment[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self,
                 name: str,
                 predictor: Predictor[Tuple[TYPE_A, TYPE_B], TYPE_A],
                 controller: Controller[SENSOR_TYPE, MOTOR_TYPE],
                 task_train: Task[MOTOR_TYPE, SENSOR_TYPE],
                 task_test: Task[MOTOR_TYPE, SENSOR_TYPE]):
        super().__init__(name, predictor, InteractiveStream(task_train, controller), InteractiveStream(task_test, controller))
        self.data["train_reward"] = avrg_train_reward
        self.data["test_reward"] = avrg_test_reward


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

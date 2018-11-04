# coding=utf-8
from typing import TypeVar, Tuple, Any, Type, Dict, Generic

from _framework.data_types import PREDICTOR_STATE
from _framework.streams.abstract import ExampleStream
from _framework.systems.tasks.abstract import Task
from _framework.systems.controllers.abstract import Controller
from _framework.systems.predictors.abstract import Predictor

MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


SENSORIMOTOR_INPUT = Tuple[SENSOR_TYPE, MOTOR_TYPE]
SENSORIMOTOR_HISTORY = Tuple[SENSORIMOTOR_INPUT, ...]
SENSORIMOTOR_EXAMPLE = Tuple[SENSORIMOTOR_HISTORY, SENSOR_TYPE]
CONTROLLER_PERCEPTION = Tuple[SENSORIMOTOR_HISTORY, SENSOR_TYPE, PREDICTOR_STATE]


class InteractionStream(ExampleStream[SENSORIMOTOR_INPUT[SENSOR_TYPE, MOTOR_TYPE], SENSOR_TYPE], Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self,
                 task_def: Tuple[Type[Task[MOTOR_TYPE, SENSOR_TYPE]], Dict[str, Any]],
                 predictor: Predictor[MOTOR_TYPE, SENSOR_TYPE],
                 controller: Controller[CONTROLLER_PERCEPTION, MOTOR_TYPE],
                 learn_control: bool,
                 history_length: int = 1):
        super().__init__(learn_control, history_length=history_length, no_examples=1)

        self._predictor = predictor

        task_class, task_args = task_def
        self._task = task_class(**task_args)
        self._controller = controller

        self._sensor = None
        self._next_sensor = None
        self._action = None

    def __str__(self):
        return f"({str(self._task):s}, {str(self._controller):s})"

    def _before(self):
        sensorimotor_history = self._input_histories[0]
        if 0 < len(sensorimotor_history):
            sensorimotor_history.pop(0)

        perception = tuple(sensorimotor_history), self._sensor, self._predictor.get_state()
        self._action = self._controller.decide(perception)                  # (sensor, motor)*, sensor, state
        self._next_sensor, self._reward = self._task.respond(self._action)  # ((sensor, motor)*, sensor, state), motor

        if self._learn_control:
            self._controller.integrate(perception, self._action, self._reward)

    def _get_inputs(self) -> SENSORIMOTOR_INPUT[SENSOR_TYPE, MOTOR_TYPE]:
        return self._sensor, self._action

    def _get_outputs(self) -> SENSOR_TYPE:
        return self._next_sensor

    def _after(self):
        self._sensor = self._next_sensor

    def _single_error(self, data_output: SENSOR_TYPE, data_target: SENSOR_TYPE) -> float:
        return float(data_output != data_target)

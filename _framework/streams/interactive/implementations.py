# coding=utf-8
from collections import deque
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
CONTROLLER_PERCEPTION = Tuple[SENSORIMOTOR_HISTORY, PREDICTOR_STATE]


class InteractionStream(ExampleStream[SENSORIMOTOR_INPUT[SENSOR_TYPE, MOTOR_TYPE], SENSOR_TYPE], Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self,
                 task_def: Tuple[Type[Task[MOTOR_TYPE, SENSOR_TYPE]], Dict[str, Any]],
                 predictor: Predictor[MOTOR_TYPE, SENSOR_TYPE],
                 controller: Controller[CONTROLLER_PERCEPTION, MOTOR_TYPE],
                 learn_control: bool,
                 history_length: int = 1):
        super().__init__(learn_control, history_length=history_length)

        self._predictor = predictor

        task_class, task_args = task_def
        self._task = task_class(**task_args)
        self._controller = controller

        self._last_sensor = None
        self._sensorimotor_condition = tuple(self._history)

    def __str__(self):
        return f"({str(self._task):s}, {str(self._controller):s})"

    def next(self) -> Tuple[SENSORIMOTOR_EXAMPLE[SENSOR_TYPE, MOTOR_TYPE], ...]:
        state = self._predictor.get_state()
        perception = self._sensorimotor_condition, state

        motor = self._controller.decide(perception)

        sensor, self._reward = self._task.respond(motor)

        condition = self._last_sensor, motor
        self._history.append(condition)
        self._sensorimotor_condition = tuple(self._history)

        if self._learn_control:
            self._controller.integrate((self._sensorimotor_condition, state), self._reward)

        self._last_sensor = sensor

        return (self._sensorimotor_condition, sensor),

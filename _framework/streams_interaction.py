# coding=utf-8
from typing import TypeVar, Tuple

from _framework.streams_abstract import ExampleStream
from _framework.systems_abstract import Task, Controller

MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")

TASK = TypeVar("TASK", bound=Task)
CONTROLLER = TypeVar("CONTROLLER", bound=Controller)


class InteractionStream(ExampleStream[MOTOR_TYPE, SENSOR_TYPE]):
    def __init__(self, task: TASK[MOTOR_TYPE, SENSOR_TYPE], controller: CONTROLLER[SENSOR_TYPE, MOTOR_TYPE]):
        super().__init__()

        self._task = task
        self._controller = controller

        self._motor = None
        self._last_sensor = None

    def __str__(self):
        return f"({str(self._task):s}, {str(self._controller):s})"

    def next(self) -> Tuple[Tuple[MOTOR_TYPE, SENSOR_TYPE], ...]:
        sensor, reward = self._task.respond(self._motor)
        self._motor = self._controller.decide(sensor, reward)

        examples = ((self._last_sensor, self._motor), sensor),

        self._last_reward = reward
        self._last_sensor = sensor

        return examples

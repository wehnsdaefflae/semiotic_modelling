# coding=utf-8
from typing import TypeVar, Tuple, Any, Type, Dict

from _framework.streams_abstract import ExampleStream
from _framework.systems_abstract import Task, Controller, Predictor

MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")

TASK = TypeVar("TASK", bound=Task)
CONTROLLER = TypeVar("CONTROLLER", bound=Controller)


class InteractionStream(ExampleStream[MOTOR_TYPE, SENSOR_TYPE]):
    def __init__(self,
                 predictor: Predictor[MOTOR_TYPE, SENSOR_TYPE],
                 task_class: Type[TASK[MOTOR_TYPE, SENSOR_TYPE]],
                 task_args: Dict[str, Any],
                 controller: CONTROLLER[SENSOR_TYPE, MOTOR_TYPE],
                 learn_control: bool):
        super().__init__(learn_control)

        self._predictor = predictor

        self._task = task_class(**task_args)
        self._controller = controller

        self._motor = None
        self._last_sensor = None
        self._last_perception = None

    def __str__(self):
        return f"({str(self._task):s}, {str(self._controller):s})"

    def next(self) -> Tuple[Tuple[MOTOR_TYPE, SENSOR_TYPE], ...]:
        sensor, reward = self._task.respond(self._motor)
        perception = sensor + self._predictor.get_state()

        if self._learn_control:
            self._controller.integrate(self._last_perception, reward)

        self._motor = self._controller.decide(perception)

        examples = ((self._last_sensor, self._motor), sensor),

        self._last_reward = reward
        self._last_sensor = sensor
        self._last_perception = perception

        return examples

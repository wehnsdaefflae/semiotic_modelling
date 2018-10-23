# coding=utf-8
from typing import TypeVar, Generic, Tuple

from _framework.abstract_systems import Controller, Task

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class ExampleStream(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, *args, **kwargs):
        self.average_reward = 0.

    def next(self) -> Tuple[Tuple[INPUT_TYPE, OUTPUT_TYPE], ...]:
        raise NotImplementedError()

    def get_average_reward(self) -> float:
        return self.average_reward


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


class InteractiveStream(ExampleStream[MOTOR_TYPE, SENSOR_TYPE]):
    def __init__(self, task: Task[MOTOR_TYPE, SENSOR_TYPE], controller: Controller[SENSOR_TYPE, MOTOR_TYPE], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.task = task
        self.controller = controller

        self.motor = None
        self.last_sensor = None

        self.iterations = 0

    def next(self) -> Tuple[Tuple[MOTOR_TYPE, SENSOR_TYPE], ...]:
        sensor, reward = self.task.respond(self.motor)
        self.motor = self.controller.decide(sensor, reward)

        examples = ((self.last_sensor, self.motor), sensor),
        self.last_sensor = sensor

        self.average_reward = (self.average_reward * self.iterations + reward) / (self.iterations + 1.)
        self.iterations += 1
        return examples

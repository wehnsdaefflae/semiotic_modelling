#!/usr/bin/env python3
# coding=utf-8
import random
from typing import Sequence, Hashable, Tuple, TypeVar

from data_generation.data_sources.systems.abstract_classes import Controller, SENSOR_TYPE, MOTOR_TYPE

NOMINAL_SENSOR = TypeVar("NOMINAL_SENSOR", bound=Hashable)
NOMINAL_MOTOR = TypeVar("NOMINAL_MOTOR", bound=Hashable)


class NominalController(Controller[NOMINAL_SENSOR, NOMINAL_MOTOR]):
    def react_to(self, sensor: SENSOR_TYPE, reward: float) -> MOTOR_TYPE:
        pass


class NoneController(Controller[NOMINAL_SENSOR, type(None)]):
    def react_to(self, sensor: SENSOR_TYPE, reward: float) -> MOTOR_TYPE:
        return None


class RandomController(NominalController):
    def __init__(self, motor_range: Sequence[NOMINAL_MOTOR]):
        super().__init__(motor_range)

    def react_to(self, sensor: NOMINAL_SENSOR, reward: float) -> NOMINAL_MOTOR:
        return random.choice(self.motor_range)


class SarsaController(NominalController):
    def __init__(self, motor_range: Sequence[NOMINAL_MOTOR], alpha: float, gamma: float, epsilon: float, default_evaluation: float = 1000.):
        super().__init__(motor_range)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.evaluation = dict()
        self.last_perception = None
        self.last_action = None
        self.action = self.motor_range[0]
        self.default_evaluation = default_evaluation

    def _evaluate(self, perception: NOMINAL_SENSOR, action: NOMINAL_MOTOR):
        sub_dict = self.evaluation.get(perception)
        if sub_dict is None:
            return self.default_evaluation
        return sub_dict.get(action, self.default_evaluation)

    def _select_action(self, perception: NOMINAL_SENSOR) -> Tuple[NOMINAL_MOTOR, float]:
        # exploration
        if random.random() < self.epsilon:
            action = random.choice(self.motor_range)
            return action, self._evaluate(perception, action)

        # new perception
        sub_dict = self.evaluation.get(perception)
        if sub_dict is None:
            action = random.choice(self.motor_range)
            return action, self._evaluate(perception, action)

        # best action
        return max(sub_dict.items(), key=lambda _x: _x[1])

    def _update_evaluation(self, reward: float, evaluation: float):
        new_value = reward + self.gamma * evaluation
        sub_dict = self.evaluation.get(self.last_perception)

        # new perception
        if sub_dict is None:
            self.evaluation[self.last_perception] = {self.last_action: new_value}

        # known perception
        else:
            last_evaluation = sub_dict.get(self.last_action, self.default_evaluation)
            last_evaluation += self.alpha * (new_value - last_evaluation)
            sub_dict[self.last_action] = last_evaluation

    def react_to(self, sensor: NOMINAL_SENSOR, reward: float) -> NOMINAL_MOTOR:
        action, evaluation = self._select_action(sensor)
        self._update_evaluation(reward, evaluation)

        self.last_perception = sensor
        self.last_action = action
        return action

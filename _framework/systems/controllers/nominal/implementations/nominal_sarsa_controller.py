# coding=utf-8
import json
import random
from typing import Collection, Optional

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.controllers.nominal.abstract import NominalController


class NominalSarsaController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], alpha: float, gamma: float, epsilon: float, default_evaluation: float = 0.):
        super().__init__(motor_space)
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._default_evaluation = default_evaluation

        self._evaluation_function = dict()

        self._last_condition = None
        self._last_reward = 0.

    def store_evaluation_function(self, file_path: str):
        with open(file_path, mode="w") as file:
            json.dump(tuple((str(_x), _y) for _x, _y in self._evaluation_function.items()), file, sort_keys=True, indent=2)

    def react(self, perception: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        # exploration
        if random.random() < self._epsilon:
            action = self._random_action()

        else:
            # new perception
            sub_dict = self._evaluation_function.get(perception)
            if sub_dict is None:
                action = self._random_action()

            else:
                # best action
                action, _ = max(sub_dict.items(), key=lambda _x: _x[1])

        return action

    def _integrate(self, perception: NOMINAL_SENSOR, action: NOMINAL_MOTOR, reward: float):
        sub_dict = self._evaluation_function.get(perception)
        if sub_dict is None:
            this_evaluation = self._default_evaluation
        else:
            this_evaluation = sub_dict.get(action, self._default_evaluation)

        if self._last_condition is not None:
            last_perception, last_action = self._last_condition

            new_value = self._last_reward + self._gamma * this_evaluation
            last_sub_dict = self._evaluation_function.get(last_perception)

            if last_sub_dict is None:
                self._evaluation_function[last_perception] = {last_action: new_value}
            else:
                last_evaluation = last_sub_dict.get(last_action, self._default_evaluation)
                last_sub_dict[last_action] = last_evaluation + self._alpha * (new_value - last_evaluation)

        self._last_condition = perception, action
        self._last_reward = reward
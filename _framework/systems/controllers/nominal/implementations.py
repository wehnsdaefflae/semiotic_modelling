# coding=utf-8
import random
from typing import Any, Type, Collection, Tuple, Optional

from _framework.systems.controllers.nominal.abstract import NominalController
from _framework.data_types import NOMINAL_SENSOR, NOMINAL_MOTOR


class NominalNoneController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], reward: float):
        pass

    def _react(self, data_in: Any) -> Type[None]:
        return None


class NominalRandomController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)
        self.motor_space = motor_space

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], reward: float):
        pass

    def _react(self, data_in: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        action, = random.sample(self.motor_space, 1)
        return action


class NominalSarsaController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], alpha: float, gamma: float, epsilon: float, default_evaluation: float = 0.):
        super().__init__(motor_space)
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._default_evaluation = default_evaluation

        self._evaluation_function = dict()
        self._last_perception = None
        self._last_action = None
        self._evaluation = default_evaluation

    def __evaluate(self, data_in: NOMINAL_SENSOR, data_out: NOMINAL_MOTOR) -> float:
        sub_dict = self._evaluation_function.get(data_in)
        if sub_dict is None:
            return self._default_evaluation
        return sub_dict.get(data_out, self._default_evaluation)

    def _react(self, perception: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        # exploration
        if random.random() < self._epsilon:
            action = random.choice(self._motor_space)
            self._evaluation = self.__evaluate(perception, action)

        else:
            # new perception
            sub_dict = self._evaluation_function.get(perception)
            if sub_dict is None:
                action = random.choice(self._motor_space)
                self._evaluation = self.__evaluate(perception, action)

            else:
                # best action
                action, self._evaluation = max(sub_dict.items(), key=lambda _x: _x[1])

        self._last_action = action
        self._last_perception = perception
        return action

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], reward: float):
        new_value = reward + self._gamma * self._evaluation
        sub_dict = self._evaluation_function.get(self._last_perception)

        # new perception
        if sub_dict is None:
            self._evaluation_function[self._last_perception] = {self._last_action: new_value}

        # known perception
        else:
            _evaluation = self._evaluation + self._alpha * (new_value - self._evaluation)
            sub_dict[self._last_action] = _evaluation

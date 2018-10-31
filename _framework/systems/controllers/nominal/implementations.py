# coding=utf-8
import json
import random
from typing import Any, Type, Collection, Optional

from _framework.systems.controllers.nominal.abstract import NominalController
from _framework.data_types import NOMINAL_SENSOR, NOMINAL_MOTOR
from tools.logger import Logger


class NominalNoneController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)

    def integrate(self, perception: Optional[Any], action: Type[None], reward: float):
        pass

    def react(self, perception: Optional[Any]) -> Type[None]:
        return None


class NominalRandomController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)

    def integrate(self, perception: Optional[NOMINAL_SENSOR], action: NOMINAL_MOTOR, reward: float):
        pass

    def react(self, perception: Optional[NOMINAL_SENSOR]) -> NOMINAL_MOTOR:
        action, = random.sample(self._motor_space, 1)
        return action


class NominalManualController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(motor_space, *args, **kwargs)
        self._space_string = str(list(sorted(self._motor_space)))

    def react(self, perception: Optional[NOMINAL_SENSOR]) -> NOMINAL_MOTOR:
        Logger.log(f"\nController {id(self):d} perceives:\n{str(perception):s}")
        action = input(f"Target action {self._space_string:s}: ")
        while action not in self._motor_space:
            action = input(f"Action {action:s} is not among {self._space_string}. Try again: ")
        return action

    def integrate(self, perception: Optional[NOMINAL_SENSOR], action: NOMINAL_MOTOR, reward: float):
        Logger.log(f"### Controller {id(self):d} received reward: {reward:f}.")


class NominalSarsaController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], alpha: float, gamma: float, epsilon: float, default_evaluation: float = 0.):
        super().__init__(motor_space)
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._default_evaluation = default_evaluation

        self._evaluation_function = dict()

        self._save_steps = 10000
        self._iterations = 0

        self._last_condition = None
        self._last_reward = 0.

    def store_evaluation_function(self, file_path: str):
        with open(file_path, mode="w") as file:
            json.dump(tuple((str(_x), _y) for _x, _y in self._evaluation_function.items()), file, sort_keys=True, indent=2)

    def react(self, perception: Optional[NOMINAL_SENSOR]) -> NOMINAL_MOTOR:
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

    def integrate(self, perception: Optional[NOMINAL_SENSOR], action: NOMINAL_MOTOR, reward: float):
        # update last_condition evaluation with last reward and this evaluation

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
                last_sub_dict = dict()
                self._evaluation_function[last_perception] = last_sub_dict
                old_value = self._default_evaluation
            else:
                old_value = last_sub_dict.get(action, self._default_evaluation)

            last_sub_dict[last_action] = old_value + self._alpha * (new_value - old_value)

        self._last_reward = reward
        self._last_condition = perception, action

        if self._iterations >= self._save_steps:
            self._iterations = 0
            self.store_evaluation_function(self.__class__.__name__ + f"_{id(self):d}.json")

        self._last_reward = reward
        self._iterations += 1


class NominalSemioticSarsaController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(motor_space, *args, **kwargs)
        raise NotImplementedError()

    def react(self, perception: Optional[NOMINAL_SENSOR]) -> NOMINAL_MOTOR:
        raise NotImplementedError()

    def integrate(self, perception: Optional[NOMINAL_SENSOR], action: NOMINAL_MOTOR, reward: float):
        raise NotImplementedError()

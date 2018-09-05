# coding=utf-8
import random
from typing import Sequence, Any, Tuple, TypeVar, Generator, Hashable

PERCEPTION = TypeVar("PERCEPTION")
ACTION = TypeVar("ACTION")
REWARD = float

FEEDBACK = Tuple[PERCEPTION, REWARD]
CONTROLLER = Generator[ACTION, FEEDBACK, None]


def random_nominal_controller(actions: Sequence[ACTION]) -> CONTROLLER[ACTION, Any]:
    while True:
        yield random.choice(actions)


RATIONAL_ACTION = Tuple[float, ...]


def random_rational_controller(dimension: int, frame: Tuple[float, float]) -> CONTROLLER[RATIONAL_ACTION, Any]:
    while True:
        yield tuple(min(frame) + random.random() * abs(frame[0] - frame[1]) for _ in range(dimension))


NOMINAL_FEEDBACK = FEEDBACK[Hashable]


def sarsa_nominal_controller(actions: Sequence[ACTION], alpha: float, gamma: float, epsilon: float) -> CONTROLLER[ACTION, NOMINAL_FEEDBACK]:
    memory = dict()

    last_perception = None
    last_action = None
    last_reward = 0.
    action = actions[0]

    while True:
        perception, reward = yield action

        # evaluation update
        if last_perception is not None:
            last_sub_dict = memory.get(last_perception)
            if last_sub_dict is None:
                last_sub_dict = dict()
                memory[last_perception] = last_sub_dict
            last_evaluation = last_sub_dict.get(last_action, 0.)

            sub_dict = memory.get(perception)
            if sub_dict is None:
                evaluation = 0.
            else:
                evaluation = sub_dict.get(action)
            last_sub_dict[last_action] = last_evaluation + alpha * (last_reward + gamma * evaluation - last_evaluation)

        # action selection
        last_action = action
        if random.random() < epsilon:
            action = random.choice(actions)

        else:
            sub_dict = memory.get(perception)
            if sub_dict is None:
                action = actions[0]
            else:
                action, _ = max(sub_dict.items(), key=lambda _x: _x[1])

        last_perception = perception
        last_reward = reward

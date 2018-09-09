#!/usr/bin/env python3
# coding=utf-8
import random
import string
from typing import TypeVar, Generic, Tuple, Sequence, List, Optional

INPUT_DATA = TypeVar("INPUT_DATA")
OUTPUT_DATA = TypeVar("OUTPUT_DATA")


class System(Generic[INPUT_DATA, OUTPUT_DATA]):
    def react_to(self, input_data: INPUT_DATA) -> OUTPUT_DATA:
        raise NotImplementedError()


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")
EXPERIENCE = Tuple[SENSOR_TYPE, float]


class Environment(System[MOTOR_TYPE, EXPERIENCE]):
    def react_to(self, motor: Optional[MOTOR_TYPE]) -> EXPERIENCE:
        raise NotImplementedError()


class Controller(System[EXPERIENCE, MOTOR_TYPE]):
    def __init__(self, motor_range: Generic[MOTOR_TYPE]):
        self.motor_range = motor_range

    def react_to(self, experience: EXPERIENCE) -> MOTOR_TYPE:
        raise NotImplementedError()


class RandomController(Controller[str, str]):
    def __init__(self, motor_range: Sequence[str]):
        super().__init__(motor_range)

    def react_to(self, experience: EXPERIENCE) -> MOTOR_TYPE:
        return random.choice(self.motor_range)


class SarsaController(Controller[str, str]):
    def __init__(self, motor_range: Sequence[str], alpha: float, gamma: float, epsilon: float):
        super().__init__(motor_range)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.evaluation = dict()
        self.last_perception = ""
        self.last_action = ""
        self.action = self.motor_range[0]

    def react_to(self, experience: EXPERIENCE) -> MOTOR_TYPE:
        perception, reward = experience

        # evaluation update
        if 0 < len(self.last_perception):
            last_sub_dict = self.evaluation.get(self.last_perception)
            if last_sub_dict is None:
                last_sub_dict = dict()
                self.evaluation[self.last_perception] = last_sub_dict
            last_evaluation = last_sub_dict.get(self.last_action, 0.)

            sub_dict = self.evaluation.get(perception)
            if sub_dict is None:
                evaluation = 0.
            else:
                evaluation = sub_dict.get(self.action, 0.)
            last_sub_dict[self.last_action] = last_evaluation + self.alpha * (reward + self.gamma * evaluation - last_evaluation)

        # action selection
        if random.random() < self.epsilon:
            action = random.choice(self.motor_range)

        else:
            sub_dict = self.evaluation.get(perception)
            if sub_dict is None:
                action = random.choice(self.motor_range)
            else:
                action, _ = max(sub_dict.items(), key=lambda _x: _x[1])

        self.last_perception = perception
        self.last_action = action
        return action


class GridWorldAbsolute(Environment[str, Tuple[str, ...]]):
    def __init__(self, file_path: str):
        super().__init__()
        self.grid = GridWorldAbsolute._parse_text_to_grid(file_path)

        self.height = len(self.grid)
        self.width = len(self.grid[0])

        start_positions = tuple((x, y) for y, each_row in enumerate(self.grid) for x in range(len(each_row)) if each_row[x] == "s")
        goals = [(x, y, int(g)) for y, each_row in enumerate(self.grid) for x, g in enumerate(each_row) if g in string.digits]
        sorted_goals = sorted(goals, key=lambda _x: _x[2])
        self.goal_positions = [(x, y) for x, y, g in sorted_goals]
        self.current_goal_index = 0

        self.position = list(start_positions[0])    # type: List[int, int]
        self.orientation = 0                        # type: int

    @staticmethod
    def _parse_text_to_grid(file_path: str) -> Tuple[Tuple[str, ...], ...]:
        grid = []
        width = -1
        with open(file_path, mode="r") as file:
            for each_line in file:
                stripped = each_line.strip()
                if width < 0:
                    width = len(stripped)
                else:
                    assert width == len(stripped)

                grid.append(tuple(stripped))
        return tuple(grid)

    def _north(self):
        target_x = self.position[0]
        target_y = (self.position[1] - 1) % self.height
        if not self.grid[target_y][target_x] == "x":
            self.position[1] = target_y

    def _east(self):
        target_x = (self.position[0] + 1) % self.width
        target_y = self.position[1]
        if not self.grid[target_y][target_x] == "x":
            self.position[0] = target_x

    def _south(self):
        target_x = self.position[0]
        target_y = (self.position[1] + 1) % self.height
        if not self.grid[target_y][target_x] == "x":
            self.position[1] = target_y

    def _west(self):
        target_x = (self.position[0] - 1) % self.width
        target_y = self.position[1]
        if not self.grid[target_y][target_x] == "x":
            self.position[0] = target_x

    def __str__(self):
        grid_str = [["a" if [_x, _y] == self.position else _c for _x, _c in enumerate(each_row)] for _y, each_row in enumerate(self.grid)]
        return "\n".join([" ".join(each_row) for each_row in grid_str])

    def change_state(self, motor: MOTOR_TYPE):
        if motor == "r":
            self.orientation = (self.orientation + 1) % 4       # type: int

        elif motor == "l":
            self.orientation = (self.orientation - 1) % 4       # type: int

        elif motor == "f":
            if self.orientation == 0:
                self._north()
            elif self.orientation == 1:
                self._east()
            elif self.orientation == 2:
                self._south()
            elif self.orientation == 3:
                self._west()
            else:
                raise ValueError("undefined orientation")

        elif motor == "b":
            if self.orientation == 0:
                self._south()
            elif self.orientation == 1:
                self._west()
            elif self.orientation == 2:
                self._north()
            elif self.orientation == 3:
                self._east()
            else:
                raise ValueError("undefined orientation")

        elif motor == "n":
            self._north()

        elif motor == "e":
            self._east()

        elif motor == "s":
            self._south()

        elif motor == "w":
            self._west()

        else:
            raise ValueError("undefined transition")

    def react_to(self, motor: Optional[MOTOR_TYPE]) -> EXPERIENCE:
        if motor is not None:
            self.change_state(motor)

        sensor = str(self.position), str(self.orientation), str(self.current_goal_index)
        if tuple(self.position) == self.goal_positions[self.current_goal_index]:
            self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_positions)
            reward = 10.
        else:
            reward = -1.

        return sensor, reward


class GridWorld(GridWorldAbsolute):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    @staticmethod
    def _get_perception(grid: Tuple[Tuple[str, ...], ...], position: Sequence[int], orientation: int) -> Tuple[str, ...]:
        assert len(position) == 2
        x, y = position
        height = len(grid)
        width = len(grid[0])
        assert y < height
        assert x < width
        north = "x" if grid[(y - 1) % height][x] == "x" else "."
        east = "x" if grid[y][(x + 1) % width] == "x" else "."
        south = "x" if grid[(y + 1) % height][x] == "x" else "."
        west = "x" if grid[y][(x - 1) % width] == "x" else "."
        perception = north, east, south, west
        no_perceptions = len(perception)
        return tuple(perception[(orientation + _x) % no_perceptions] for _x in range(no_perceptions))

    def react_to(self, motor: Optional[MOTOR_TYPE]) -> EXPERIENCE:
        if motor is not None:
            self.change_state(motor)

        sensor = GridWorld._get_perception(self.grid, self.position, self.orientation)

        if tuple(self.position) == self.goal_positions[self.current_goal_index]:
            self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_positions)
            reward = 10.
        else:
            reward = -1.

        return sensor, reward

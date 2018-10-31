# coding=utf-8
import random
import string
from typing import Tuple, Optional


class GridWorldGlobal:
    def __init__(self, file_path: str, rotational: bool = True):
        self.actions = "f", "b", "l", "r" if rotational else "n", "e", "s", "w"
        self.grid = GridWorldGlobal._parse_text_to_grid(file_path)

        self.height = len(self.grid)
        self.width = len(self.grid[0])

        start_positions = tuple((x, y) for y, each_row in enumerate(self.grid) for x in range(len(each_row)) if each_row[x] == "s")
        goals = [(x, y, int(g)) for y, each_row in enumerate(self.grid) for x, g in enumerate(each_row) if g in string.digits]
        sorted_goals = sorted(goals, key=lambda _x: _x[2])
        self.goal_positions = [(x, y) for x, y, g in sorted_goals]
        self.current_goal_index = 0

        self.position = list(start_positions[0])
        self.orientation = 0

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

    def _change_state(self, motor: str):
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

    def react_to(self, motor: Optional[str]) -> Tuple[Tuple[Tuple[int, ...], int, int], float]:
        if motor is not None:
            self._change_state(motor)

        if tuple(self.position) == self.goal_positions[self.current_goal_index]:
            self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_positions)
            reward = 10.
        else:
            reward = -1.

        sensor = tuple(self.position), self.orientation, self.current_goal_index

        return sensor, reward


class GridWorldLocal(GridWorldGlobal):
    def __init__(self, file_path: str, rotational: bool = True):
        super().__init__(file_path, rotational=rotational)

    def _local_perception(self) -> Tuple[str, ...]:
        assert len(self.position) == 2
        x, y = self.position
        height = len(self.grid)
        width = len(self.grid[0])
        assert y < height
        assert x < width
        north = "x" if self.grid[(y - 1) % height][x] == "x" else "."
        east = "x" if self.grid[y][(x + 1) % width] == "x" else "."
        south = "x" if self.grid[(y + 1) % height][x] == "x" else "."
        west = "x" if self.grid[y][(x - 1) % width] == "x" else "."
        perception = north, east, south, west
        no_perceptions = len(perception)
        rotated_perception = tuple(perception[(self.orientation + _x) % no_perceptions] for _x in range(no_perceptions))
        return rotated_perception

    def react_to(self, motor: Optional[str]) -> Tuple[Tuple[str, ...], float]:
        if motor is not None:
            self._change_state(motor)

        sensor = self._local_perception()

        if tuple(self.position) == self.goal_positions[self.current_goal_index]:
            self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_positions)
            reward = 10.
        else:
            reward = -1.

        return sensor, reward


class DebugSarsa:
    def __init__(self, actions, alpha, gamma, epsilon):
        self._actions = actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._evaluation = dict()
        self._last_condition = None

    def get_evaluation(self, perception, action):
        sub_dict = self._evaluation.get(perception)
        if sub_dict is None:
            return 0.
        return sub_dict.get(action, 0.)

    def react(self, perception, reward):
        # action selection
        if random.random() < self._epsilon:
            action, = random.sample(self._actions, 1)
        else:
            sub_dict = self._evaluation.get(perception)
            if sub_dict is None:
                action, = random.sample(self._actions, 1)
            else:
                action, _ = max(sub_dict.items(), key=lambda _x: _x[1])

        if self._last_condition is not None and self._last_condition[1] is not None:
            # evaluation update
            evaluation = self.get_evaluation(perception, action)

            last_perception, last_action = self._last_condition
            sub_dict = self._evaluation.get(last_perception)
            if sub_dict is None:
                self._evaluation[last_perception] = {last_action: reward + self._gamma * evaluation}
            else:
                last_evaluation = sub_dict.get(last_action, 0.)
                sub_dict[last_action] = last_evaluation + self._alpha * (reward + self._gamma * evaluation - last_evaluation)

        self._last_condition = perception, action
        return action


if __name__ == "__main__":
    c = DebugSarsa(("e", "w"), .1, .5, .1)
    w = GridWorldGlobal("D:/Data/grid_worlds/simple.txt", rotational=False)

    avrg_reward = 0
    m = None
    for _i in range(1000000):
        s, r = w.react_to(m)
        m = c.react(s, r)

        avrg_reward = (avrg_reward * _i + r) / (_i + 1)

        if _i % 100000 == 0:
            print(f"{_i:d}: {avrg_reward:f}")

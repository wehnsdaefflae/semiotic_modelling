#!/usr/bin/env python3
# coding=utf-8
import string
from typing import Tuple, List

from data_generation.data_sources.systems.abstract_classes import Environment, EXPERIENCE

GRID_SENSOR = Tuple[str, ...]
GRID_MOTOR = str


class GridWorld(Environment[GRID_MOTOR, GRID_SENSOR]):
    def react_to(self, motor: GRID_MOTOR) -> EXPERIENCE[GRID_SENSOR]:
        raise NotImplementedError()


class GridWorldGlobal(GridWorld):
    def __init__(self, file_path: str):
        super().__init__()
        self.grid = GridWorldGlobal._parse_text_to_grid(file_path)

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

    def change_state(self, motor: GRID_MOTOR):
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

    def react_to(self, motor: GRID_MOTOR) -> EXPERIENCE[GRID_SENSOR]:
        if motor is not None:
            self.change_state(motor)

        sensor = str(self.position), str(self.orientation), str(self.current_goal_index)

        if tuple(self.position) == self.goal_positions[self.current_goal_index]:
            self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_positions)
            reward = 10.
        else:
            reward = -1.

        experience = sensor, reward     # type: EXPERIENCE[GRID_SENSOR]
        return experience


class GridWorldLocal(GridWorldGlobal):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def _local_perception(self) -> GRID_SENSOR:
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

    def react_to(self, motor: GRID_MOTOR) -> EXPERIENCE[GRID_SENSOR]:
        if motor is not None:
            self.change_state(motor)

        sensor = self._local_perception()

        if tuple(self.position) == self.goal_positions[self.current_goal_index]:
            self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_positions)
            reward = 10.
        else:
            reward = -1.

        experience = sensor, reward     # type: EXPERIENCE[GRID_SENSOR]
        return experience

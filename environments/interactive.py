# coding=utf-8
import random
from math import sqrt
from typing import Generator, Optional, Tuple, Sequence, Iterable


def env_gradient_world(size: int, centers: Iterable[Tuple[float, float]], tile_size: int) -> Generator[Tuple[float, ...], Optional[float], None]:
    dimensions = float(size), float(size)
    start_position = (4., 4.)
    position = list(start_position)
    momentum = [0., 0.]

    while True:
        floor_tile = float((position[0] % 2 >= 1.) == (position[1] % 2 >= 1.))
        distances = []
        for each_center in centers:
            each_distance = sqrt(sum((v1 - v2) ** 2. for v1, v2 in zip(position, each_center)))
            distances.append(each_distance)
        yield_value = floor_tile, *distances
        if any(_x < tile_size / 2. for _x in distances):
            position = list(start_position)

        impulse = yield yield_value

        if impulse is None:
            impulse = random.random() / tile_size, random.random() / tile_size

        for _i, m in enumerate(momentum):
            momentum[_i] = m + impulse[_i]
        for _i, p in enumerate(position):
            position[_i] = (p + momentum[_i]) % dimensions[_i]
        for _i, m in enumerate(momentum):
            momentum[_i] = max(0., m - .5 / tile_size) if m >= 0. else min(0., m + .5 / tile_size)


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


def _get_perception(grid: Tuple[Tuple[str, ...], ...], position: Sequence[int], orientation: int) -> Tuple[str, ...]:
    assert len(position) == 2
    x, y = position
    height = len(grid)
    width = len(grid[0])
    assert y < height
    assert x < width
    north: str = grid[y][(x - 1) % width]
    east: str = grid[(y + 1) % height][x]
    south: str = grid[y][(x + 1) % width]
    west: str = grid[(y - 1) % height][x]
    perception: Tuple[str, str, str, str] = (north, east, south, west)
    no_perceptions = len(perception)
    return tuple(perception[(orientation + _x) % no_perceptions] for _x in range(no_perceptions))


def change_state(grid: Tuple[Tuple[str, ...], ...],
                 start_positions: Tuple[Tuple[int, int], ...]) -> Generator[Tuple[Tuple[int, int], int], Optional[str], None]:
    height = len(grid)
    width = len(grid[0])

    position = list(start_positions[0])
    orientation = 0

    def _north():
        target_x = position[0]
        target_y = (position[1] - 1) % height
        if not grid[target_y][target_x] == "x":
            position[1] = target_y

    def _east():
        target_x = (position[0] + 1) % width
        target_y = position[1]
        if not grid[target_y][target_x] == "x":
            position[0] = target_x

    def _south():
        target_x = position[0]
        target_y = (position[1] + 1) % height
        if not grid[target_y][target_x] == "x":
            position[1] = target_y

    def _west():
        target_x = (position[0] - 1) % width
        target_y = position[1]
        if not grid[target_y][target_x] == "x":
            position[0] = target_x

    while True:
        # grid_str = [["a" if [_x, _y] == position else _c for _x, _c in enumerate(each_row)] for _y, each_row in enumerate(grid)]
        # print("\n".join([" ".join(each_row) for each_row in grid_str]), end="\n\n")

        motor = yield tuple(position), orientation

        if motor == "r":
            orientation = (orientation + 1) % 4

        elif motor == "l":
            orientation = (orientation - 1) % 4

        elif motor == "f":
            if orientation == 0:
                _north()
            elif orientation == 1:
                _east()
            elif orientation == 2:
                _south()
            elif orientation == 3:
                _west()
            else:
                raise ValueError("undefined orientation")

        elif motor == "b":
            if orientation == 0:
                _south()
            elif orientation == 1:
                _west()
            elif orientation == 2:
                _north()
            elif orientation == 3:
                _east()
            else:
                raise ValueError("undefined orientation")

        elif motor == "n":
            _north()

        elif motor == "e":
            _east()

        elif motor == "s":
            _south()

        elif motor == "w":
            _west()

        else:
            raise ValueError("undefined transition")


def env_grid_world(file_path: str) -> Generator[Tuple[Tuple[str, str, str, str], float], Optional[str], None]:
    grid = _parse_text_to_grid(file_path)
    start_positions = tuple((x, y) for y, each_row in enumerate(grid) for x in range(len(each_row)) if each_row[x] == "s")

    state_generator = change_state(grid, start_positions)
    position, orientation = state_generator.send(None)

    while True:
        x, y = position
        if grid[y][x] == "g":
            position = list(random.choice(start_positions))
            reward = 1.
        else:
            reward = -1.

        sensor = _get_perception(grid, position, orientation)
        motor = yield sensor, reward

        position, orientation = state_generator.send(motor)


def test_env_gradient_world():
    for sensor in env_gradient_world(8, {(4., 4.)}, 50):
        print(sensor)


if __name__ == "__main__":
    test_env_gradient_world()

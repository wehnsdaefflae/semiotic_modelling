import random
from math import sqrt
from typing import Generator, Optional, Tuple, Sequence, Iterable


def env_gradient_world(size: int, centers: Iterable[Tuple[float, float]], tile_size: int) -> Generator[Tuple[float, ...], Optional[float], None]:
    dimensions = float(size), float(size)
    position = [4., 4.]
    momentum = [0., 0.]

    while True:
        floor_tile = float((position[0] % 2 >= 1.) == (position[1] % 2 >= 1.))
        distances = []
        for each_center in centers:
            each_distance = sqrt(sum((v1 - v2) ** 2. for v1, v2 in zip(position, each_center)))
            distances.append(each_distance)
        yield_value = floor_tile, *distances
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


def _get_perception(grid: Tuple[Tuple[str, ...], ...], position: Sequence[int]) -> Tuple[str, str, str, str]:
    assert len(position) == 2
    x, y = position
    height = len(grid)
    width = len(grid[0])
    assert y < height
    assert x < width
    north = grid[x][(y - 1) % height]
    east = grid[(x + 1) % width][y]
    south = grid[x][(y + 1) % height]
    west = grid[(x - 1) % width][y]
    return north, east, south, west


def env_grid_world(file_path: str) -> Generator[Tuple[str, ...], Optional[str], None]:
    grid = _parse_text_to_grid(file_path)
    height = len(grid)
    width = len(grid[0])
    start_positions = tuple((x, y) for x, y in zip(range(width), range(height)) if grid[y][x] == "s")
    position = list(start_positions[0])

    while True:
        motor = yield _get_perception(grid, position)
        if motor is None:
            motor = random.choice("nesw")

        if motor == "n":
            position[1] = (position[1] - 1) % height

        elif motor == "e":
            position[0] = (position[0] + 1) % width

        elif motor == "s":
            position[1] = (position[1] + 1) % height

        elif motor == "w":
            position[0] = (position[0] - 1) % width

        else:
            raise ValueError()


def test_env_gradient_world():
    for sensor in env_gradient_world(8, {(4., 4.)}, 50):
        print(sensor)


if __name__ == "__main__":
    test_env_gradient_world()

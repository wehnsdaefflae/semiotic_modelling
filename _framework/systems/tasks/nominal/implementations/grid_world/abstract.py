# coding=utf-8
from typing import Optional, Tuple

from _framework.systems.tasks.nominal.implementations.grid_world.resources.grid_world import GridWorldLocal, GridWorldGlobal


class GridWorld:
    def __init__(self, rotational: bool, local: bool, file_path: str):
        self._local = local
        self._grid_wold = GridWorldLocal(file_path, rotational=rotational) if local else GridWorldGlobal(file_path, rotational=rotational)
        self._reward = 0.

    def react(self, data_in: Optional[str]) -> Tuple[str, ...]:
        output, self._reward = self._grid_wold.react_to(data_in)
        return output

    def _evaluate_action(self, data_in: Optional[str]) -> float:
        return self._reward

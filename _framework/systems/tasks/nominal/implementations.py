# coding=utf-8
from typing import Collection, Optional

from _framework.miscellaneous.grid_world import GridWorldGlobal, GridWorldLocal
from _framework.systems.tasks.nominal.abstract import NominalTask


class GridWorld:
    def __init__(self, rotational: bool, local: bool, file_path: str):
        self._local = local
        self._grid_wold = GridWorldLocal(file_path, rotational=rotational) if local else GridWorldGlobal(file_path, rotational=rotational)
        self._reward = 0.

    def react(self, data_in: Optional[str]) -> str:
        output, self._reward = self._grid_wold.react_to(data_in)
        return str(output)

    def _evaluate_action(self, data_in: Optional[str]) -> float:
        return self._reward


class RotationalMixin:
    @staticmethod
    def motor_space() -> Collection[str]:
        return "f", "b", "l", "r"


class RotationalGridWorld(GridWorld, RotationalMixin, NominalTask):
    def __init__(self, local: bool, file_path: str):
        super().__init__(True, local, file_path)


class TransitionalMixin:
    @staticmethod
    def motor_space() -> Collection[str]:
        return "e", "w"
        # return "n", "e", "s", "w"


class TransitionalGridWorld(GridWorld, TransitionalMixin, NominalTask):
    def __init__(self, local: bool, file_path: str):
        super().__init__(False, local, file_path)

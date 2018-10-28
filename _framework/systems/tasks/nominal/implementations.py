# coding=utf-8
from typing import Collection

from _framework.miscellaneous.grid_world import GridWorldGlobal, GridWorldLocal
from _framework.systems.tasks.nominal.abstract import NominalTask
from tools.load_configs import Config


class GridWorld:
    def __init__(self, rotational: bool, local: bool):
        self._local = local
        config = Config("../../configs/config.json")

        file_path = config["data_dir"] + "grid_worlds/square.txt"
        self._grid_wold = GridWorldLocal(file_path, rotational=rotational) if local else GridWorldGlobal(file_path, rotational=rotational)
        self._reward = 0.

    def _react(self, data_in: str) -> str:
        output, self._reward = self._grid_wold.react_to(data_in)
        return str(output)

    def _evaluate_action(self, data_in: str) -> float:
        return self._reward


class RotationalMixin:
    @staticmethod
    def motor_space() -> Collection[str]:
        return "f", "b", "l", "r"


class RotationalGridWorld(GridWorld, RotationalMixin, NominalTask):
    def __init__(self, local: bool):
        super().__init__(True, local)


class TransitionalMixin:
    @staticmethod
    def motor_space() -> Collection[str]:
        return "n", "e", "s", "w"


class TransitionalGridWorld(GridWorld, TransitionalMixin, NominalTask):
    def __init__(self, local: bool):
        super().__init__(False, local)

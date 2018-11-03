# coding=utf-8
from typing import Collection

from _framework.systems.tasks.nominal.abstract import NominalTask
from _framework.systems.tasks.nominal.implementations.grid_world.abstract import GridWorld


class TransitionalMixin:
    @staticmethod
    def motor_space() -> Collection[str]:
        return "n", "e", "s", "w"


class TransitionalGridWorld(GridWorld, TransitionalMixin, NominalTask):
    def __init__(self, local: bool, file_path: str):
        super().__init__(False, local, file_path)

    def __str__(self) -> str:
        return str(self._grid_wold)
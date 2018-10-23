# coding=utf-8
from typing import Any, Type

from _framework.abstract_systems import Controller


class NoneController(Controller[Any, Type[None]]):
    def __init__(self):
        super().__init__()

    def _integrate(self, evaluation: float):
        pass

    def _react(self, data_in: Any) -> Type[None]:
        return None

# coding=utf-8
from typing import Collection, Optional, Any, Type

from _framework.data_types import NOMINAL_MOTOR
from _framework.systems.controllers.nominal.abstract import NominalController


class NominalNoneController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)

    def _integrate(self, perception: Any, action: Type[None], reward: float):
        pass

    def react(self, perception: Any) -> Type[None]:
        return None

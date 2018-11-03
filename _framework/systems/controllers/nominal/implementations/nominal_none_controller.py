# coding=utf-8
from typing import Collection, Optional, Any, Type

from _framework.data_types import NOMINAL_MOTOR
from _framework.systems.controllers.nominal.abstract import NominalController


class NominalNoneController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)

    def integrate(self, perception: Optional[Any], action: Type[None], reward: float):
        pass

    def react(self, perception: Optional[Any]) -> Type[None]:
        return None
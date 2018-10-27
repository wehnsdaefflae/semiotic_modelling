# coding=utf-8
from typing import Tuple, Optional

from _framework.systems.controllers.abstract import Controller
from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR


class NominalController(Controller[NOMINAL_SENSOR, NOMINAL_MOTOR]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _react(self, data_in: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        raise NotImplementedError()

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], evaluation: float):
        raise NotImplementedError()

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()

# coding=utf-8
from typing import Tuple, Optional

from _framework.data_types import RATIONAL_MOTOR, RATIONAL_SENSOR
from _framework.systems.tasks.rational.abstract import RationalTask


# TODO: implement crypto investment task
class CryptoInvest(RationalTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def react(self, data_in: Optional[RATIONAL_MOTOR]) -> RATIONAL_SENSOR:
        pass

    def _evaluate_action(self, data_in: RATIONAL_MOTOR) -> float:
        pass

    @staticmethod
    def motor_range() -> Tuple[Tuple[float, float], ...]:
        pass

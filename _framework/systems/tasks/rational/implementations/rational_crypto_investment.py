# coding=utf-8
from typing import Tuple, Optional, Generator, List

from _framework.data_types import RATIONAL_MOTOR, RATIONAL_SENSOR
from _framework.systems.tasks.rational.abstract import RationalTask
from tools.logger import Logger


class Portfolio:
    def __init__(self, values: List[float], transfer_fee: float = .005):
        self._values = values
        self._fee_factor = 1. - transfer_fee

    def _base_transfer(self, source_currency: int, amount: float, target_currency: int, rate: float) -> float:
        assert 0. < amount
        assert 0. < rate
        assert source_currency == 0 or target_currency == 0
        current_value = self._values[source_currency]
        if current_value < amount:
            Logger.log("not enough of currency")
            return 0.
        self._values[source_currency] -= current_value
        received_value = current_value * self._fee_factor * rate
        self._values[target_currency] += received_value
        return received_value

    def transfer(self, source_currency: int, amount: float, target_currency: int, from_base_rate: float, to_base_rate: float) -> float:
        if source_currency == target_currency:
            return 0.

        elif source_currency == 0:
            return self._base_transfer(0, amount, target_currency, from_base_rate)

        elif target_currency == 0:
            return self._base_transfer(source_currency, amount, 0, to_base_rate)

        base_amount = self._base_transfer(source_currency, amount, 0, to_base_rate)
        return self._base_transfer(0, base_amount, target_currency, from_base_rate)

    def value(self, currency: int):
        return self._values[currency]

    def total_base_value(self, rates: Tuple[float, ...]) -> float:
        assert len(rates) + 1 == len(self._values)
        return sum(self._values[_i + 1] * _r for _i, _r in enumerate(rates))


# TODO: implement crypto investment task
class CryptoInvestFour(RationalTask):
    def __init__(self, initial_values: List[float], rate_sequences: Tuple[Generator[float, None, None], ...], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(initial_values) == 5
        assert len(initial_values) == len(rate_sequences)
        # five initial values, first is base value.

        self._rate_sequences = rate_sequences
        self._values = initial_values
        self._portfolio = Portfolio(self._values)

        self._total_value = -1.
        self._last_total_value = -1.

    def react(self, data_in: Optional[RATIONAL_MOTOR]) -> RATIONAL_SENSOR:
        assert len(data_in) == 4
        to_base_rates = tuple(next(_g) for _g in self._rate_sequences)
        if data_in is None:
            return to_base_rates

        for _i, _v in enumerate(data_in):
            to_base_rate = to_base_rates[_i]
            if _v < 0:
                amount = -self._portfolio.value(_i) * _v
                self._portfolio.transfer(_i, amount, 0, 1. / to_base_rate, to_base_rate)

            elif 0 < _v:
                amount = self._portfolio.value(_i) * _v
                self._portfolio.transfer(0, amount, _i, 1. / to_base_rate, to_base_rate)

        self._last_total_value = self._total_value
        self._total_value = self._portfolio.total_base_value(to_base_rates)
        return to_base_rates

    def _evaluate_action(self, data_in: RATIONAL_MOTOR) -> float:
        if self._last_total_value < 0.:
            return 0.

        return self._total_value / self._last_total_value - 1.

    @staticmethod
    def motor_range() -> Tuple[Tuple[float, float], ...]:
        return (.0, 1.), (.0, 1.), (.0, 1.), (.0, 1.)

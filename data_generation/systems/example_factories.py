#!/usr/bin/env python3
# coding=utf-8
from typing import Tuple, Collection

from data_generation.systems.abstract_classes import ExampleFactory, EXAMPLE_INPUT, EXAMPLE_OUTPUT
from environments.non_interactive import sequence_rational_crypto
from tools.load_configs import Config


class ExchangeRates(ExampleFactory[Tuple[Tuple[float, ...], ...], float]):
    def __init__(self, in_currencies: Collection[str], out_currency: str,
                 seconds_interval: int = 60, start: int=-1, end: int=-1, history_length: int=1):
        self.history_length = history_length
        c = Config("../../configs/config.json")
        data_dir = c["data_dir"] + "{:s}ETH.csv"

        file_paths = tuple(data_dir.format(each_cur.upper()) for each_cur in in_currencies)
        self.in_generators = tuple(sequence_rational_crypto(each_path, seconds_interval, start_val=start, end_val=end) for each_path in file_paths)
        self.out_generator = sequence_rational_crypto(data_dir.format(out_currency.upper()), seconds_interval, start_val=start, end_val=end)

        self.histories = tuple([] for _ in in_currencies)

    def get_example(self) -> Tuple[EXAMPLE_INPUT, EXAMPLE_OUTPUT]:
        for _ in range(self.history_length - len(self.histories[0])):
            for _i, each_history in enumerate(self.histories):
                each_history.append(next(self.in_generators[_i]))

        input_value = tuple(tuple(each_history) for each_history in self.histories)
        target_value = next(self.out_generator)

        for _ in range(len(self.histories[0]) - self.history_length + 1):
            for each_history in self.histories:
                each_history.pop(0)

        return input_value, target_value

# coding=utf-8
from typing import TypeVar, Generic, Tuple


INPUT_TYPE = TypeVar("INPUT_TYPE")
INPUT_HISTORY = Tuple[INPUT_TYPE, ...]
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")
EXAMPLE_TYPE = Tuple[INPUT_HISTORY, OUTPUT_TYPE]

ANY_TYPE = TypeVar("ANY_TYPE")


class ExampleStream(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, learn_control: bool, no_examples: int = 1, history_length: int = 1):
        self._learn_control = learn_control
        self._no_examples = no_examples
        self._reward = 0.
        self._history_length = history_length

        self._input_histories = tuple([] for _ in range(no_examples))

    def __str__(self):
        raise NotImplementedError()

    def __memorize_inputs(self, inputs: Tuple[INPUT_TYPE, ...]):
        if self._history_length < 1:
            return

        for each_history, each_input in zip(self._input_histories, inputs):
            each_history.append(each_input)
            del each_history[:-self._history_length]

    def _before(self):
        raise NotImplementedError()

    def _get_inputs(self) -> Tuple[INPUT_TYPE, ...]:
        raise NotImplementedError()

    def _get_outputs(self) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError()

    def _after(self):
        raise NotImplementedError()

    def next(self) -> Tuple[EXAMPLE_TYPE, ...]:
        self._before()

        inputs = self._get_inputs()
        self.__memorize_inputs(inputs)
        outputs = self._get_outputs()
        examples = tuple((tuple(each_input_history), each_output) for each_input_history, each_output in zip(self._input_histories, outputs))

        self._after()
        return examples

    def get_reward(self) -> float:
        return self._reward

    def _single_error(self, data_output: OUTPUT_TYPE, data_target: OUTPUT_TYPE) -> float:
        raise NotImplementedError()

    def total_error(self, data_outputs: Tuple[OUTPUT_TYPE, ...], data_targets: Tuple[OUTPUT_TYPE, ...]) -> float:
        return sum(self._single_error(each_output, each_target) for each_output, each_target in zip(data_outputs, data_targets)) / self._no_examples

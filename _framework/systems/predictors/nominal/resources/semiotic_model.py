# coding=utf-8
from typing import TypeVar, Generic, Callable, Tuple, Sequence

from modelling.content import ContentFactory

from modelling import semiotic_functions

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class SemioticModel(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self,
                 no_examples: int, alpha: int, sigma: float, is_nominal: bool,
                 trace_length: int = 1,
                 fix_level_size_at: Callable[[int], int] = lambda _level: -1,
                 input_dimensions: int = 1,
                 output_dimensions: int = 1,
                 drag: int = 1):

        if is_nominal:
            assert input_dimensions == 1
            assert output_dimensions == 1
            assert drag == 1

            self._model = [{0: self._base_content_factory.nominal(0)}]                                    # type: semiotic_functions.MODEL
        else:
            self._model = [{0: self._base_content_factory.rational(0)}]                                   # type: semiotic_functions.MODEL

        self._base_content_factory = ContentFactory(input_dimensions, output_dimensions, drag, alpha)

        self._traces = tuple([] for _ in range(no_examples))                                             # type: Tuple[semiotic_functions.TRACE, ...]

        self._is_nominal = is_nominal

        self._alpha = alpha
        self._sigma = sigma
        self._trace_length = trace_length
        self._fix_level_size_at = fix_level_size_at

        self._states = tuple([0] for _ in range(no_examples))                                            # type: Tuple[semiotic_functions.STATE, ...]

    def _update_states(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        for _i, (input_value, target_value) in enumerate(zip(input_values, target_values)):
            semiotic_functions.update_state(input_value, target_value,
                                            self._model, self._traces[_i], self._states[_i],
                                            self._sigma, self._fix_level_size_at)

    def fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        self._update_states(input_values, target_values)
        semiotic_functions.generate_state_layer(self._model, self._states)
        semiotic_functions.generate_content(self._model, self._states, self._base_content_factory, self._is_nominal)
        semiotic_functions.generate_trace_layer(self._trace_length, self._model, self._traces)

        semiotic_functions.adapt_abstract_content(self._model, self._traces, self._states)
        semiotic_functions.adapt_base_contents(input_values, target_values, self._model, self._states)

        semiotic_functions.update_traces(self._traces, self._states, self._trace_length)

    def predict(self, input_values: Sequence[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE, ...]:
        output_values = semiotic_functions.get_outputs(input_values, self._model, self._states)
        return tuple(output_values)

    def get_structure(self) -> Tuple[int, ...]:
        return tuple(len(_x) for _x in self._model)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(each_state) for each_state in self._states)

    def get_certainty(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]) -> Tuple[float, ...]:
        base_shapes = tuple(each_state[0] for each_state in self._states)
        base_layer = self._model[0]
        base_contents = tuple(base_layer[each_shape] for each_shape in base_shapes)
        return tuple(content.probability(_input, _target) for content, _input, _target in zip(base_contents, input_values, target_values))

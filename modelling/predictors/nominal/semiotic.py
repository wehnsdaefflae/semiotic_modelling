# coding=utf-8
from typing import Callable, Tuple

from modelling.content import ContentFactory
from modelling.predictors.abstract_predictor import Predictor, NOMINAL_INPUT, NOMINAL_OUTPUT
from modelling.semiotic_functions import update_state, generate_state_layer, generate_content, generate_trace_layer, \
    adapt_abstract_content, adapt_base_contents, update_traces, get_outputs, TRACE, MODEL, STATE


class NominalSemioticModel(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, no_examples: int, alpha: int, sigma: float, trace_length: int,
                 fix_level_size_at: Callable[[int], int] = lambda _level: -1):
        super().__init__(no_examples)
        self.base_content_factory = ContentFactory(1, 1, 1, alpha)
        self.alpha = alpha
        self.sigma = sigma
        self.trace_length = trace_length
        self.fix_level_size_at = fix_level_size_at

        self.model = [{0: self.base_content_factory.nominal(0)}]                              # type: MODEL
        self.traces = tuple([[0 for _ in range(trace_length)]] for _ in range(no_examples))   # type: Tuple[TRACE, ...]
        self.states = tuple([0] for _ in range(no_examples))                                  # type: Tuple[STATE, ...]
        self.last_input = None
        self.last_target = None

    def _update_states(self, input_values: Tuple[NOMINAL_INPUT, ...], target_values: Tuple[NOMINAL_OUTPUT, ...]):
        for _i, (input_value, target_value) in enumerate(zip(input_values, target_values)):
            update_state(input_value, target_value,
                         self.model, self.traces[_i], self.states[_i],
                         self.sigma, self.fix_level_size_at)

    def _fit(self, abs_input: Tuple[Tuple[NOMINAL_INPUT, ...], ...], abs_target: Tuple[NOMINAL_OUTPUT, ...]):
        input_values = abs_input
        target_values = abs_target

        self._update_states(input_values, target_values)
        generate_state_layer(self.model, self.states)
        generate_content(self.model, self.states, self.base_content_factory, True)
        generate_trace_layer(self.trace_length, self.model, self.traces)

        adapt_abstract_content(self.model, self.traces, self.states)
        adapt_base_contents(input_values, target_values, self.model, self.states)

        update_traces(self.traces, self.states, self.trace_length)

        self.last_input = abs_input
        self.last_target = abs_target

    def _predict(self, input_values: Tuple[Tuple[NOMINAL_INPUT, ...], ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        output_values = get_outputs(input_values, self.model, self.states)
        return tuple(output_values)

    def save(self, file_path: str):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        return tuple(len(_x) for _x in self.model)

    def get_states(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(each_state) for each_state in self.states)

    def get_certainty(self, input_values: Tuple[NOMINAL_INPUT, ...], target_values: Tuple[NOMINAL_OUTPUT, ...]) -> Tuple[float, ...]:
        base_shapes = tuple(each_state[0] for each_state in self.states)
        base_layer = self.model[0]
        base_contents = tuple(base_layer[each_shape] for each_shape in base_shapes)
        return tuple(content.probability(_input, _target) for content, _input, _target in zip(base_contents, input_values, target_values))

    def _get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(_sub_state) for _sub_state in self.states)

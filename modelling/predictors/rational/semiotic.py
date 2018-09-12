# coding=utf-8
from typing import Callable, Tuple

from modelling.content import ContentFactory
from modelling.predictors.abstract_predictor import INPUT_TYPE, OUTPUT_TYPE
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.semiotic_functions import generate_state_layer, generate_content, generate_trace_layer, \
    adapt_abstract_content, adapt_base_contents, update_traces, MODEL


class RationalSemioticModel(NominalSemioticModel):
    # TODO: instead of fix_level_size_at preconstruct model and prohibit content generation with boolean
    # avoids problem of two states writing to the same content until model is fixed
    def __init__(self, input_dimension: int, output_dimension: int, no_examples: int, alpha: int, sigma: float, drag: int, trace_length: int,
                 fix_level_size_at: Callable[[int], int] = lambda _level: -1):
        super().__init__(no_examples, alpha, sigma, trace_length, fix_level_size_at=fix_level_size_at)
        self.output_dimensions = output_dimension
        self.base_content_factory = ContentFactory(input_dimension, output_dimension, drag, alpha)
        self.model = [{0: self.base_content_factory.rational(0)}]                                           # type: MODEL

    def _fit(self, abs_input: Tuple[Tuple[INPUT_TYPE, ...], ...], abs_target: Tuple[OUTPUT_TYPE, ...]):
        input_values = abs_input
        target_values = abs_target

        self._update_states(input_values, target_values)
        generate_state_layer(self.model, self.states)
        generate_content(self.model, self.states, self.base_content_factory, False)
        generate_trace_layer(self.trace_length, self.model, self.traces)

        adapt_abstract_content(self.model, self.traces, self.states)
        adapt_base_contents(input_values, target_values, self.model, self.states)

        update_traces(self.traces, self.states, self.trace_length)

        self.last_input = abs_input
        self.last_target = abs_target

    def save(self, file_path: str):
        raise NotImplementedError

from typing import Generic, TypeVar, Tuple, Callable, Hashable

from modelling.content import ContentFactory, NominalContent
from modelling.semiotic_functions import update_state, generate_state_layer, generate_content, generate_trace_layer, adapt_abstract_content, \
    adapt_base_contents, update_traces, get_outputs, MODEL, TRACE, STATE
from tools.regression import Regressor

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


# TODO: differentiate no. inputs and input size b,kiloß# ß# !
class Predictor(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int):
        self.no_examples = no_examples

    def _fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        raise NotImplementedError

    def fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        assert len(input_values) == len(target_values) == self.no_examples
        self._fit(input_values, target_values)

    def save(self, file_path):
        raise NotImplementedError

    def _predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError

    def predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        assert len(input_values) == self.no_examples
        output_values = self._predict(input_values)
        assert self.no_examples == len(output_values)
        return output_values

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError


RATIONAL_VECTOR = Tuple[float, ...]


class MovingAverage(Predictor[RATIONAL_VECTOR, RATIONAL_VECTOR]):
    def __init__(self, output_dimensions: int, no_examples: int, drag: int):
        super().__init__(no_examples)
        self.output_dimensions = output_dimensions
        self.drag = drag
        self.average = tuple([0. for _ in range(output_dimensions)] for _ in range(self.no_examples))
        self.initial = True

    def _fit(self, input_values: Tuple[RATIONAL_VECTOR, ...], target_values: Tuple[RATIONAL_VECTOR, ...]):
        if self.initial:
            for each_target, each_average in zip(target_values, self.average):
                for _i, each_target_value in enumerate(each_target):
                    each_average[_i] = each_target_value
            self.initial = False

        else:
            for each_target, each_average in zip(target_values, self.average):
                for _i, each_target_value in enumerate(each_target):
                    each_average[_i] = (each_average[_i] * self.drag + each_target_value) / (self.drag + 1)

    def _predict(self, input_values: Tuple[RATIONAL_VECTOR, ...]) -> Tuple[RATIONAL_VECTOR, ...]:
        return tuple(tuple(each_average) for each_average in self.average)

    def save(self, file_path):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError


class Regression(Predictor[RATIONAL_VECTOR, RATIONAL_VECTOR]):
    def __init__(self, input_dimension: int, output_dimension: int, no_examples: int, drag: int):
        super().__init__(no_examples)
        self.input_dimension = input_dimension
        self.drag = drag
        self.regressions = tuple(tuple(Regressor(input_dimension, self.drag) for _ in range(output_dimension)) for _ in range(no_examples))

    def _fit(self, input_values: Tuple[RATIONAL_VECTOR, ...], target_values: Tuple[RATIONAL_VECTOR, ...]):
        for example_index in range(self.no_examples):
            each_regression = self.regressions[example_index]
            each_input = input_values[example_index]
            each_target = target_values[example_index]
            for each_single_regression, each_target_value in zip(each_regression, each_target):
                each_single_regression.fit(each_input, each_target_value)

    def _predict(self, input_values: Tuple[RATIONAL_VECTOR, ...]) -> Tuple[RATIONAL_VECTOR, ...]:
        return tuple(
                tuple(single_regression.output(each_input) for single_regression in each_regression)
                for each_regression, each_input in zip(self.regressions, input_values)
        )

    def save(self, file_path):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError


class NominalMarkovModel(Predictor[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int):
        super().__init__(no_examples)
        self.models = tuple(NominalContent(0, 0) for _ in range(no_examples))

    def _fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        for _i, (each_input, each_target) in enumerate(zip(input_values, target_values)):
            each_model = self.models[_i]
            each_model.adapt(each_input, each_target)

    def save(self, file_path):
        pass

    def _predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        return tuple(each_model.predict(input_values[_i]) for _i, each_model in enumerate(self.models))

    def get_structure(self) -> Tuple[int, ...]:
        return 0,


class NominalSemioticModel(Predictor[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int, alpha: int, sigma: float, trace_length: int,
                 fix_level_size_at: Callable[[int], int] = lambda _level: -1):
        super().__init__(no_examples)
        self.base_content_factory = ContentFactory(1, 1, 1, alpha)
        self.alpha = alpha
        self.sigma = sigma
        self.trace_length = trace_length
        self.fix_level_size_at = fix_level_size_at

        self.model = [{0: self.base_content_factory.nominal(0)}]                                            # type: MODEL
        self.traces = tuple([[0 for _ in range(trace_length)]] for _ in range(no_examples))                 # type: Tuple[TRACE, ...]
        self.states = tuple([0] for _ in range(no_examples))                                                # type: Tuple[STATE, ...]
        self.last_input = None
        self.last_target = None

    def _update_states(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        for _i, (input_value, target_value) in enumerate(zip(input_values, target_values)):
            update_state(input_value, target_value, self.model, self.traces[_i], self.states[_i], self.sigma, self.fix_level_size_at)

    def _fit(self, abs_input: Tuple[INPUT_TYPE, ...], abs_target: Tuple[OUTPUT_TYPE, ...]):
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

    def _predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        output_values = get_outputs(input_values, self.model, self.states)
        return tuple(output_values)

    def save(self, file_path: str):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        return tuple(len(_x) for _x in self.model)

    def get_states(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(each_state) for each_state in self.states)

    def get_certainty(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]) -> Tuple[float, ...]:
        base_shapes = tuple(each_state[0] for each_state in self.states)
        base_layer = self.model[0]
        base_contents = tuple(base_layer[each_shape] for each_shape in base_shapes)
        return tuple(content.probability(_input, _target) for (content, _input, _target) in zip(base_contents, input_values, target_values))


class RationalSemioticModel(NominalSemioticModel):
    # TODO: instead of fix_level_size_at preconstruct model and prohibit content generation with boolean
    # avoids problem of two states writing to the same content until model is fixed
    def __init__(self, input_dimensions: int, output_dimensions: int, no_examples: int, alpha: int, sigma: float, drag: int, trace_length: int,
                 fix_level_size_at: Callable[[int], int] = lambda _level: -1):
        super().__init__(no_examples, alpha, sigma, trace_length, fix_level_size_at=fix_level_size_at)
        self.output_dimensions = output_dimensions
        self.base_content_factory = ContentFactory(input_dimensions, output_dimensions, drag, alpha)
        self.model = [{0: self.base_content_factory.rational(0)}]                                           # type: MODEL

    def _fit(self, abs_input: Tuple[INPUT_TYPE, ...], abs_target: Tuple[OUTPUT_TYPE, ...]):
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

from __future__ import annotations
from typing import Hashable, Sequence, Collection, Generator

from representation import Representation


class SemioticModel[C: Hashable, E: Hashable]:
    type Shape = int

    @staticmethod
    def build[C, E](levels: Sequence[int], threshold: float = .1, frozen: bool = True) -> SemioticModel[C, E]:
        base_model = SemioticModel[C, E](threshold=threshold, frozen=frozen)
        current_model = base_model

        for i, each_level in enumerate(levels):
            for each_predictor in range(each_level - 1):
                current_model._generate_predictor()

            if i < len(levels) - 1:
                each_model = SemioticModel[SemioticModel.Shape, SemioticModel.Shape](threshold=threshold, frozen=frozen)
                current_model.parent = each_model
                current_model = each_model

        return base_model

    @staticmethod
    def _check_current(
            predictor: Representation[C, E, Shape],
            cause: C, effect_observed: E,
            threshold: float,
            open_world: bool = True
    ) -> bool:
        fit = predictor.max_scaled_fit(cause, effect_observed)
        if fit < 0.:
            return open_world
        return fit >= threshold

    @staticmethod
    def _check_expected(
            predictor: Representation[C, E, Shape],
            cause: C, effect_observed: E,
            threshold: float,
            open_world: bool = True
    ) -> bool:
        fit = predictor.prop_scaled_fit(cause, effect_observed)
        if fit < 0.:
            return open_world
        return fit >= threshold

    @staticmethod
    def _check_best(
            predictor: Representation[C, E, Shape],
            cause: C, effect_observed: E,
            threshold: float,
            open_world: bool = True) -> bool:
        fit = predictor.strict_fit(cause, effect_observed)
        if fit < 0.:
            return open_world
        return fit >= threshold

    @staticmethod
    def _find_best_predictor(
            predictors: Collection[Representation],
            cause: C, effect: E,
            open_world: bool = True) -> Representation[C, E, Shape]:

        def fit_wrapper(predictor: Representation[C, E, SemioticModel.Shape]) -> float:
            fit = predictor.prop_scaled_fit(cause, effect)
            if fit < 0.:
                return float(open_world)
            return fit

        best_predictor = max(predictors, key=fit_wrapper)
        return best_predictor

    def __init__(self, level: int = 0, threshold: float = .1, frozen: bool = True, open_world: bool = True) -> None:
        self.level = level
        self.threshold_current = threshold
        self.threshold_expected = threshold
        self.threshold_best = threshold
        self.frozen = frozen
        self.open_world = open_world

        self.cache_representation = Representation[C, E, SemioticModel.Shape](-1)
        self.likelihood = 1.

        self._last_predictor_shape: SemioticModel.Shape | None = None

        self._no_predictors = 0
        self.predictors = dict[SemioticModel.Shape, Representation[C, E, SemioticModel.Shape]]()
        self.this_predictor = self._generate_predictor()

        self.parent: SemioticModel[SemioticModel.Shape, SemioticModel.Shape] | None = None

    @property
    def state(self) -> tuple[Shape, ...]:
        return tuple(each_level.this_predictor.shape for each_level in self.parent_iter())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(each_level.predictors) for each_level in self.parent_iter())

    def _generate_predictor(self) -> Representation[C, E, Shape]:
        predictor = Representation[C, E, int](self._no_predictors)
        self.predictors[predictor.shape] = predictor
        self._no_predictors += 1
        return predictor

    def _handle_unexpected(self, query_representation: Representation[C, E, Shape]) -> Representation[C, E, Shape]:
        open_world = False

        predictor_best = None
        best_fit = -1.
        for each_predictor in self.predictors.values():
            each_fit = each_predictor.max_scaled_likelihood(query_representation, open_world=open_world)
            if best_fit < each_fit:
                best_fit = each_fit
                predictor_best = each_predictor

        is_best = best_fit >= self.threshold_best
        if is_best or self.frozen:
            return predictor_best

        predictor_new = self._generate_predictor()
        return predictor_new


    def _handle_breakdown(
            self, cause: C, effect: E, query_representation: Representation[C, E, Shape]
    ) -> Representation[C, E, Shape]:

        best_predictor_definitive = None
        max_value_definitive = -1.

        best_predictor_provisional = None
        max_value_provisional = -1.

        # get definitive last_representation (with duration)
        # get provisional this_representation (for prediction)
        # use viterbi to find the best predictor tuple
        # remember transition sequence instead of query_representation
        for each_predictor in self.predictors.values():
            each_value_definitive = each_predictor.max_scaled_likelihood(query_representation, self.open_world)
            each_value_provisional = each_predictor.max_scaled_fit(cause, effect)

            if self.parent is not None:
                each_top_value_definitive = self.parent.this_predictor.max_scaled_fit(self._last_predictor_shape, each_predictor.shape)
                each_value_definitive = each_value_definitive * each_top_value_definitive + each_value_definitive

                each_top_value_provisional = self.parent.this_predictor.max_scaled_fit(self.this_predictor.shape, each_predictor.shape)
                each_value_provisional = each_value_provisional * each_top_value_provisional + each_value_provisional

            if max_value_definitive < each_value_definitive:
                max_value_definitive = each_value_definitive
                best_predictor_definitive = each_predictor

            if max_value_provisional < each_value_provisional:
                max_value_provisional = each_value_provisional
                best_predictor_provisional = each_predictor

        if self.parent is not None:
            self.parent.transition(self._last_predictor_shape, best_predictor_definitive.shape, duration=len(query_representation))

        self._last_predictor_shape = best_predictor_definitive.shape
        self.this_predictor = best_predictor_provisional


    def __handle_breakdown(self, query_representation: Representation[C, E, Shape]) -> Representation[C, E, Shape]:
        representation_finished = self.this_predictor

        if 1 < len(self.this_predictor):
            is_expected = False

            if self.parent is not None:
                predictor_expected_shape = self.parent.predict(
                    self._last_predictor_shape,
                    default=self.this_predictor.shape
                )
                representation_finished = self.predictors.get(predictor_expected_shape, self.this_predictor)
                likelihood = representation_finished.max_scaled_likelihood(query_representation, open_world=self.open_world)
                is_expected = likelihood >= self.threshold_expected

            if self.parent is None or not is_expected:
                representation_finished = self._handle_unexpected(query_representation)

            if self.parent is None and len(self.predictors) >= 2:
                self.parent = SemioticModel[C, E](level=self.level+1, frozen=self.frozen)

        representation_finished.update(query_representation)

        predictor_assumed_shape = representation_finished.shape
        if self.parent is not None:
            if self._last_predictor_shape is not None:
                duration = len(query_representation)
                self.parent.transition(self._last_predictor_shape, representation_finished.shape, duration=duration)

            predictor_assumed_shape = self.parent.predict(
                representation_finished.shape, default=representation_finished.shape
            )

        self._last_predictor_shape = representation_finished.shape

        return self.predictors.get(predictor_assumed_shape, self.this_predictor)

    def transition(self, cause: C, effect: E, duration: int = 1) -> None:
        base_predictor = self.cache_representation + self.this_predictor
        this_fit = base_predictor.max_scaled_fit(cause, effect)

        self.likelihood *= abs(this_fit) if self.open_world else max(0., this_fit)
        is_breakdown = self.likelihood < self.threshold_current

        if is_breakdown:
            self.this_predictor = self._handle_breakdown(cause, effect, self.cache_representation)

            self.likelihood = self.this_predictor.max_scaled_fit(cause, effect)
            self.likelihood = abs(self.likelihood) if self.open_world else max(0., self.likelihood)
            self.cache_representation.clear()

        else:
            self.cache_representation.transition(cause, effect, duration)

    def predict(self, cause: C, default: E | None = None) -> E | None:
        base_predictor = self.cache_representation + self.this_predictor
        return base_predictor.predict(cause, default=default)

    def parent_iter(self) -> Generator[SemioticModel[Shape, Shape], None, None]:
        yield self
        level = self.parent
        while level is not None:
            yield level
            level = level.parent

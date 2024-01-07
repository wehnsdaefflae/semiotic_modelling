from __future__ import annotations
from typing import Hashable, Generator, Sequence, Collection

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

        self._pre_last_predictor_shape: SemioticModel.Shape | None = None
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
        if not is_best and not self.frozen:
            predictor_new = self._generate_predictor()
            return predictor_new

        return predictor_best

    def _handle_breakdown(self, query_representation: Representation[C, E, Shape]) -> Representation[C, E, Shape]:
        is_expected = False
        predictor_queried = self.this_predictor

        if self.parent is not None:
            predictor_expected_shape = self.parent.predict(
                self._last_predictor_shape,
                default=self._last_predictor_shape
            )
            predictor_queried = self.predictors.get(predictor_expected_shape, self.this_predictor)
            likelihood = predictor_queried.max_scaled_likelihood(query_representation, open_world=self.open_world)
            is_expected = likelihood >= self.threshold_expected

        if self.parent is None or not is_expected:
            predictor_queried = self._handle_unexpected(query_representation)

        predictor_queried.update(query_representation)

        if self.parent is None:
            self.parent = SemioticModel[C, E](level=self.level+1, frozen=self.frozen)

        self._pre_last_predictor_shape = self._last_predictor_shape
        self._last_predictor_shape = predictor_queried.shape
        predictor_assumed_shape = self.parent.predict(self._last_predictor_shape, default=self._last_predictor_shape)

        if self._pre_last_predictor_shape is not None:
            duration = len(self.cache_representation)
            self.parent.transition(self._pre_last_predictor_shape, self._last_predictor_shape, duration=duration)

        return self.predictors.get(predictor_assumed_shape, self.this_predictor)

    def transition(self, cause: C, effect: E, duration: int = 1) -> None:
        base_predictor = self.cache_representation + self.this_predictor
        this_fit = base_predictor.max_scaled_fit(cause, effect)

        self.likelihood *= abs(this_fit) if self.open_world else max(0., this_fit)
        is_breakdown = self.likelihood < self.threshold_current

        if is_breakdown:
            self.this_predictor = self._handle_breakdown(self.cache_representation)

            self.likelihood = 1.
            self.cache_representation.clear()

        self.cache_representation.transition(cause, effect, duration)

    def predict(self, cause: C, default: E | None = None) -> E | None:
        return self.this_predictor.predict(cause, default=default)

    def parent_iter(self) -> Generator[SemioticModel[Shape, Shape], None, None]:
        yield self
        level = self.parent
        while level is not None:
            yield level
            level = level.parent


def iterate_text() -> Generator[str, None, None]:
    with open("/home/mark/nas/data/text/lovecraft_namelesscity.txt", mode="r") as file:
        for each_line in file:
            for each_char in each_line.strip():
                yield each_char.lower()


def main() -> None:
    # model = SemioticModel.build([5, 3], frozen=True)
    model = SemioticModel[str, str](threshold=.9, frozen=False)
    # model = SemioticModel.build([1], frozen=True)

    last_char = ""
    total = success = 0

    for i, char in enumerate(iterate_text()):
        if i % 1_000 == 0:
            accuracy = 0. if total < 1 else success / total
            print(f"Accuracy: {accuracy}")
            print(f"State: {model.state}")
            print(f"Shape: {model.shape}")

        if len(last_char) >= 1:
            prediction = model.predict(last_char, default=last_char)
            is_correct = prediction == char
            model.transition(last_char, char)

        else:
            is_correct = False

        success += is_correct
        total += 1

        last_char = char


if __name__ == "__main__":
    main()

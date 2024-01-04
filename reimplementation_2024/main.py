from __future__ import annotations
import dataclasses
from typing import Hashable, Generator, Sequence


@dataclasses.dataclass
class TransitionInfo:
    frequency: int = 1
    average_duration: float = 1


class Representation[C: Hashable, E: Hashable, S: Hashable]:
    def __init__(self, shape: S) -> None:
        self.shape = shape
        self.content = dict[C, dict[E, TransitionInfo]]()

    def match_strict(self, cause: C, effect: E, open_world: bool = True) -> bool:
        prediction = self.predict(cause)
        if prediction is None and open_world:
            return True
        return prediction == effect

    def match_threshold(self, cause: C, effect: E, threshold: float) -> bool:
        transition_info = self.get_transition_info(cause, effect)
        return transition_info.average_duration >= threshold

    def match_value(self, cause: C, effect: E) -> float:
        transition_info = self.get_transition_info(cause, effect)
        return transition_info.average_duration

    def update(self, cause: C, effect: E, duration: int) -> None:
        sub_dict = self.content.get(cause)
        if sub_dict is None:
            sub_dict = dict[E, TransitionInfo]()
            self.content[cause] = sub_dict

        transition_info = sub_dict.get(effect)
        if transition_info is None:
            transition_info = TransitionInfo()
            sub_dict[effect] = transition_info

        else:
            total_duration = transition_info.frequency * transition_info.average_duration + duration
            transition_info.frequency += 1
            transition_info.average_duration = total_duration / transition_info.frequency

    def predict(self, cause: C, default: E | None = None) -> E | None:
        sub_dict = self.content.get(cause)
        if sub_dict is None:
            return default

        return max(
            sub_dict,
            key=lambda effect: (
                sub_dict[effect].average_duration,
                sub_dict[effect].frequency)
        )

    def get_transition_info(self, cause: C, effect: E) -> TransitionInfo:
        sub_dict = self.content.get(cause)
        if sub_dict is None:
            return TransitionInfo(frequency=0, average_duration=0)

        transition_info = sub_dict.get(effect, TransitionInfo(frequency=0, average_duration=0))
        return transition_info


class SemioticModel[C: Hashable, E: Hashable]:
    type Shape = int

    @staticmethod
    def build[C, E](levels: Sequence[int], frozen: bool = True) -> SemioticModel[C, E]:
        base_model = SemioticModel[C, E](frozen=frozen)
        current_model = base_model

        for i, each_level in enumerate(levels):
            for each_predictor in range(each_level - 1):
                current_model._generate_predictor()

            if i < len(levels) - 1:
                each_model = SemioticModel[SemioticModel.Shape, SemioticModel.Shape](frozen=frozen)
                current_model.parent = each_model
                current_model = each_model

        return base_model

    @staticmethod
    def _check_expected(predictor: Representation[C, E, Shape], cause: C, effect_observed: E) -> bool:
        return predictor.match_strict(cause, effect_observed)

    @staticmethod
    def _check_best(predictor: Representation[C, E, Shape], cause: C, effect_observed: E) -> bool:
        return predictor.match_strict(cause, effect_observed)

    def __init__(self, level: int = 0, frozen: bool = True) -> None:
        if not frozen:
            raise NotImplementedError("Only frozen models are supported at this time.")

        self.level = level
        self.frozen = frozen

        self._pre_last_predictor_shape: SemioticModel.Shape | None = None
        self._last_predictor_shape: SemioticModel.Shape | None = None

        self._generated_predictors = 0
        self.predictors = dict[SemioticModel.Shape, Representation[C, E, SemioticModel.Shape]]()
        self.this_predictor = self._generate_predictor()

        self._duration = 0

        self.parent: SemioticModel[SemioticModel.Shape, SemioticModel.Shape] | None = None

    def _generate_predictor(self) -> Representation[C, E, Shape]:
        predictor = Representation[C, E, int](self._generated_predictors)
        self.predictors[predictor.shape] = predictor
        self._generated_predictors += 1
        return predictor

    def _breakdown(self, cause: C, effect_observed: E) -> bool:
        return not self.this_predictor.match_strict(cause, effect_observed)

    def _find_best_predictor(self, cause: C, effect: E) -> Representation[C, E, Shape]:
        def inverse_order_value(predictor: Representation[C, E, SemioticModel.Shape]) -> float:
            value = predictor.match_value(cause, effect)
            if 0. >= value:
                return value
            return 1. / value

        best_predictor = min(self.predictors.values(), key=inverse_order_value)
        return best_predictor

    def _handle_unexpected(self, cause: C, effect: E) -> Representation[C, E, Shape]:
        predictor_best = self._find_best_predictor(cause, effect)
        is_best = SemioticModel._check_best(predictor_best, cause, effect)
        if not is_best and not self.frozen:
            predictor_new = self._generate_predictor()
            return predictor_new

        return predictor_best

    def _handle_breakdown(self, cause: C, effect: E) -> None:
        is_expected = False
        predictor_next = self.this_predictor

        if self.parent is not None:
            predictor_expected_shape = self.parent.predict(
                self.this_predictor.shape,
                default=self.this_predictor.shape
            )
            predictor_next = self.predictors.get(predictor_expected_shape, self.this_predictor)
            is_expected = SemioticModel._check_expected(predictor_next, cause, effect)

        if self.parent is None or not is_expected:
            predictor_next = self._handle_unexpected(cause, effect)

        self._pre_last_predictor_shape = self._last_predictor_shape
        self._last_predictor_shape = self.this_predictor.shape
        self.this_predictor = predictor_next

        if self.parent is None:
            self.parent = SemioticModel[C, E](level=self.level+1, frozen=self.frozen)

        if self._pre_last_predictor_shape is not None:
            self.parent.update(self._pre_last_predictor_shape, self._last_predictor_shape, duration=self._duration)

        self._duration = 0

    def update(self, cause: C, effect: E, duration: int = 1) -> None:
        is_breakdown = self._breakdown(cause, effect)
        if is_breakdown:
            self._handle_breakdown(cause, effect)

        self._duration += 1
        self.this_predictor.update(cause, effect, duration)

    def predict(self, cause: C, default: E | None = None) -> E | None:
        return self.this_predictor.predict(cause, default=default)

    def parent_iter(self) -> Generator[SemioticModel[Shape, Shape], None, None]:
        yield self
        level = self.parent
        while level is not None:
            yield level
            level = level.parent

    def get_state(self) -> tuple[Shape, ...]:
        return tuple(each_level.this_predictor.shape for each_level in self.parent_iter())


def iterate_text() -> Generator[str, None, None]:
    with open("/home/mark/nas/data/text/lovecraft_namelesscity.txt", mode="r") as file:
        for line in file:
            for char in line.strip():
                yield char.lower()


def main() -> None:
    # model = SemioticModel[str, str](frozen=True)
    model = SemioticModel.build([5, 3], frozen=True)
    last_char = ""
    total = success = 0

    for i, char in enumerate(iterate_text()):
        if i % 1_000 == 0:
            accuracy = 0. if total < 1 else success / total
            print(f"Accuracy: {accuracy}")

        if len(last_char) >= 1:
            prediction = model.predict(last_char, default=last_char)
            is_correct = prediction == char
            model.update(last_char, char)

        else:
            is_correct = False

        success += is_correct
        total += 1

        last_char = char


if __name__ == "__main__":
    main()

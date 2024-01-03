import dataclasses
from typing import Hashable


@dataclasses.dataclass
class TransitionInfo:
    frequency: int = 1
    average_duration: float = 1


class Predictor[C: Hashable, E: Hashable, S: Hashable]:
    def __init__(self, shape: S) -> None:
        self.shape = shape
        self.transitions = dict[C, dict[E, TransitionInfo]]()

    def update(self, cause: C, effect: E, duration: int) -> None:
        sub_dict = self.transitions.get(cause)
        if sub_dict is None:
            sub_dict = dict[E, TransitionInfo]()
            self.transitions[cause] = sub_dict

        transition_info = sub_dict.get(effect)
        if transition_info is None:
            transition_info = TransitionInfo()
            sub_dict[effect] = transition_info

        else:
            total_duration = transition_info.frequency * transition_info.average_duration + duration
            transition_info.frequency += 1
            transition_info.average_duration = total_duration / transition_info.frequency

    def predict(self, cause: C, default: E | None = None) -> E | None:
        sub_dict = self.transitions.get(cause)
        if sub_dict is None:
            return default

        return max(
            sub_dict,
            key=lambda effect: (
                sub_dict[effect].average_duration,
                sub_dict[effect].frequency)
        )

    def get_transition_info(self, cause: C, effect: E) -> TransitionInfo:
        sub_dict = self.transitions.get(cause)
        if sub_dict is None:
            return TransitionInfo(frequency=0, average_duration=0)

        transition_info = sub_dict.get(effect, TransitionInfo(frequency=0, average_duration=0))
        return transition_info


class SemioticModel[C: Hashable, E: Hashable]:
    type Shape = int

    def __init__(self, level: int = 0, frozen: bool = True) -> None:
        if not frozen:
            raise NotImplementedError("Only frozen models are supported at this time.")

        self.level = level
        self.frozen = frozen

        self._pre_last_predictor_shape: SemioticModel.Shape | None = None
        self._last_predictor_shape: SemioticModel.Shape | None = None

        self.generated_predictors = 0
        self.this_predictor = self._generate_predictor()
        self.predictors = {self.generated_predictors: self.this_predictor}

        self._duration = 0

        self.parent: SemioticModel[SemioticModel.Shape, SemioticModel.Shape] | None = None

    def _generate_predictor(self) -> Predictor[C, E, Shape]:
        predictor = Predictor[C, E, int](self.generated_predictors)
        self.generated_predictors += 1
        return predictor

    def get_state(self) -> tuple[Shape, ...]:
        state = [self.this_predictor.shape]

        level = self.parent
        while level is not None:
            state.append(level.this_predictor.shape)
            level = level.parent

        return tuple(state)

    def predict(self, cause: C, default: E | None = None) -> E | None:
        return self.this_predictor.predict(cause, default=default)

    def _strict_check(self, predictor: Predictor[C, E, Shape], cause: C, effect_observed: E) -> bool:
        effect_expected = predictor.predict(cause)
        return effect_expected == effect_observed

    def _threshold_check(
            self,
            predictor: Predictor[C, E, Shape],
            cause: C, effect_observed: E,
            threshold: float) -> bool:
        transition_info = predictor.get_transition_info(cause, effect_observed)
        return transition_info.average_duration >= threshold

    def _breakdown(self, cause: C, effect_observed: E) -> bool:
        return not self._strict_check(self.this_predictor, cause, effect_observed)

    def _check_expected(self, predictor: Predictor[C, E, Shape], cause: C, effect_observed: E) -> bool:
        return self._strict_check(predictor, cause, effect_observed)

    def _check_best(self, predictor: Predictor[C, E, Shape], cause: C, effect_observed: E) -> bool:
        return self._strict_check(predictor, cause, effect_observed)

    def _find_best_predictor(self, cause: C, effect: E) -> Predictor[C, E, Shape]:
        best_predictor = max(
            self.predictors.values(),
            key=lambda p: (
                p.get_transition_info(cause, effect).average_duration,
                p.get_transition_info(cause, effect).frequency
            )
        )
        return best_predictor

    def _handle_unexpected(self, cause: C, effect: E) -> Predictor[C, E, Shape]:
        predictor_best = self._find_best_predictor(cause, effect)
        is_best = self._check_best(predictor_best, cause, effect)
        if not is_best and not self.frozen:
            predictor_new = self._generate_predictor()
            self.generated_predictors += 1
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
            is_expected = self._check_expected(predictor_next, cause, effect)

        if self.parent is None or not is_expected:
            predictor_next = self._handle_unexpected(cause, effect)

        self._pre_last_predictor_shape = self._last_predictor_shape
        self._last_predictor_shape = self.this_predictor.shape
        self.this_predictor = predictor_next

        if self.parent is None:
            self.parent = SemioticModel[C, E](level=self.level + 1, frozen=self.frozen)

        if self._pre_last_predictor_shape is not None:
            self.parent.update(self._pre_last_predictor_shape, self._last_predictor_shape, duration=self._duration)

        self._duration = 0

    def update(self, cause: C, effect: E, duration: int = 1) -> None:
        is_breakdown = self._breakdown(cause, effect)
        if is_breakdown:
            self._handle_breakdown(cause, effect)

        self._duration += 1
        self.this_predictor.update(cause, effect, duration)


def main() -> None:
    pass


if __name__ == "__main__":
    main()

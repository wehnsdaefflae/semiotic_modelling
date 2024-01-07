from __future__ import annotations

import dataclasses
from typing import Hashable


@dataclasses.dataclass
class TransitionInfo:
    frequency: int
    average_duration: float
    total_sub_frequencies: int


class Representation[C: Hashable, E: Hashable, S: Hashable]:
    def __init__(self, shape: S) -> None:
        self._shape = shape
        self.content = dict[C, dict[E, TransitionInfo]]()

    @property
    def shape(self) -> S:
        return self._shape

    def __len__(self) -> int:
        transitions = 0
        for effects_dict in self.content.values():
            for each_transition_info in effects_dict.values():
                transitions += each_transition_info.frequency

        return transitions

    def strict_fit(self, cause: C, effect: E) -> float:
        prediction = self.predict(cause)
        if prediction is None:
            return -1.

        return float(prediction == effect)

    def max_scaled_fit(self, cause: C, effect: E) -> float:
        effects = self.content.get(cause)
        if effects is None:
            return -1.

        max_value = max(x.total_sub_frequencies for x in effects.values())
        transition_info = self.get_transition_info(cause, effect)
        return transition_info.total_sub_frequencies / max_value

    def prop_scaled_fit(self, cause: C, effect: E) -> float:
        effects = self.content.get(cause)
        if effects is None:
            return -1.

        total_value = sum(x.total_sub_frequencies for x in effects.values())
        transition_info = self.get_transition_info(cause, effect)
        return transition_info.total_sub_frequencies / total_value

    def transition(self, cause: C, effect: E, duration: int) -> None:
        sub_dict = self.content.get(cause)
        if sub_dict is None:
            sub_dict = dict[E, TransitionInfo]()
            self.content[cause] = sub_dict

        transition_info = sub_dict.get(effect)
        if transition_info is None:
            transition_info = TransitionInfo(frequency=1, average_duration=duration, total_sub_frequencies=duration)
            sub_dict[effect] = transition_info

        else:
            total_duration = transition_info.frequency * transition_info.average_duration + duration
            transition_info.frequency += 1
            transition_info.average_duration = total_duration / transition_info.frequency
            transition_info.total_sub_frequencies += duration

    def predict(self, cause: C, default: E | None = None) -> E | None:
        sub_dict = self.content.get(cause)
        if sub_dict is None:
            return default

        key, value = max(sub_dict.items(), key=lambda item: item[1].total_sub_frequencies)
        return key

    def get_transition_info(self, cause: C, effect: E) -> TransitionInfo:
        sub_dict = self.content.get(cause)
        if sub_dict is None:
            return TransitionInfo(frequency=0, average_duration=0, total_sub_frequencies=0)

        transition_info = sub_dict.get(effect, TransitionInfo(frequency=0, average_duration=0, total_sub_frequencies=0))
        return transition_info

    def max_scaled_likelihood(self, other: Representation[C, E, S], open_world: bool = True) -> float:
        likelihood = 1.
        for each_cause, effects_dict in other.content.items():
            for each_effect in effects_dict:
                each_fit = self.max_scaled_fit(each_cause, each_effect)
                likelihood *= abs(each_fit) if open_world else max(0., each_fit)

        return likelihood

    def prop_scaled_likelihood(self, other: Representation[C, E, S], open_world: bool = True) -> float:
        likelihood = 1.
        for each_cause, effects_dict in other.content.items():
            for each_effect in effects_dict:
                each_fit = self.prop_scaled_fit(each_cause, each_effect)
                likelihood *= abs(each_fit) if open_world else max(0., each_fit)

        return likelihood

    def clear(self) -> None:
        self.content.clear()

    def update(self, other: Representation[C, E, S]) -> None:
        for each_cause, effects_dict in other.content.items():
            for each_effect, each_transition_info in effects_dict.items():
                self.transition(each_cause, each_effect, each_transition_info.frequency)

    def __add__(self, other: Representation[C, E, S]) -> Representation[C, E, S]:
        combined = Representation(shape=(self.shape, other.shape))
        combined.update(self)
        combined.update(other)
        return combined

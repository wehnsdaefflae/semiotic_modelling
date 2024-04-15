from __future__ import annotations
from typing import Generator, Hashable, Sequence

from representation import Representation


class SemioticModel[C: Hashable, E: Hashable]:
    @staticmethod
    def build(*, shape: Sequence[int], threshold: float, frozen: bool = False) -> SemioticModel:
        height = len(shape)
        base_model = SemioticModel[C, E](threshold=threshold, frozen=frozen)
        model = base_model
        for index_level in range(height - 1):
            each_size = shape[index_level]
            for _ in range(each_size - 1):
                model._new_context()

            model = model.parent

        each_size = shape[-1]
        for _ in range(each_size - 1):
            model._new_context()

        return base_model

    def __init__(self, *, threshold: float, frozen: bool = False) -> None:
        self.threshold = threshold
        self.frozen = frozen
        self.parent: SemioticModel[int, int] | None = None
        self.context: Representation = Representation[C, E, int](shape=0)
        self.all_contexts: dict[int, Representation] = {self.context.shape: self.context}

    def _parent_iter(self) -> Generator[SemioticModel[Hashable, Hashable], None, None]:
        yield self
        level = self.parent
        while level is not None:
            yield level
            level = level.parent

    @property
    def state(self) -> tuple[Hashable, ...]:
        return tuple(each_level.context.shape for each_level in self._parent_iter())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(each_level.all_contexts) for each_level in self._parent_iter())

    def predict(self, cause: C, default: E | None = None) -> E:
        prediction = self.context.predict(cause)
        if prediction is None:
            return default or cause

    def _get_best_context(self, cause: C, effect: E) -> tuple[Representation[C, E, int], float]:
        if self.parent is None:
            return self.context, self.context.prop_scaled_fit(cause, effect)

        best_probability = -1.
        best_context: Representation[C, E, int] | None = None
        parent_context = self.parent.context

        for each_context in self.all_contexts.values():
            observation_transition_probability = each_context.prop_scaled_fit(cause, effect)
            if observation_transition_probability < 0.:
                observation_transition_probability = 1. # 0.

            if self.context.shape == each_context.shape:
                state_transition_probability = 1.

            else:
                state_transition_probability = parent_context.prop_scaled_fit(self.context.shape, each_context.shape)
                if state_transition_probability < 0.:
                    state_transition_probability = 1. # 0.

            probability = observation_transition_probability * state_transition_probability
            if best_probability < probability:
                best_probability = probability
                best_context = each_context

        return best_context, best_probability

    def _new_context(self) -> Representation[C, E, int]:
        if self.parent is None:
            self.parent = SemioticModel[int, int](threshold=self.threshold, frozen=self.frozen)

        new_shape = len(self.all_contexts)
        new_context = Representation[C, E, int](shape=new_shape)
        self.all_contexts[new_shape] = new_context
        return new_context

    def transition(self, cause: C, effect: E) -> None:
        best_context, expectedness = self._get_best_context(cause, effect)

        if expectedness < self.threshold:
            if not self.frozen:
                best_context = self._new_context()

        elif best_context.shape == self.context.shape:
            self.context.transition(cause, effect, duration=1)
            return

        if self.parent is not None:
            self.parent.transition(self.context.shape, best_context.shape)

        self.context = best_context
        self.context.transition(cause, effect, duration=1)


def iterate_text() -> Generator[str, None, None]:
    with open("/home/mark/nas/data/text/lovecraft_namelesscity.txt", mode="r") as file:
        for each_line in file:
            for each_char in each_line.strip():
                yield each_char.lower()


def main() -> None:
    # model = SemioticModel.build(shape=[5, 3], threshold=.5, frozen=True)
    model = SemioticModel[str, str](threshold=.5, frozen=False)
    # model = SemioticModel.build([1], frozen=True)

    # todo:
    #  implement frozen
    #  implement build
    #  if better than single predictor, then back to graphs
    # max_scaled_fit or prop_scaled_fit?
    # open world or closed world?
    #   in current context
    #   in alternative contexts

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

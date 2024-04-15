from __future__ import annotations
from typing import Generator, Hashable

from representation import Representation


class SemioticModel[C: Hashable, E: Hashable]:
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
        best_probability = -1.
        best_context: Representation[C, E, int] | None = None
        parent_context = self.parent.context

        for each_context in self.all_contexts.values():
            observation_transition_probability = each_context.max_scaled_fit(cause, effect)
            if observation_transition_probability < 0.:
                observation_transition_probability = 1.
                # observation_transition_probability = 0.

            state_transition_probability = parent_context.max_scaled_fit(self.context.shape, each_context.shape)
            if state_transition_probability < 0.:
                state_transition_probability = 1.
                # state_transition_probability = 0.

            probability = observation_transition_probability * state_transition_probability
            if best_probability < probability:
                best_probability = probability
                best_context = each_context

        return best_context, best_probability

    def _new_context(self, cause: C, effect: E) -> None:
        new_shape = len(self.all_contexts)
        new_context = Representation[C, E, int](shape=new_shape)
        new_context.transition(cause, effect, duration=1)
        self.all_contexts[new_shape] = new_context
        self.parent.transition(self.context.shape, new_context.shape)
        self.context = new_context

    def transition(self, cause: C, effect: E) -> None:
        expectedness = self.context.max_scaled_fit(cause, effect)
        if expectedness < 0. or expectedness >= self.threshold:
            self.context.transition(cause, effect, duration=1)
            return

        if self.parent is None and not self.frozen:
            self.parent = SemioticModel[int, int](threshold=self.threshold)
            self._new_context(cause, effect)

        else:
            new_context, new_expectedness = self._get_best_context(cause, effect)
            # breakpoint()
            if self.frozen or new_expectedness >= self.threshold:
                self.parent.transition(self.context.shape, new_context.shape)
                self.context = new_context
                return

            self._new_context(cause, effect)


def iterate_text() -> Generator[str, None, None]:
    with open("/home/mark/nas/data/text/lovecraft_namelesscity.txt", mode="r") as file:
        for each_line in file:
            for each_char in each_line.strip():
                yield each_char.lower()


def main() -> None:
    # model = SemioticModel.build([5, 3], frozen=True)
    model = SemioticModel[str, str](threshold=.01, frozen=False)
    # model = SemioticModel.build([1], frozen=True)

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

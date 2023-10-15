from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Sequence, Callable


# https://chat.openai.com/share/70c05b7e-92fe-40d5-be45-85c103a1264d
# https://chat.openai.com/share/06f31c59-8e43-4ab3-8c23-a522f1b6b131

class TransitionTable:
    def __init__(self, smoothing: float = 0.):
        self.smoothing = smoothing
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.total_counts = defaultdict(float)

    def __str__(self) -> str:
        lines = list()
        for prev_symbol, next_symbols in self.transitions.items():
            for next_symbol, count in next_symbols.items():
                probability = self.probability(prev_symbol, next_symbol)
                lines.append(f"{prev_symbol} -> {next_symbol} ({probability:.2f})")
        return "\n".join(lines)

    def update(self, prev_symbol: str, next_symbol: str, weight: float = 1.) -> None:
        self.transitions[prev_symbol][next_symbol] += weight
        self.total_counts[prev_symbol] += weight

    def probability(self, prev_symbol: str, next_symbol: str) -> float:
        num_symbols = len(self.transitions[prev_symbol])
        numerator = self.transitions[prev_symbol].get(next_symbol, 0.) + self.smoothing
        denominator = self.total_counts[prev_symbol] + num_symbols * self.smoothing

        return numerator / denominator

    def likelihood(self, segment: Sequence[str], bias: Callable[[float], float] | None = None) -> float:
        likelihood = 1.
        for i in range(len(segment) - 1):
            likelihood *= self.probability(segment[i], segment[i + 1])

        if bias:
            return bias(likelihood)

        return likelihood

    def predict_next_symbol(self, prev_symbol: str) -> tuple[str, float]:
        if self.total_counts[prev_symbol] == 0:
            return prev_symbol, 0

        symbol, count = max(self.transitions[prev_symbol].items(), key=lambda x: x[1])
        probability = self.probability(prev_symbol, symbol)
        return symbol, probability


class SequenceModel:
    def __init__(self, history_length: int = 5, threshold: float = .1) -> None:
        self.history_length = history_length
        self.threshold = threshold

        self.tables = [TransitionTable()]
        self.current_table_indices = defaultdict(int)
        self.histories = defaultdict(lambda: deque(maxlen=history_length))

        self.next_layer = None

    def update(self, sequence_id: int, symbol: str) -> None:
        history = self.histories[sequence_id]

        table_index = self.current_table_indices[sequence_id]
        table = self.tables[table_index]

        if len(history) >= 1:
            if len(history) >= self.history_length:

                if table.likelihood(history) < self.threshold and self.next_layer:
                    next_table_index_str, _ = self.next_layer.predict_next_symbol(sequence_id, str(table_index))
                    table_index = int(next_table_index_str)
                    table = self.tables[table_index]

                    if table.likelihood(history, bias=math.sqrt) < self.threshold:
                        table_index, table = max(enumerate(self.tables), key=lambda _i, _x: _x.likelihood(history))

                        if table.likelihood(history) < self.threshold:
                            table_index = len(self.tables)
                            table = TransitionTable()
                            self.tables.append(table)
                            self.next_layer = self.next_layer or SequenceModel()

                    self.current_table_indices[sequence_id] = table_index

                if self.next_layer:
                    self.next_layer.update(sequence_id, str(table_index))

            predicted_symbol, max_prob = table.predict_next_symbol(history[-1])
            weight = 1. if predicted_symbol == symbol else .5
            table.update(history[-1], symbol, weight)

        history.append(symbol)

    def predict_next_symbol(self, sequence_id: int, symbol: str) -> tuple[str, float]:
        table_index = self.current_table_indices[sequence_id]
        table = self.tables[table_index]
        best_symbol, best_prob = table.predict_next_symbol(symbol)
        return best_symbol, best_prob

    def get_model_state(self, sequence_id: int) -> list[int]:
        if sequence_id not in self.current_table_indices:
            return list()
        state = [self.current_table_indices[sequence_id]]
        if self.next_layer:
            state.extend(self.next_layer.get_model_state(sequence_id))
        return state


def main() -> None:
    sequences = [
        "To create a main method that iterates through two sequences of natural text in parallel,",
        "alternating between the letters of each sequence, you can use the following approach:"
        ]

    sequence_model = SequenceModel()
    predictions = tuple(list() for _ in sequences)
    precision = dict()

    for symbols in zip(*sequences):
        for sequence_index, each_symbol in enumerate(symbols):
            each_prediction = predictions[sequence_index]
            if len(each_prediction) >= 1:
                predicted_symbol = each_prediction[-1]
                precision[sequence_index] = precision.get(sequence_index, 0) + int(predicted_symbol == each_symbol)

            sequence_model.update(sequence_index, each_symbol)
            predicted_symbol, _ = sequence_model.predict_next_symbol(sequence_index, each_symbol)

            each_prediction += predicted_symbol

    print()

    for sequence_index in range(len(sequences)):
        print(sequence_model.get_model_state(sequence_index))
        print(f"precision sequence {sequence_index}: {precision[sequence_index] / len(sequences[sequence_index]):.2f}")
        print("".join(predictions[sequence_index]))

    print()


if __name__ == "__main__":
    main()

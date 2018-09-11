# coding=utf-8
from typing import Tuple, Sequence, Iterator, TypeVar

INPUT = TypeVar("INPUT")
TARGET = TypeVar("TARGET")

SEQUENCE_OF_CONCURRENT_EXAMPLES = Iterator[Tuple[Tuple[INPUT, ...], Tuple[TARGET, ...]]]
RATIONAL_SEQUENCE = Iterator[float]

INPUT_SEQUENCES = Sequence[RATIONAL_SEQUENCE]
TARGET_SEQUENCES = Sequence[RATIONAL_SEQUENCE]

EXAMPLE_SEQUENCE = Tuple[INPUT_SEQUENCES, TARGET_SEQUENCES]


def from_sequences(sequences: Sequence[EXAMPLE_SEQUENCE],
                   in_frame_size: int = 1,
                   frame_start_offset: int = 0,
                   out_frame_size: int = 1) -> SEQUENCE_OF_CONCURRENT_EXAMPLES[float, float]:

    all_sequences = [id(_seq) for _ex in sequences for _io in _ex for _seq in _io]
    if not len(all_sequences) == len(set(all_sequences)):
        raise ValueError("All sequences must be individual iterator instances.")

    # initialize input and target frames
    in_frames = tuple([] for _ in sequences)
    out_frames = tuple([] for _ in sequences)

    # progress targets by defined offset
    for _, target_sequences in sequences:
        for _ in range(frame_start_offset):
            for _seq in target_sequences:
                next(_seq)

    while True:
        for _i, (input_sequences, target_sequences) in enumerate(sequences):
            each_in_frame = in_frames[_i]
            # add new
            for _ in range(in_frame_size - len(each_in_frame)):
                new_input = tuple(next(_seq) for _seq in input_sequences)
                if len(new_input) < 1:
                    raise StopIteration()
                each_in_frame.append(new_input)

            each_out_frame = out_frames[_i]
            # add new
            for _ in range(out_frame_size - len(each_out_frame)):
                new_target = tuple(next(_seq) for _seq in target_sequences)
                if len(new_target) < 1:
                    raise StopIteration()
                each_out_frame.append(new_target)

        yield tuple(in_frames), tuple(out_frames)

        for each_in_frame, each_out_frame in zip(in_frames, out_frames):
            # remove old
            for _ in range(len(each_in_frame) - in_frame_size + 1):
                each_in_frame.pop(0)

            # remove old
            for _ in range(len(each_out_frame) - out_frame_size + 1):
                each_out_frame.pop(0)


if __name__ == "__main__":
    sequence_a = (_x for _x in range(0, 10))
    sequence_b = (_x for _x in range(0, 10))
    sequence_c = (_x for _x in range(0, 10))

    s = [([sequence_a], [sequence_b])]

    g = from_sequences(s)
    for _v in g:
        print(_v)
        pass

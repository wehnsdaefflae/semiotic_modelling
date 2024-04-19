def main() -> None:
    """
    class Segment[O: Hashable]:
        def __init__(self, segment_id: Hashable):
            self._segment_id = segment_id
            self._graph = nx.DiGraph()
            # transitions must be integers

        def _current_threshold(self) -> float:
            # c / log(1 + n), n: total frequencies
            return 0.

        def train(self, cause: O, effect: O) -> None:
            edge increment
            pass

        def get_probability(self, cause: O, effect: O) -> float:
            return 0.

        def get_ipt(self, cause: O, effect: O) -> float:
            # Interconnected Transition Probability
            # frequency(cause, effect) / (all_outgoing(cause) + all_incoming(effect))
            return 0.

        def predict(self, cause: O) -> O:
            return cause

        def get_graph_segments_and_edges(self) -> tuple[tuple[Segment[O], ...], dict[tuple[Hashable, Hashable], set[tuple[O, O, int]]]]:
            # return segments as well as the transitions between segment elements below the threshold
            return tuple()


    class Layer[O: Hashable]:
        def __init__(self):
            self._parent: Layer[int] | None = None
            self._segments: dict[int, Segment[O]] = {}
            self._current_segment = Segment[O](0)
            self._parent: Layer[int] | None = None

        def _new_segment(self) -> Segment[O]:
            return Segment[O](len(self._segments))

        def _parent_probability(self, cause_shape: int, effect_shape: int) -> float:
            parent_segment = self._parent._current_segment
            each_transition = parent_segment._graph[self._current_segment._segment_id][effect_shape]
            frequency = sum(each_transition['frequency'].values())

            all_frequencies = 0
            for each_shape in parent_segment._graph.successors(cause_shape):
                each_transition = parent_segment._graph[self._current_segment._segment_id][each_shape]
                frequencies = sum(each_transition['frequency'].values())
                all_frequencies += frequencies

            return frequency / all_frequencies


        def _find_segment(self, cause: O, effect: O) -> Segment[O]:
            if self._parent is None:
                return self._current_segment

            current_probability = self._current_segment.get_probability(cause, effect)
            parent_segment = self._parent._current_segment

            # next probability
            highest_frequency = 0
            best_next_shape = None
            all_frequencies = 0
            for each_shape in parent_segment._graph.successors(self._current_segment._segment_id):
                each_parent_segment = self._parent._segments[each_shape]
                each_transition = parent_segment._graph[self._current_segment._segment_id][each_shape]
                each_frequency = each_transition['frequency']
                all_frequencies += each_frequency
                if highest_frequency < each_frequency:
                    highest_frequency = each_frequency
                    best_next_shape = each_shape
            next_probability = highest_frequency / all_frequencies

            # any probability
            best_probability = .0
            best_shape = None
            for each_shape in parent_segment._graph.successors(self._current_segment._segment_id):
                each_parent_probability = self._parent_probability(self._current_segment._segment_id, each_shape)
                each_observation_probability = each_segment.get_probability(cause, effect)
                each_probability = each_parent_probability * each_observation_probability
                if best_probability < each_probability:
                    best_probability = each_probability
                    best_shape = each_shape

            if best_probability < next_probability:
                return self._segments[best_next_shape]

            return self._segments[best_shape]

        def train(self, cause: O, effect: O) -> None:
            self._current_segment.train(cause, effect)
            segments, edges = self._current_segment.get_graph_segments_and_edges()
            if len(segments) < 2 and segments[0] is self._current_segment:
                return

            self._replace_segment(self._current_segment, segments, edges)

        def self._replace(self, segment: Segment[O], segments: tuple[Segment[O], ...], edges: dict[tuple[Hashable, Hashable], set[tuple[O, O, int]]]) -> None:
            ### todo

        def predict(self, observation: O) -> O:
            return self._current_segment.predict(observation)


    model = Layer[str]()
    expectation = None
    last_observation = None
    for each_observation in stream:
        model.train(last_observation, each_observation)

        new_cause = each_observation  # add action if interactive
        expectation = model.predict(new_cause)
        last_observation = new_cause
    """


if __name__ == "__main__":
    main()

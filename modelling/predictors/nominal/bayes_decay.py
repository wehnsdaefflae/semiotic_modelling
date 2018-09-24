# coding=utf-8
import random
from typing import Hashable, Tuple, Sequence

import networkx
from matplotlib import pyplot
from networkx_viewer import Viewer

from modelling.predictors.abstract_predictor import Predictor, INPUT_TYPE, OUTPUT_TYPE

NOMINAL_INPUT = Hashable
NOMINAL_OUTPUT = Hashable


class BayesDecayModel(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def fit(self, examples: Sequence[Tuple[INPUT_TYPE, OUTPUT_TYPE]]):
        pass

    def save(self, file_path):
        pass

    def predict(self, input_values: Sequence[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE, ...]:
        pass

    def get_structure(self) -> Tuple[int, ...]:
        pass

    def get_state(self) -> Hashable:
        pass


if __name__ == "__main__":
    G = networkx.petersen_graph()
    pyplot.subplot(121)
    networkx.draw(G, with_labels=True, font_weight='bold')
    pyplot.subplot(122)
    networkx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    pyplot.show()

    #app = Viewer(petersen)
    #app.mainloop()

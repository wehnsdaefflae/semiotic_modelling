# coding=utf-8
import time
from math import sqrt
from typing import Tuple

from modelling.predictors.abstract_predictor import Predictor
from tools.logger import Logger
from tools.timer import Timer
from visualization.visualization import Visualize


class SetupPrediction:
    def __init__(self, predictor: Predictor, stream_train, stream_test, max_iterations: int, log_steps: int = 1):
        self.predictor = predictor
        self.stream_train = stream_train
        self.stream_test = stream_test
        self.log_steps = log_steps

        self.average_train_error = 0.
        self.average_test_error = 0.
        self.average_duration = 0.

        self.iterations = 0
        self.max_iterations = max_iterations

        self.last_data_keys = []

    def __log_data__(self, data):
        if self.last_data_keys != data.keys():
            self.last_data_keys = sorted(data.keys())
        Logger.log_to(self.predictor.__class__.__name__, *tuple(data[_x] for _x in self.last_data_keys), directory_path="data_log/", header=self.last_data_keys)

    def __step__(self):
        examples_train = next(self.stream_train)
        inputs_train, targets_train = zip(*examples_train)

        examples_test = next(self.stream_test)
        inputs_test, targets_test = zip(*examples_test)

        # perform prediction and fit
        this_time = time.time()
        outputs_train = self.predictor.predict(inputs_train)
        outputs_test = self.predictor.predict(inputs_test)

        self.predictor.fit(examples_train)

        duration = time.time() - this_time

        data = dict()
        data["outputs_train"] = outputs_train
        data["outputs_test"] = outputs_test
        data["duration"] = duration

        self.iterations += 1

        if 0 < self.log_steps and self.iterations % self.log_steps == 0:
            self.__log_data__(data)

    def __iter__(self):
        if 0 < self.max_iterations <= self.iterations:
            raise StopIteration()

        self.__step__()


def setup(predictor: Predictor, train_generator, test_generator, visualization_steps: int, iterations: int = 500000):
    print("Starting experiment with {:s} for {:d} iterations...".format(predictor.name(), iterations))

    average_train_error = 0.
    average_test_error = 0.
    average_duration = 0.

    # exchange rate adaptation
    # error_list = []

    for t in range(iterations):
        # get concurrent examples
        examples_train = next(train_generator)
        inputs_train, targets_train = zip(*examples_train)

        examples_test = next(test_generator)
        inputs_test, targets_test = zip(*examples_test)

        # perform prediction and fit
        this_time = time.time()
        outputs_train = predictor.predict(inputs_train)
        outputs_test = predictor.predict(inputs_test)

        predictor.fit(examples_train)

        duration = time.time() - this_time

        # todo: continue from here

        # update plot
        try:
            train_error = sum(sqrt(sum((__o - __t) ** 2 for __o, __t in zip(_o, _t))) for _o, _t in zip(outputs_train, targets_train)) / len(targets_train)
        except TypeError:
            train_error = sum(float(_o != _t) for _o, _t in zip(outputs_train, targets_train)) / len(targets_train)

        try:
            test_error = sum(sqrt(sum((__o - __t) ** 2 for __o, __t in zip(_o, _t))) for _o, _t in zip(outputs_test, targets_test)) / len(targets_test)
        except TypeError:
            test_error = sum(float(_o != _t) for _o, _t in zip(outputs_test, targets_test)) / len(targets_test)

        # exchange rate adaptation
        # if .5 < concurrent_outputs[0][0]:
        #     error_list.append(error)

        average_train_error = (average_train_error * t + train_error) / (t + 1)
        average_test_error = (average_test_error * t + test_error) / (t + 1)

        average_duration = (average_duration * t + duration) / (t + 1)
        if (t + 1) % visualization_steps == 0:
            # exchange rate adaptation
            Visualize.append("error train", predictor.__class__.__name__, average_train_error)
            Visualize.append("error test", predictor.__class__.__name__, average_test_error)

            Visualize.append("duration", predictor.__class__.__name__, average_duration)

            try:
                for _e, (each_train_output, each_train_target) in enumerate(zip(outputs_train, targets_train)):
                    for _o, (train_output_value, train_target_value) in enumerate(zip(each_train_output, each_train_target)):
                        axis_key = f"output train {_o:02d}/{_e:02d}"
                        Visualize.append(axis_key, predictor.__class__.__name__, train_output_value)
                        Visualize.append(axis_key, "target train", train_target_value)

            except TypeError:
                pass

            try:
                for _e, (each_test_output, each_test_target) in enumerate(zip(outputs_test, targets_test)):
                    for _o, (test_output_value, test_target_value) in enumerate(zip(each_test_output, each_test_target)):
                        axis_key = f"output test {_o:02d}/{_e:02d}"
                        Visualize.append(axis_key, predictor.__class__.__name__, test_output_value)
                        Visualize.append(axis_key, "target test", test_target_value)

            except TypeError:
                pass

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    Visualize.finalize("error train", predictor.__class__.__name__)
    Visualize.finalize("error test", predictor.__class__.__name__)
    Visualize.finalize("duration", predictor.__class__.__name__)
    # todo: finalize outputs?

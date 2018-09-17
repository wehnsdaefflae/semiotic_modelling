# coding=utf-8
import datetime
import os
import random
import sys
import time
from math import sin, cos, tan
from typing import TypeVar, Generic, Tuple, Collection, Dict, List

from matplotlib import pyplot, axes
from matplotlib.artist import Artist

OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Logger:
    _time = datetime.datetime.now()
    _file_path = sys.argv[0]
    _base_name = os.path.basename(_file_path)
    _first_name = os.path.splitext(_base_name)[0]
    _time_str = _time.strftime("%Y-%m-%d_%H-%M-%S")
    log_name = _first_name + _time_str + ".log"

    @staticmethod
    def log(message: str):
        print(message)
        with open(Logger.log_name, mode="a") as file:
            file.write(message + "\n")


class VisualizeSingle:
    fig = None
    plot_axes = dict()                  # type: Dict[str, axes]
    labels_axes_to_plots = dict()       # type: Dict[str, Collection[str]]

    current_series = dict()             # type: Dict[str, Dict[str, List[float]]]
    average_series = dict()             # type: Dict[str, Dict[str, List[float]]]

    average_plots = dict()              # type: Dict[str, Dict[str, Artist]]
    iteration = 0

    @staticmethod
    def initialize(labels_axes_to_plots: Dict[str, Collection[str]], title: str):
        VisualizeSingle.fig, plot_axes = pyplot.subplots(len(labels_axes_to_plots), sharex="all")
        VisualizeSingle.fig.suptitle(title)

        VisualizeSingle.current_series.clear()
        VisualizeSingle.average_series.clear()
        VisualizeSingle.labels_axes_to_plots.clear()

        VisualizeSingle.labels_axes_to_plots.update(labels_axes_to_plots)

        for (_axis_name, _plot_names), _ax in zip(labels_axes_to_plots.items(), plot_axes):
            _ax.set_ylabel(_axis_name)

            VisualizeSingle.plot_axes[_axis_name] = _ax

            axis_series = {_each_plot_name: [] for _each_plot_name in _plot_names}
            VisualizeSingle.current_series[_axis_name] = axis_series

            average_series = {_each_plot_name: [] for _each_plot_name in _plot_names}
            VisualizeSingle.average_series[_axis_name] = average_series

            VisualizeSingle.average_plots[_axis_name] = dict()

    @staticmethod
    def update(axis_label: str, plot_label: str, value: float):
        try:
            axis = VisualizeSingle.current_series[axis_label]
        except KeyError:
            raise ValueError(f"no axis called '{axis_label}'.")

        try:
            series = axis[plot_label]
        except KeyError:
            raise ValueError(f"no plot called '{plot_label}' in axis '{axis_label}'.")

        series.append(value)

    @staticmethod
    def plot():
        for each_axis_name in VisualizeSingle.labels_axes_to_plots:
            each_axis = VisualizeSingle.plot_axes[each_axis_name]
            each_series = VisualizeSingle.current_series[each_axis_name]
            each_average = VisualizeSingle.average_series[each_axis_name]
            each_average_plot = VisualizeSingle.average_plots[each_axis_name]

            for _i, each_plot_name in enumerate(VisualizeSingle.labels_axes_to_plots[each_axis_name]):
                _series = each_series[each_plot_name]
                if 0 < VisualizeSingle.iteration:
                    _plot = each_average_plot[each_plot_name]
                    _plot.remove()
                    _average = [(_a * VisualizeSingle.iteration + _s) / (VisualizeSingle.iteration + 1) for _a, _s in zip(each_average[each_plot_name], _series)]

                else:
                    _average = _series

                each_axis.plot(_series, color=f"C{_i%10}", alpha=.5, label=each_plot_name)
                each_average[each_plot_name] = _average
                each_average_plot[each_plot_name],  = each_axis.plot(_average, color="black", label="average " + each_plot_name)

                _series.clear()

            if VisualizeSingle.iteration < 1:
                each_axis.legend()

        VisualizeSingle.iteration += 1

        pyplot.draw()
        pyplot.pause(.001)

    @staticmethod
    def finish():
        pyplot.show()


class Visualization(Generic[OUTPUT_TYPE]):
    def __init__(self, title: str):
        self.title = title
        self.iteration = 0
        self.time = []

    def _update(self, reward: float, output: OUTPUT_TYPE, target: OUTPUT_TYPE, error: float, duration: float, structure: Tuple[int, ...]):
        raise NotImplementedError()

    def update(self, reward: float, output: OUTPUT_TYPE, target: OUTPUT_TYPE, error: float, duration: float, structure: Tuple[int, ...]):
        self._update(reward, output, target, error, duration, structure)
        self.iteration += 1


class VisualizationPyplot(Visualization[OUTPUT_TYPE]):
    def __init__(self, title: str, accumulation_steps: int):
        super().__init__(title)
        self.fig, axes = pyplot.subplots(5, sharex="all")
        self.fig.suptitle(title)
        self.axis_reward, self.axis_out, self.axis_error, self.axis_structure, self.axis_duration = axes

        self.no_runs = 0
        self.average_reward_dict = [[], None]
        self.average_error_dict = [[], None]
        self.average_duration_dict = [[], None]

        self.accumulation_steps = accumulation_steps

        self.values_reward = []
        self.values_output = []
        self.values_target = []
        self.values_error = []
        self.values_structure = []
        self.values_duration = []

        self.average_error = 1.
        self.average_duration = 0.
        self.average_reward = 0.

        self.colors = dict()

    def _update(self, reward: float, output: OUTPUT_TYPE, target: OUTPUT_TYPE, error: float, duration: float, structure: Tuple[int, ...]):
        self.average_reward = (self.average_reward * self.iteration + reward) / (self.iteration + 1)
        self.average_error = (self.average_error * self.iteration + error) / (self.iteration + 1)
        self.average_duration = (self.average_duration * self.iteration + duration) / (self.iteration + 1)

        if (self.iteration + 1) % self.accumulation_steps == 0:
            self.values_reward.append(self.average_reward)
            self.values_error.append(self.average_error)
            self.values_duration.append(self.average_duration)

            self.values_output.append(output)
            self.values_target.append(target)

            self.time.append(self.iteration)

    def average(self):
        raise NotImplementedError()

    def _clear(self):
        self.iteration = 0
        self.average_error = 1.
        self.average_duration = 0.
        self.average_reward = 0.

        self.time.clear()

        self.values_reward.clear()
        self.values_output.clear()
        self.values_target.clear()
        self.values_error.clear()
        self.values_structure.clear()
        self.values_duration.clear()

        self.no_runs = 0
        self.average_reward_dict = [[], None]
        self.average_error_dict = [[], None]
        self.average_duration_dict = [[], None]

    def show(self, name: str, legend: bool = True):
        color = self.colors.get(name)
        if color is None:
            color = "C{:d}".format(len(self.colors))
            self.colors[name] = color

        self.axis_reward.plot(self.time, self.values_reward, alpha=.5, color=color, label=name)
        self.axis_reward.set_ylabel("reward")
        if legend:
            self.axis_reward.legend()

        #self.axis_out.plot(self.time_output, self.values_output, label="output")
        #self.axis_out.plot(self.time_target, self.values_target, label="target")
        self.axis_out.set_ylabel("output / target")
        if legend:
            self.axis_out.legend()

        self.axis_error.plot(self.time, self.values_error, alpha=.5, color=color, label=name)
        self.axis_error.set_ylabel("error")
        self.axis_error.set_ylim([0., 1.])
        if legend:
            self.axis_error.legend()

        # self.axis_structure.plot
        self.axis_structure.set_ylabel("structure")
        if legend:
            self.axis_structure.legend()

        self.axis_duration.plot(self.time, self.values_duration, alpha=.5, color=color, label=name)
        self.axis_duration.set_ylabel("duration (ms)")
        if legend:
            self.axis_duration.legend()

        if self.no_runs < 1:
            self.average_reward_dict[0] = self.values_reward[:]
            self.average_error_dict[0] = self.values_error[:]
            self.average_duration_dict[0] = self.values_duration[:]

        else:
            self.average_reward_dict[0] = [(_a * self.no_runs + _v) / (self.no_runs + 1) for _a, _v in zip(self.average_reward_dict[0], self.values_reward)]
            self.average_error_dict[0] = [(_a * self.no_runs + _v) / (self.no_runs + 1) for _a, _v in zip(self.average_error_dict[0], self.values_error)]
            self.average_duration_dict[0] = [(_a * self.no_runs + _v) / (self.no_runs + 1) for _a, _v in zip(self.average_duration_dict[0], self.values_duration)]

            self.average_reward_dict[1].remove()
            self.average_error_dict[1].remove()
            self.average_duration_dict[1].remove()

        self.average_reward_dict[1], = self.axis_reward.plot(self.time, self.average_reward_dict[0], color="black")
        self.average_error_dict[1], = self.axis_error.plot(self.time, self.average_error_dict[0], color="black")
        self.average_duration_dict[1], = self.axis_duration.plot(self.time, self.average_duration_dict[0], color="black")

        self.no_runs += 1

        pyplot.draw()
        pyplot.pause(.001)

        self._clear()

    def finish(self):
        pyplot.show()


if __name__ == "__main__":
    labels = {"axis0": {"plot0", "plot1"}, "axis1": {"plot2"}}
    VisualizeSingle.initialize(labels, "test figure")

    f0 = lambda _x: sin(_x / 100.) * random.uniform(.5, 2.)
    f1 = lambda _x: cos(_x / 100.) * random.uniform(.5, 2.)
    f2 = lambda _x: tan(_x / 100.) * random.uniform(.5, 2.)

    for _ in range(100):
        for x in range(1000):
            VisualizeSingle.update("axis0", "plot0", f0(x))
            VisualizeSingle.update("axis0", "plot1", f1(x))
            VisualizeSingle.update("axis1", "plot2", f2(x))
        VisualizeSingle.plot()
        # time.sleep(1.)
    VisualizeSingle.finish()

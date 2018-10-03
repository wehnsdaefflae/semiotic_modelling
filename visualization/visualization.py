# coding=utf-8
import random
import time
from math import sin, cos, tan
from typing import TypeVar, Generic, Tuple, Collection, Dict, List, Sequence, Any

from matplotlib import pyplot, axes, animation
from matplotlib.artist import Artist
from matplotlib.colors import hsv_to_rgb

from tools.math_functions import distribute_circular

OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Visualize:
    _refresh_rate = 0
    _designators = None
    _figure = None

    _axes = dict()
    _plot_lines = dict()
    _progress_lines = dict()

    _current_series = None
    _average_series = None

    _finished_iterations = dict()

    _plot_legend = set()

    @ staticmethod
    def init(title: str, designators: Dict[str, Collection[str]], x_range: int = 0, refresh_rate: int = 1000):
        Visualize._figure, all_axes = pyplot.subplots(len(designators), sharex="all")
        Visualize._figure.suptitle(title)

        if 0 < x_range:
            for each_axis in all_axes:
                each_axis.set_xlim(left=0, right=x_range)

        Visualize._axes = {axis_name: _axis for axis_name, _axis in zip(designators, all_axes)}
        for _axis_label, _axis in Visualize._axes.items():
            _axis.set_ylabel(_axis_label)

        Visualize._refresh_rate = refresh_rate
        Visualize._current_series = {_axis: {_plot: [] for _plot in plot_names} for _axis, plot_names in designators.items()}
        Visualize._average_series = {_axis: {_plot: [] for _plot in plot_names} for _axis, plot_names in designators.items()}

        Visualize._designators = {axis_name: sorted(plot_names) for axis_name, plot_names in designators.items()}

    @staticmethod
    def _get_series(axis_name: str, plot_name: str, average: bool) -> List[float]:
        sub_dict = Visualize._average_series.get(axis_name) if average else Visualize._current_series.get(axis_name)
        if sub_dict is None:
            raise ValueError(f"No axis '{axis_name}' defined.")

        series = sub_dict.get(plot_name)
        if series is None:
            raise ValueError(f"No plot '{plot_name}' defined on axis '{axis_name}'.")

        return series

    @staticmethod
    def _iteration_increment(axis_name: str, plot_name: str, by: int = 0) -> int:
        sub_dict = Visualize._finished_iterations.get(axis_name)
        if sub_dict is None:
            sub_dict = {plot_name: by}
            Visualize._finished_iterations[axis_name] = sub_dict
            return by

        iterations = sub_dict.get(plot_name, 0)

        if by < 1:
            return iterations

        new_value = iterations + by
        sub_dict[plot_name] = new_value

        return new_value

    @staticmethod
    def _update_plot(axis_name: str, plot_name: str):
        axis = Visualize._axes.get(axis_name)
        if axis is None:
            raise ValueError(f"No axis for name '{axis_name}'.")

        plot_names = Visualize._designators.get(axis_name)
        if plot_names is None:
            raise ValueError(f"No plot names for axis '{axis_name}'.")

        key_string = axis_name + "_" + plot_name
        plot_index = plot_names.index(plot_name)
        hue_value = distribute_circular(plot_index)

        current_key = key_string + "_current"
        plot_line = Visualize._plot_lines.get(current_key)
        series = Visualize._get_series(axis_name, plot_name, False)
        color_soft = hsv_to_rgb((hue_value, .5, .8))
        if plot_line is not None:
            plot_line.remove()
        Visualize._plot_lines[current_key], = axis.plot(series, color=color_soft, alpha=.2)

        average_key = key_string + "_average"
        plot_line = Visualize._plot_lines.get(average_key)
        average = Visualize._get_series(axis_name, plot_name, True)
        color_hard = hsv_to_rgb((hue_value, .7, .5))
        if plot_line is not None:
            plot_line.remove()
        Visualize._plot_lines[average_key], = axis.plot(average, color=color_hard, label=plot_name)

        legend_key = axis_name + plot_name
        if legend_key not in Visualize._plot_legend:
            axis.legend()
            Visualize._plot_legend.add(legend_key)

        previous_progress = Visualize._progress_lines.get(key_string)
        if previous_progress is not None:
            previous_progress.remove()
        Visualize._progress_lines[key_string] = axis.axvline(x=len(series), color=color_hard)

        pyplot.draw()
        pyplot.pause(.001)

    @staticmethod
    def append(axis_name: str, plot_name: str, value: float):
        try:
            series = Visualize._get_series(axis_name, plot_name, False)
        except ValueError:
            return

        index = len(series)
        series.append(value)

        average = Visualize._get_series(axis_name, plot_name, True)
        iterations = Visualize._iteration_increment(axis_name, plot_name)
        if iterations < 1:
            average.append(value)
        else:
            average[index] = (average[index] * iterations + value) / (iterations + 1)

        if (0 < Visualize._refresh_rate) and (len(series) % Visualize._refresh_rate == 0):
            Visualize._update_plot(axis_name, plot_name)

    @staticmethod
    def finalize_all():
        for each_axis, plot_names in Visualize._designators.items():
            for _plot in plot_names:
                Visualize.finalize(each_axis, _plot)

    @staticmethod
    def finalize(axis_name: str, plot_name: str):
        Visualize._update_plot(axis_name, plot_name)

        series = Visualize._get_series(axis_name, plot_name, False)
        series.clear()

        Visualize._plot_lines.pop(axis_name + "_" + plot_name + "_current")
        Visualize._iteration_increment(axis_name, plot_name, by=1)

    @staticmethod
    def reset(axis_name: str, plot_name: str):
        pass

    @staticmethod
    def show():
        pyplot.show()


class VisualizeSingle:
    fig = None
    plot_axes = dict()                  # type: Dict[str, axes]
    labels_axes_to_plots = dict()       # type: Dict[str, Collection[str]]

    current_series = dict()             # type: Dict[str, Dict[str, List[float]]]
    average_series = dict()             # type: Dict[str, Dict[str, List[float]]]

    average_plots = dict()              # type: Dict[str, Dict[str, Artist]]
    iteration = dict()                  # type: Dict[str, Dict[str, int]]

    indices = dict()                    # type: Dict[str, int]

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

            for _each_plot_name in _plot_names:
                VisualizeSingle.indices[_axis_name + _each_plot_name] = len(VisualizeSingle.indices)

    @staticmethod
    def update(axis_label: str, plot_label: str, value: float):
        try:
            axis = VisualizeSingle.current_series[axis_label]
        except KeyError:
            print(f"no axis called '{axis_label}'.")
            return

        try:
            series = axis[plot_label]
        except KeyError:
            print(f"no plot called '{plot_label}' in axis '{axis_label}'.")
            return

        series.append(value)

    @staticmethod
    def plot(each_axis_name: str, each_plot_name: str):
        each_axis = VisualizeSingle.plot_axes[each_axis_name]
        each_series = VisualizeSingle.current_series[each_axis_name]
        each_average = VisualizeSingle.average_series[each_axis_name]
        each_average_plot = VisualizeSingle.average_plots[each_axis_name]

        sub_dict = VisualizeSingle.iteration.get(each_axis_name)
        if sub_dict is None:
            sub_dict = dict()
            VisualizeSingle.iteration[each_axis_name] = sub_dict
        iterations = sub_dict.get(each_plot_name, 0)

        _series = each_series[each_plot_name]
        if 0 < iterations:
            _plot = each_average_plot[each_plot_name]
            _plot.remove()
            _average = [(_a * iterations + _s) / (iterations + 1) for _a, _s in zip(each_average[each_plot_name], _series)]

        else:
            _average = _series[:]

        hue_value = distribute_circular(VisualizeSingle.indices.get(each_axis_name + each_plot_name, 0))
        each_color_soft = hsv_to_rgb((hue_value, .5, .8))
        each_color_hard = hsv_to_rgb((hue_value, .7, .5))

        each_axis.plot(_series, color=each_color_soft, alpha=.5, label=each_plot_name)
        each_average[each_plot_name] = _average
        each_average_plot[each_plot_name],  = each_axis.plot(_average, color=each_color_hard, label="average " + each_plot_name)

        _series.clear()

        if iterations < 1:
            each_axis.legend()

        sub_dict[each_plot_name] = iterations + 1

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
    size = 1000
    labels = {"axis0": {"plot0", "plot1"}, "axis1": {"plot2"}}
    Visualize.init("test figure", labels, x_range=size, refresh_rate=100)

    f0 = lambda _x: sin(_x / 10.) + random.gauss(1., .5)
    f1 = lambda _x: cos(_x / 10.) + random.gauss(1., .5)
    f2 = lambda _x: (sin(_x / 7.) + cos(_x / 60.)) + random.gauss(1., .5)

    for i in range(100):
        print(f"starting iteration {i + 1:03d}/10...")
        now = time.time()
        for x in range(size):
            Visualize.append("axis0", "plot0", f0(x))
            Visualize.append("axis0", "plot1", f1(x))
            Visualize.append("axis1", "plot2", f2(x))

            # time.sleep(.0001)

        Visualize.finalize_all()
        print(f"lasted {time.time() - now} seconds.")

    print("finished")
    Visualize.show()

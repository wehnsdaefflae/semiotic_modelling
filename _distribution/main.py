# coding=utf-8
import math
import random
import time
from typing import Dict, Collection, List

from matplotlib import pyplot, cbook
from matplotlib.colors import hsv_to_rgb

from tools.math_functions import distribute_circular


class Visualize:
    _refresh_rate = 0
    _designators = None
    _figure = None

    _axes = dict()
    _plot_lines = dict()
    _progress_lines = dict()

    _current_series = None
    _average_series = None

    _finished_iterations = None

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

        Visualize._finished_iterations = {_axis: {_plot: 0 for _plot in plot_names} for _axis, plot_names in designators.items()}
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
            raise ValueError(f"No axis '{axis_name}' defined.")

        iterations = sub_dict.get(plot_name)
        if iterations is None:
            raise ValueError(f"No plot '{plot_name}' defined on axis '{axis_name}'.")

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
        plot_index = plot_names.shape(plot_name)
        hue_value = distribute_circular(plot_index)

        current_key = key_string + "_current"
        plot_line = Visualize._plot_lines.get(current_key)
        series = Visualize._get_series(axis_name, plot_name, False)
        color_soft = hsv_to_rgb((hue_value, .5, .8))
        if plot_line is not None:
            plot_line.remove()
        Visualize._plot_lines[current_key], = axis.plot(series, color=color_soft, alpha=.1)

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
    def _update_all():
        for axis_name in Visualize._designators:
            _axis = Visualize._axes.get(axis_name)
            if _axis is None:
                raise ValueError(f"No axis {axis_name}.")

            plot_names = Visualize._designators.get(axis_name)
            if plot_names is None:
                raise ValueError(f"No plot names for axis '{axis_name}'.")

            for _plot in plot_names:
                Visualize._update_plot(axis_name, _plot)

            _axis.legend()

    @staticmethod
    def append(axis_name: str, plot_name: str, value: float):
        series = Visualize._get_series(axis_name, plot_name, False)
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
        # Visualize._update_all()
        for each_axis, plot_names in Visualize._designators.items():
            for _plot in plot_names:
                Visualize._finalize(each_axis, _plot)

    @staticmethod
    def _finalize(axis_name: str, plot_name: str):
        Visualize._update_plot(axis_name, plot_name)

        series = Visualize._get_series(axis_name, plot_name, False)
        series.clear()

        Visualize._plot_lines.pop(axis_name + "_" + plot_name + "_current")
        Visualize._iteration_increment(axis_name, plot_name, by=1)

    @staticmethod
    def show():
        pyplot.show()


def test_visualize():
    size = 1000
    labels = {"axis0": {"plot0", "plot1"}, "axis1": {"plot2"}}
    Visualize.init("test figure", labels, x_range=size, refresh_rate=100)

    f0 = lambda _x: math.sin(_x / 10.) + random.gauss(1., .5)
    f1 = lambda _x: math.cos(_x / 10.) + random.gauss(1., .5)
    f2 = lambda _x: (math.sin(_x / 7.) + math.cos(_x / 60.)) + random.gauss(1., .5)

    for i in range(100):
        print(f"starting iteration {i + 1:03d}/10...")
        now = time.time()
        for x in range(size):
            Visualize.append("axis0", "plot0", f0(x))
            Visualize.append("axis0", "plot1", f1(x))
            Visualize.append("axis1", "plot2", f2(x))

            time.sleep(.0001)

        Visualize.finalize_all()
        print(f"lasted {time.time() - now} seconds.")

    print("finished")
    Visualize.show()


def create_image(x, y):
    return tuple(tuple((math.sqrt(_x ** 2 + _y ** 2), 0., 0., 0.) for _x in range(x)) for _y in range(y))


def show_image():
    x_size = 50
    y_size = 10
    image = create_image(x_size, y_size)

    #image_file = cbook.get_sample_data("D:/Projects/semiotic_modelling/_distribution/468px-Ada_Lovelace_color.svg.png")
    #image = pyplot.imread(image_file)

    print("\n".join(str(_x) for _x in image))

    fig, (ax1, ax2) = pyplot.subplots(2, sharex="all")
    im1 = ax1.imshow(image,
                     interpolation="bilinear", origin="lower",
                     extent=[-x_size//2, x_size//2, -y_size//2, y_size//2],
                     #vmax=max(max(_r) for _r in image),
                     #vmin=min(min(_r) for _r in image),
                     aspect='auto')
    im2 = ax2.imshow(image,
                     interpolation="bilinear", origin="upper",
                     extent=[-x_size//2, x_size//2, -y_size//2, y_size//2],
                     #vmax=max(max(_r) for _r in image),
                     #vmin=min(min(_r) for _r in image),
                     aspect='auto')
    pyplot.show()


if __name__ == "__main__":
    test_visualize()

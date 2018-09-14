# coding=utf-8
from typing import TypeVar, Generic, Tuple

from matplotlib import pyplot

OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


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

    def clear(self):
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

        pyplot.draw()
        pyplot.pause(.001)

        self.clear()

    def finish(self):
        pyplot.show()

# coding=utf-8
from typing import TypeVar, Generic, Tuple

from matplotlib import pyplot

TIME_TYPE = TypeVar("TIME_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Visualization(Generic[TIME_TYPE, OUTPUT_TYPE]):
    def __init__(self, title: str):
        self.title = title

    def update_output(self, time: TIME_TYPE, output: OUTPUT_TYPE, target: OUTPUT_TYPE):
        raise NotImplementedError()

    def update_error(self, time: TIME_TYPE, error: float):
        raise NotImplementedError()

    def update_duration(self, time: TIME_TYPE, duration: float):
        raise NotImplementedError()


class MixinInteractive(Generic[TIME_TYPE]):
    def update_reward(self, time: TIME_TYPE, reward: float):
        raise NotImplementedError()


class MixinSemiotic(Generic[TIME_TYPE]):
    def update_structure(self, time: TIME_TYPE, structure: Tuple[int, ...]):
        raise NotImplementedError()


class VisualizationInteractive(MixinInteractive[TIME_TYPE], Visualization[TIME_TYPE, OUTPUT_TYPE]):
    def __init__(self, title: str):
        super().__init__(title)

    def update_reward(self, time: TIME_TYPE, reward: float):
        raise NotImplementedError()

    def update_output(self, time: TIME_TYPE, output: OUTPUT_TYPE, target: OUTPUT_TYPE):
        raise NotImplementedError()

    def update_error(self, time: TIME_TYPE, error: float):
        raise NotImplementedError()

    def update_duration(self, time: TIME_TYPE, duration: float):
        raise NotImplementedError()


class VisualizationSemiotic(MixinSemiotic[TIME_TYPE], Visualization[TIME_TYPE, OUTPUT_TYPE]):
    def __init__(self, title: str):
        super().__init__(title)

    def update_structure(self, time: TIME_TYPE, structure: Tuple[int, ...]):
        raise NotImplementedError()

    def update_output(self, time: TIME_TYPE, output: OUTPUT_TYPE, target: OUTPUT_TYPE):
        raise NotImplementedError()

    def update_error(self, time: TIME_TYPE, error: float):
        raise NotImplementedError()

    def update_duration(self, time: TIME_TYPE, duration: float):
        raise NotImplementedError()


class VisualizationInteractiveSemiotic(MixinSemiotic[TIME_TYPE], VisualizationInteractive[TIME_TYPE, OUTPUT_TYPE]):
    def __init__(self, title: str):
        super().__init__(title)

    def update_structure(self, time: TIME_TYPE, structure: Tuple[int, ...]):
        raise NotImplementedError()

    def update_reward(self, time: TIME_TYPE, reward: float):
        raise NotImplementedError()

    def update_output(self, time: TIME_TYPE, output: OUTPUT_TYPE, target: OUTPUT_TYPE):
        raise NotImplementedError()

    def update_error(self, time: TIME_TYPE, error: float):
        raise NotImplementedError()

    def update_duration(self, time: TIME_TYPE, duration: float):
        raise NotImplementedError()


class VisualizationPyplot(VisualizationSemiotic[TIME_TYPE, OUTPUT_TYPE]):
    def __init__(self, title: str, accumulation_steps: int):
        super().__init__(title)
        self.fig, axes = pyplot.subplots(4, sharex="all")
        self.fig.suptitle(title)
        self.axis_out, self.axis_error, self.axis_structure, self.axis_duration = axes

        self.accumulation_steps = accumulation_steps

        self.time_output, self.values_output = [], []
        self.time_target, self.values_target = [], []
        self.time_error, self.values_error = [], []
        self.time_structure, self.values_structure = [], []
        self.time_duration, self.values_duration = [], []

    def update_output(self, time: TIME_TYPE, output: OUTPUT_TYPE, target: OUTPUT_TYPE):
        self.time_output.append(time)
        self.values_output.append(output)

        self.time_target.append(time)
        self.values_target.append(target)

    def update_error(self, time: TIME_TYPE, error: float):
        self.time_error.append(time)
        len_error = len(self.values_error)
        last_value = 1. if len_error < 1 else self.values_error[-1]
        self.values_error.append((last_value * len_error + error) / (len_error + 1))

    def update_duration(self, time: TIME_TYPE, duration: float):
        self.time_duration.append(time)
        len_duration = len(self.values_duration)
        last_value = 1. if len_duration < 1 else self.values_duration[-1]
        self.values_duration.append((last_value * len_duration + duration) / (len_duration + 1))

    def update_structure(self, time: TIME_TYPE, structure: Tuple[int, ...]):
        pass

    def show(self):
        #self.axis_out.plot(self.time_output, self.values_output, label="output")
        #self.axis_out.plot(self.time_target, self.values_target, label="target")
        self.axis_out.set_ylabel("output / target")
        self.axis_out.legend()

        self.axis_error.plot(self.time_error, self.values_error, label="error")
        self.axis_error.set_ylabel("error")
        self.axis_error.legend()

        # self.axis_structure.plot
        self.axis_structure.set_ylabel("structure")
        self.axis_structure.legend()

        self.axis_duration.plot(self.time_duration, self.values_duration, label="duration")
        self.axis_duration.set_ylabel("duration (ms)")
        self.axis_duration.legend()

        pyplot.draw()
        pyplot.pause(.001)

        self.time_output.clear()
        self.time_target.clear()
        self.time_error.clear()
        self.time_structure.clear()
        self.time_duration.clear()

        self.values_output.clear()
        self.values_target.clear()
        self.values_error.clear()
        self.values_structure.clear()
        self.values_duration.clear()

    def finish(self):
        pyplot.show()


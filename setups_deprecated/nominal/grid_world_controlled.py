# coding=utf-8
from matplotlib import pyplot

from data_generation.data_sources.systems.environments import GridWorldGlobal, GridWorldLocal
from data_generation.data_sources.systems import SarsaController
from setups_deprecated.evaluations import interaction
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from tools.load_configs import Config
from visualization.old_visualization import Canvas


def semiotic(rotational, iterations):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = SarsaController(movement, alpha=.1, gamma=.1, epsilon=.1)
    environment = GridWorldLocal(c["data_dir"] + "grid_worlds/simple.txt")
    predictor = NominalSemioticModel(no_examples=1, alpha=50, sigma=.2, trace_length=1)

    return interaction(environment, controller, predictor, rational=False, iterations=iterations)


def markov(rotational, iterations):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = SarsaController(movement, alpha=.1, gamma=.8, epsilon=.1)
    environment = GridWorldLocal(c["data_dir"] + "grid_worlds/simple.txt")
    predictor = NominalMarkovModel(no_examples=1)

    return interaction(environment, controller, predictor, rational=False, iterations=iterations)


def markov_absolute(rotational, iterations):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = SarsaController(movement, alpha=.1, gamma=.8, epsilon=.1)
    environment = GridWorldGlobal(c["data_dir"] + "grid_worlds/simple.txt")
    predictor = NominalMarkovModel(no_examples=1)

    return interaction(environment, controller, predictor, rational=False, iterations=iterations)


def multiple_runs(experiment, repetitions, label, color: str = "C0"):
    average_time_axis = []
    average_errors = []
    average_rewards = []

    error_line = None
    reward_line = None
    for i in range(repetitions):
        time_axis, errors, rewards = experiment()

        label_error = "error {:s}".format(label)
        label_error = ""
        #Canvas.ax1.plot(time_axis, errors, color=color, label=label_error, alpha=.5)
        label_reward = "reward {:s}".format(label)
        label_reward = ""
        #Canvas.ax2.plot(time_axis, rewards, color=color, label=label_reward, alpha=.5)

        lines, labels = Canvas.ax1.get_legend_handles_labels()
        lines2, labels2 = Canvas.ax11.get_legend_handles_labels()

        # Canvas.ax11.legend(lines + lines2, labels + labels2)

        #Canvas.ax2.set_ylabel("iteration time (ms)")
        #Canvas.ax2.plot(time_axis, durations, label="{:s} {:d}".format(predictor.__class__.__name__, predictor.no_examples))
        #Canvas.ax2.legend()

        if len(average_time_axis) < 1:
            average_time_axis = time_axis

        if len(average_errors) < 1:
            average_errors = errors
        else:
            assert len(average_errors) == len(errors)
            for _i, (each_avrg, each_error) in enumerate(zip(average_errors, errors)):
                average_errors[_i] = (each_avrg * i + each_error) / (i + 1)

        if error_line is not None:
            error_line.remove()
        error_line, = Canvas.ax1.plot(average_time_axis, average_errors, color=color, label=label)

        if len(average_rewards) < 1:
            average_rewards = rewards
        else:
            assert len(average_rewards) == len(rewards)
            for _i, (each_avrg, each_reward) in enumerate(zip(average_rewards, rewards)):
                average_rewards[_i] = (each_avrg * i + each_reward) / (i + 1)

        if reward_line is not None:
            reward_line.remove()
        reward_line, = Canvas.ax2.plot(average_time_axis, average_rewards, color=color, label=label)

        Canvas.ax1.legend()
        Canvas.ax2.legend()

        pyplot.draw()
        pyplot.pause(.001)
        print("{:03d}/{:03d} finished".format(i + 1, repetitions))


if __name__ == "__main__":
    iterate = 200000
    repeat = 50

    Canvas.ax1.set_ylabel("error")
    Canvas.ax2.set_ylabel("reward")

    multiple_runs(lambda: markov(rotational=True, iterations=iterate), repeat, "markov", color="C0")
    multiple_runs(lambda: markov_absolute(rotational=True, iterations=iterate), repeat, "markov global", color="C1")
    multiple_runs(lambda: semiotic(rotational=True, iterations=iterate), repeat, "semiotic", color="C2")

    pyplot.show()

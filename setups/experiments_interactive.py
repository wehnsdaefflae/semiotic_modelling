# coding=utf-8
import time

from data_generation.data_sources.systems.controller_nominal import SarsaController
from data_generation.data_sources.systems.environments import GridWorldLocal
from modelling.predictors.abstract_predictor import Predictor
from modelling.predictors.nominal.baseline import NominalMarkovModel
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from tools.load_configs import Config
from tools.timer import Timer
from visualization.visualization import VisualizeSingle


def experiment(repeat: int = 10):
    VisualizeSingle.initialize(
        {
            "reward": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "error": {NominalSemioticModel.__name__, NominalMarkovModel.__name__},
            "duration": {NominalSemioticModel.__name__, NominalMarkovModel.__name__}
        },
        "grid world")

    for _i in range(repeat):
        print("Run {:d} of {:d}...".format(_i * 2 + 1, repeat * 2))
        predictor = NominalSemioticModel(
            no_examples=1,
            alpha=100,
            sigma=.2,
            trace_length=1)
        controlled_grid_interaction(predictor, iterations=500000)

        print("Run {:d} of {:d}...".format(_i * 2 + 2, repeat * 2))
        predictor = NominalMarkovModel(no_examples=1)
        controlled_grid_interaction(predictor, iterations=500000)

    print("done!")
    VisualizeSingle.finish()


def controlled_grid_interaction(predictor: Predictor, iterations: int = 500000):
    c = Config("../configs/config.json")
    data_dir = c["data_dir"] + "grid_worlds/"

    grid_world = GridWorldLocal(data_dir + "square.txt", rotational=True)
    # grid_world = GridWorldLocal(data_dir + "simple.txt", rotational=False)
    # grid_world = GridWorldGlobal(data_dir + "sutton.txt", rotational=False)

    controller = SarsaController(grid_world.get_motor_range(), alpha=.8, gamma=.1, epsilon=.1)
    # controller = RandomController(grid_world.get_motor_range())

    last_sensor = None
    last_motor = None
    sensor, reward = grid_world.react_to(None)

    visualization_steps = iterations // 1000
    average_reward = .0
    average_error = .0
    average_duration = .0
    for t in range(iterations):
        # get data
        this_time = time.time()
        concurrent_inputs = (last_sensor, last_motor),
        concurrent_outputs = predictor.predict(concurrent_inputs)
        concurrent_targets = sensor,
        concurrent_examples = (concurrent_inputs[0], concurrent_targets[0]),
        predictor.fit(concurrent_examples)
        d = time.time() - this_time

        # query controller
        perception = predictor.get_state(), last_sensor
        motor = controller.react_to(perception, reward)

        error = sum(float(_o != _t) for _o, _t in zip(concurrent_outputs, concurrent_targets)) / len(concurrent_targets)

        average_reward = (average_reward * t + reward) / (t + 1)
        average_error = (average_error * t + error) / (t + 1)
        average_duration = (average_duration * t + d) / (t + 1)

        if (t + 1) % visualization_steps == 0:
            VisualizeSingle.update("reward", predictor.__class__.__name__, average_reward)
            VisualizeSingle.update("error", predictor.__class__.__name__, average_error)
            VisualizeSingle.update("duration", predictor.__class__.__name__, average_duration)

        last_sensor = sensor
        last_motor = motor

        sensor, reward = grid_world.react_to(motor)

        if Timer.time_passed(2000):
            print("Finished {:05.2f}%...".format(100. * t / iterations))

    VisualizeSingle.plot("reward", predictor.__class__.__name__)
    VisualizeSingle.plot("error", predictor.__class__.__name__)
    VisualizeSingle.plot("duration", predictor.__class__.__name__)
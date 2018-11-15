# coding=utf-8
import math
from math import cos
from typing import Optional, Tuple, Dict, Any

import gym
import numpy

from _framework.data_types import RATIONAL_MOTOR, RATIONAL_SENSOR
from _framework.systems.tasks.rational.abstract import RationalTask
from tools.functionality import signum, smear, clip


class MountainCar(RationalTask, gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mass = 100.
        self._at_top = False
        self._location = 0.
        self._velocity = 0.
        self._hill = lambda _x: (-cos(_x) + 1.) / 2.
        self._hill_force = lambda _x: -signum(_x) * (-cos(_x * 2.) + 1.) / 2.

        self._viewer = None
        self._track = None
        self._car_trans = None
        self._car = None
        self._car_green = 0.

        self.reset()

    def reset(self):
        self._location = 0.
        self._velocity = 0.
        return self._location, self._velocity

    def react(self, data_in: Optional[RATIONAL_MOTOR]) -> RATIONAL_SENSOR:
        force = data_in[0] * .15 + self._hill_force(self._location)
        acceleration = force / self._mass
        self._velocity = clip((self._velocity + acceleration) * .99, -.02, .02)
        self._location += self._velocity

        if self._location >= math.pi:
            self._at_top = True
            self._location += -2. * math.pi

        elif -math.pi >= self._location:
            self._at_top = True
            self._location = 2. * math.pi - self._location

        elif self._at_top:
            self._at_top = False

        return self._location, self._velocity

    @staticmethod
    def motor_range() -> Tuple[Tuple[float, float], ...]:
        return (-1., 1.),

    def _get_height(self, location: float) -> float:
        return self._hill(location)

    def _get_reward(self) -> float:
        return self._hill(self._location) - 1.  #  10. if self._at_top else -1.

    def step(self, action: numpy.ndarray) -> Tuple[numpy.ndarray, float, bool, Dict[str, Any]]:
        sensor = self.react(action)
        return numpy.array(sensor), self._get_reward(), False, {}

    def render(self, mode: str = "human"):
        screen_width = 600
        screen_height = 400

        car_width = 40
        car_height = 20

        def v_x(real_x: float) -> float:
            return screen_width * (real_x + math.pi) / (2. * math.pi)

        def v_y(real_y: float) -> float:
            return real_y * screen_height / 25.

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.Viewer(screen_width, screen_height)

            x_range = numpy.linspace(-math.pi, math.pi, 100)
            y_values = numpy.array(tuple(v_y(self._hill(_y)) for _y in x_range))
            track_data = list(zip((v_x(_x) for _x in x_range), (v_y(_y) for _y in y_values)))

            self._track = rendering.make_polyline(track_data)
            self._track.set_linewidth(4)
            self._viewer.add_geom(self._track)

            clearance = 10

            l, r, t, b = -car_width / 2, car_width / 2, car_height, 0
            self._car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self._car.add_attr(rendering.Transform(translation=(0, clearance)))

            self._car_trans = rendering.Transform()
            self._car.add_attr(self._car_trans)
            self._viewer.add_geom(self._car)
            front_wheel = rendering.make_circle(car_height / 2.5)
            front_wheel.set_color(.5, .5, .5)
            front_wheel.add_attr(rendering.Transform(translation=(car_width / 4, clearance)))
            front_wheel.add_attr(self._car_trans)
            self._viewer.add_geom(front_wheel)

            back_wheel = rendering.make_circle(car_height / 2.5)
            back_wheel.add_attr(rendering.Transform(translation=(-car_width / 4, clearance)))
            back_wheel.add_attr(self._car_trans)
            back_wheel.set_color(.5, .5, .5)
            self._viewer.add_geom(back_wheel)

        if self._at_top:
            self._car_green = 1.
        else:
            self._car_green = smear(self._car_green, 0., 10)

        self._car.set_color(0., self._car_green, 0.)

        pos = v_x(self._location)
        self._car_trans.set_translation(pos, v_y(self._hill(self._location)) * 20.)
        self._car_trans.set_rotation(math.sin(self._location))

        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

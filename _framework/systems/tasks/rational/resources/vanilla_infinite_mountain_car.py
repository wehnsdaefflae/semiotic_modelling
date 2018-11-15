# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from 
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
import numpy
from gym import spaces
from gym.utils import seeding
import numpy as np


class ContinuousMountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.state = None
        self.np_random = None
        self.track = None
        self.car_trans = None

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: numpy.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force * self.power - .0025 * math.cos(3. * position)

        if velocity > self.max_speed:
            velocity = self.max_speed

        elif velocity < -self.max_speed:
            velocity = -self.max_speed

        position += velocity

        if position > self.max_position:
            position = self.max_position

        elif position < self.min_position:
            position = self.min_position

        if position == self.min_position and velocity < .0:
            velocity = .0

        done = bool(position >= self.goal_position)

        reward = 0
        if done:
            reward = 100.0

        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3. * xs) * .45 + .55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        car_width = 40
        car_height = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -car_width / 2, car_width / 2, car_height, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))

            self.car_trans = rendering.Transform()
            car.add_attr(self.car_trans)
            self.viewer.add_geom(car)
            front_wheel = rendering.make_circle(car_height / 2.5)
            front_wheel.set_color(.5, .5, .5)
            front_wheel.add_attr(rendering.Transform(translation=(car_width / 4, clearance)))
            front_wheel.add_attr(self.car_trans)
            self.viewer.add_geom(front_wheel)

            back_wheel = rendering.make_circle(car_height / 2.5)
            back_wheel.add_attr(rendering.Transform(translation=(-car_width / 4, clearance)))
            back_wheel.add_attr(self.car_trans)
            back_wheel.set_color(.5, .5, .5)
            self.viewer.add_geom(back_wheel)

            flag_x = (self.goal_position - self.min_position) * scale
            flag_y1 = self._height(self.goal_position) * scale
            flag_y2 = flag_y1 + 50
            flagpole = rendering.Line((flag_x, flag_y1), (flag_x, flag_y2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flag_x, flag_y2), (flag_x, flag_y2 - 10), (flag_x + 25, flag_y2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.car_trans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.car_trans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

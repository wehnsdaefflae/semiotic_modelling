# coding=utf-8
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import numpy
from gym import spaces
from gym.utils import seeding
import numpy as np


class InfiniteCartPoleEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.
        self.mass_pole = .1
        self.total_mass = (self.mass_pole + self.mass_cart)
        self.length = .5                                        # actually half the pole's length
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 10.
        self.tau = .02                                          # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12. * 2. * math.pi / 360.
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array(
            [
                self.x_threshold * 2.,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2.,
                np.finfo(np.float32).max
            ]
        )

        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.np_random = None

        self.seed()
        self.viewer = None
        self.state = None

        self.cart_trans = None
        self.pole_trans = None
        self.axle = None
        self.track = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: numpy.ndarray):
        assert self.action_space.contains(action), f"{action.__repr__():s} ({str(type(action)):s}) invalid"

        state = self.state
        x_pos, x_vel, theta_ang, theta_vel = state
        force = self.force_mag * action[0]
        cos_theta = math.cos(theta_ang)
        sin_theta = math.sin(theta_ang)
        temp = (force + self.pole_mass_length * theta_vel * theta_vel * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (self.length * (4. / 3. - self.mass_pole * cos_theta * cos_theta / self.total_mass))
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass

        if self.kinematics_integrator == "euler":
            x_pos += self.tau * x_vel
            x_vel += self.tau * x_acc
            theta_ang += self.tau * theta_vel
            theta_vel += self.tau * theta_acc

        else:   # semi-implicit euler
            x_vel += self.tau * x_acc
            x_pos += self.tau * x_vel
            theta_vel += self.tau * theta_acc
            theta_ang += self.tau * theta_vel

        theta_vel *= .99    # friction

        if x_vel < -10.:
            x_vel = -10.

        elif 10 < x_vel:
            x_vel = 10.

        if x_pos < -self.x_threshold:
            x_pos += self.x_threshold + self.x_threshold

        elif self.x_threshold < x_pos:
            x_pos += -self.x_threshold - self.x_threshold

        self.state = x_pos, x_vel, theta_ang, theta_vel

        # reward = (abs(theta_ang + math.pi) / math.pi) - 1.
        reward = abs(abs(theta_ang % (2. * math.pi)) - math.pi) / math.pi - 1.
        assert -1. <= reward <= 0.
        return np.array(self.state), reward, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-.05, high=.05, size=(4,))
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2.
        scale = screen_width/world_width
        cart_y = 100                        # TOP OF CART
        pole_width = 10.
        pole_len = scale * 1.
        cart_width = 50.
        cart_height = 30.

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cart_width / 2., cart_width / 2., cart_height / 2., -cart_height / 2.
            axle_offset = cart_height / 4.
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)
            l, r, t, b = -pole_width / 2., pole_width / 2., pole_len - pole_width / 2., -pole_width / 2.
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)

            self.pole_trans = rendering.Transform(translation=(0, axle_offset))
            pole.add_attr(self.pole_trans)
            pole.add_attr(self.cart_trans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(pole_width / 2.)
            self.axle.add_attr(self.pole_trans)
            self.axle.add_attr(self.cart_trans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)

            self.track = rendering.Line((0, cart_y), (screen_width, cart_y))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cart_x = x[0] * scale + screen_width / 2.           # MIDDLE OF CART
        self.cart_trans.set_translation(cart_x, cart_y)
        self.pole_trans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

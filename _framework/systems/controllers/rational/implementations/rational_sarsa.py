# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController
from tools.base_tools.approximation.rational_to_rational import MultiplePolynomialFromLinearRegression, MultivariatePolynomialRegression
from tools.functionality import clip, smear, cartesian_distance


class RationalSarsa(RationalController):
    def __init__(self,
                 motor_range: Tuple[Tuple[float, float], ...], sensor_dimensionality: int,
                 past_scope: int, learning_drag: int, gamma: float, epsilon: float, polynomial_degree: int = 1):
        super().__init__(motor_range, epsilon)
        self._past_scope = past_scope
        self._learning_drag = learning_drag
        self._gamma = gamma

        self._last_sensor = None
        self._last_motor = None
        self._last_reward = 0.
        self._last_condition = None

        self._sensor_dim = sensor_dimensionality

        self._critic_input_dim = len(motor_range) + sensor_dimensionality
        self._motor_range = motor_range

        # S x M -> float
        self._critic = MultiplePolynomialFromLinearRegression(
            sensor_dimensionality + len(motor_range),
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag)

        # S -> M
        self._actor = MultivariatePolynomialRegression(
            sensor_dimensionality,
            len(motor_range),
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag)

        self.average_critic_error = 0.
        self.average_actor_error = 0.

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        action = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(sensor), self._motor_range))
        # action = tuple(0. for _ in self._motor_range)
        return action

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        # https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html
        # todo: check out actor only policy optimization
        # todo: check out sensor-only evaluation
        # todo: check out advantage: https://medium.freecodecamp.org/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d

        if self._iteration >= 1:
            this_input = sensor + motor
            evaluation = self._critic.output(this_input)
            last_evaluation_target = self._last_reward + self._gamma * evaluation

            last_input = self._last_sensor + self._last_motor

            critic_error = self._critic.fit(last_input, last_evaluation_target)
            self.average_critic_error = smear(self.average_critic_error, critic_error, self._iteration - 1)

            best_known = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(self._last_sensor), self._motor_range))
            best_known_eval = self._critic.output(self._last_sensor + best_known)

            delta_eval = last_evaluation_target - best_known_eval
            delta_step = tuple(_l - _b for _l, _b in zip(self._last_motor, best_known))
            # better_motor = tuple(clip(_b + signum(_d * delta_eval) * .01, *_ranges) for _b, _d, _ranges in zip(best_known, delta_step, self._motor_range))
            # better_motor = tuple(clip(_b + _d * delta_eval * .01, *_ranges) for _b, _d, _ranges in zip(best_known, delta_step, self._motor_range))
            better_motor = tuple(clip(smear(_b, _b + _d * delta_eval, self._past_scope), *_ranges) for _b, _d, _ranges in zip(best_known, delta_step, self._motor_range))

            actor_errors = self._actor.fit(self._last_sensor, better_motor)
            self.average_actor_error = smear(self.average_actor_error, cartesian_distance(actor_errors), self._iteration - 1)

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

        self._iteration += 1

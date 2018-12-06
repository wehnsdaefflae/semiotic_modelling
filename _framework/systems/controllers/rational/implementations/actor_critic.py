# coding=utf-8
import random
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController
from tools.base_tools.approximation.rational_to_rational import MultiplePolynomialFromLinearRegression, MultivariatePolynomialRegression
from tools.functionality import clip, smear, cartesian_distance, signum


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

        self._motor_range = motor_range

        # S -> float
        self._critic = MultiplePolynomialFromLinearRegression(
            sensor_dimensionality,
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag)

        # S -> M (float, float)
        self._actor = MultivariatePolynomialRegression(
            sensor_dimensionality,
            2 * len(motor_range),
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag)

        self.average_critic_error = 0.
        self.average_actor_error = 0.

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        action_parameters = self._actor.output(sensor)
        return tuple(
            random.normalvariate(*action_parameters[_i:_i+2])
            for _i in range(len(action_parameters) - 1)
        )

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        # https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html
        # todo: fix gradient descent!
        # todo: optimize with score function trick: http://www.youtube.com/watch?v=bRfUxQs6xIM&t=37m24s
        # todo: implement according to:
        #       http://www.youtube.com/watch?v=bRfUxQs6xIM&t=72m0s

        iteration = self.get_iterations()

        if iteration >= 1:
            td_error = reward + self._gamma * self._critic.output(sensor) - self._critic.output(self._last_sensor)

            critic_parameters = self._critic.get_parameters()
            critic_gradient = self._critic.gradient(self._last_sensor)
            for _i, each_parameter in enumerate(critic_parameters):
                critic_parameters[_i] += .1 * td_error * critic_gradient[_i]

            actor_parameters = self._actor.get_parameters()
            actor_log_gradient = tuple(_x / _y for _x, _y in zip(self._actor.gradient(self._last_sensor), self._last_sensor))
            for _i, each_parameter in enumerate(actor_parameters):
                actor_parameters[_i] += .1 * td_error * actor_log_gradient[_i]
                # gradient of log of probability of action in state

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

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
        self._stochastic_policy = MultiplePolynomialFromLinearRegression(
            sensor_dimensionality + len(motor_range),
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag
        )

        self.average_actor_error = 0.
        self.average_value_error = 0.

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        intended_action = max(self._motor_range, lambda _x: self._stochastic_policy(sensor + _x))
        action = tuple(clip(_m, *_ranges) for _m, _ranges in zip(intended_action, self._motor_range))
        return action

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        # todo: optimize parameters according to gradient!
        # todo: fix gradient descent!
        # todo: policy as specific shape (e.g. polynomial) function
        # todo: optimize with score function trick: http://www.youtube.com/watch?v=bRfUxQs6xIM&t=37m24s
        # https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f

        iteration = self.get_iterations()

        if iteration >= 1:
            # evaluation S -> float
            evaluation_this = self._stochastic_policy.output(sensor + motor)
            evaluation_new = self._last_reward + self._gamma * evaluation_this + reward

            # todo: optimize policy parameters according to value
            parameters = self._stochastic_policy.get_parameters()
            for _i in range(len(parameters)):
                parameters[_i] += .1 * self._stochastic_policy.derivation_output(sensor + motor, _i) * evaluation_new
            self._stochastic_policy.set_parameters(parameters)

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

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

        """
        # S x M -> float  (probability of action in state)
        self._policy = MultiplePolynomialFromLinearRegression(
            sensor_dimensionality + len(motor_range),
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag
        )
        """

        # S -> M
        self._actor = MultivariatePolynomialRegression(
            sensor_dimensionality,
            len(motor_range),
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag)

        # S -> float
        self._evaluation = MultiplePolynomialFromLinearRegression(
            sensor_dimensionality,
            polynomial_degree,
            past_scope=past_scope,
            learning_drag=learning_drag
        )

        self.average_actor_error = 0.
        self.average_value_error = 0.

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        action = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(sensor), self._motor_range))
        # action = tuple(0. for _ in self._motor_range)
        return action

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        # todo: fix gradient descent!
        # todo: policy as specific shape (e.g. polynomial) function
        # todo: optimize with score function trick: http://www.youtube.com/watch?v=bRfUxQs6xIM&t=37m24s
        # https://medium.freecodecamp.org/an-introduction-to-policy-gradients-with-cartpole-and-doom-495b5ef2207f

        iteration = self.get_iterations()

        if iteration >= 1:
            last_input = self._last_sensor + self._last_motor

            # evaluation S -> float
            evaluation = self._evaluation.output(sensor)
            evaluation_error = self._evaluation.fit(self._last_sensor, self._last_reward + self._gamma * evaluation)
            self.average_value_error = smear(self.average_value_error, evaluation_error, iteration - 1)

            # todo: optimize policy parameters according to value

            # actor S -> M (consider advantage instead of critic value)
            best_known = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(self._last_sensor), self._motor_range))
            best_known_advantage = self._advantage.output(self._last_sensor + best_known)
            delta_eval = last_advantage - best_known_advantage
            delta_step = tuple(_l - _b for _l, _b in zip(self._last_motor, best_known))
            better_motor = tuple(clip(smear(_b, _b + _d * delta_eval, self._past_scope), *_ranges) for _b, _d, _ranges in zip(best_known, delta_step, self._motor_range))
            actor_errors = self._actor.fit(self._last_sensor, better_motor)
            self.average_actor_error = smear(self.average_actor_error, cartesian_distance(actor_errors), iteration - 1)

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

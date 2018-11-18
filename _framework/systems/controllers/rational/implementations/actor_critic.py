# coding=utf-8
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
        # todo: fix gradient descent!

        iteration = self.get_iterations()

        if iteration >= 1:
            last_input = self._last_sensor + self._last_motor
            this_input = sensor + motor

            # critic S x M -> float
            critic_evaluation = self._critic.output(this_input)
            last_evaluation = self._last_reward + self._gamma * critic_evaluation
            critic_error = self._critic.fit(last_input, last_evaluation)
            self.average_critic_error = smear(self.average_critic_error, critic_error, iteration - 1)

            # actor S -> M (consider advantage instead of critic value)
            best_motor = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(self._last_sensor), self._motor_range))
            evaluation_difference = last_evaluation - self._critic.output(self._last_sensor + best_motor)
            motor_difference = tuple(_l - _b for _l, _b in zip(self._last_motor, best_motor))

            # optimized_motor = tuple(
            #     clip(_b + signum(_d * evaluation_difference) * .01, *_ranges)
            #     for _b, _d, _ranges in zip(best_motor, motor_difference, self._motor_range)
            # )
            # optimized_motor = tuple(
            #     clip(_b + _d * evaluation_difference * .01, *_ranges)
            #     for _b, _d, _ranges in zip(best_motor, motor_difference, self._motor_range)
            # )
            optimized_motor = tuple(
                clip(
                    smear(_b, _b + _d * evaluation_difference, self._past_scope),
                    *_ranges
                )
                for _b, _d, _ranges in zip(best_motor, motor_difference, self._motor_range)
            )
            actor_errors = self._actor.fit(self._last_sensor, optimized_motor)
            self.average_actor_error = smear(self.average_actor_error, cartesian_distance(actor_errors), iteration - 1)

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

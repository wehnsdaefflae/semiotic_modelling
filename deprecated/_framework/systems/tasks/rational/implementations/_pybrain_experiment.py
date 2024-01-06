# coding=utf-8
from pybrain.optimization.hillclimber import HillClimber
from pybrain.rl.agents.optimization import OptimizationAgent

from pybrain.rl.environments.cartpole.balancetask import BalanceTask
from pybrain.rl.experiments.episodic import EpisodicExperiment
from pybrain.tools.shortcuts import buildNetwork


task = BalanceTask()

net = buildNetwork(task.outdim, 3, task.indim)

agent = OptimizationAgent(net, HillClimber())
exp = EpisodicExperiment(task, agent)
exp.doEpisodes(100)

print(exp)

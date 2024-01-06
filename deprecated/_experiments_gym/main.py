# coding=utf-8
import time

import gym

env = gym.make("MyEnv-v10")

action_space = env.action_space.n
#action_meanings = env.env.get_action_meanings()
#print(action_meanings)

state = env.reset()

counter = 0
reward = -1.

while True:
    env.render()

    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    #print(state)
    #print(reward)
    #print(done)
    #print(info)

    time.sleep(.01)

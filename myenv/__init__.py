# coding=utf-8

from gym.envs.registration import register

register(
    id='MyEnv-v10',
    entry_point='myenv.myenv:MyEnv',
    max_episode_steps=999,
    reward_threshold=90.0,

)

# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/

import gym

# https://github.com/openai/gym/blob/master/gym/envs/__init__.py
env = gym.make("CartPole-v1")
env.reset()


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(50):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            #if done:
            #    break


some_random_games_first()
env.close()
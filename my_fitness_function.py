import gymnasium as gym
import numpy as np
from NEAT.ffnn import FeedForwardNetwork


# Implement your custom fitness function here
def fitness_function(genome):
    network = FeedForwardNetwork.create(genome)
    env = gym.make("BipedalWalker-v3")  # "LunarLander-v3" "BipedalWalker-v3"
    rewards = []

    for i in range(5):
        rewards.append(0)
        state = env.reset()[0]
        done = False
        while not done:
            pred = network.activate(state)
            action = pred#np.argmax(pred)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            state = next_state
            rewards[i] += reward

            if done:
                env.reset()

    env.close()
    return np.mean(rewards)

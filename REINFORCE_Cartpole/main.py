import gym
import numpy as np
from reinforce import reinforce
from utils import evaluate_policy
import matplotlib.pyplot as plt

EPISODES = 5000
RUNS = 1000
G = 0.9

if __name__ == "__main__":
    # setup gym environment
    env = gym.make('CartPole-v0')

    env.reset()

    print("-----------------------------TRAINING-----------------------------")
    policy_net, reward_per_episode = reinforce(env, gamma=G, episodes=EPISODES, render=True)

    # plot total reward per episode
    plt.figure(1)
    plt.title("Training: REINFORCE Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    r_line, = plt.plot(reward_per_episode, label="Reward")
    plt.legend(handles=[r_line])
    plt.show()

    print("----------------------------EVALUATING-----------------------------")
    wins, reward_per_episode = evaluate_policy(env, policy_net, render=True, runs=RUNS)
    # display results
    print("-------------------------------------------------------------------")
    print("Policy resulted in winning: " + str((wins / float(RUNS)) * 100) + \
          " percent of the time")
    print("Policy resulted in losing: " + str(((RUNS - wins) / float(RUNS)) * 100) + \
          " percent of the time")
    print("-------------------------------------------------------------------")

    # plot total reward per episode
    plt.figure(1)
    plt.title("Evaluating: REINFORCE Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    r_line, = plt.plot(reward_per_episode, label="Reward")
    plt.legend(handles=[r_line])
    plt.show()

    env.close()

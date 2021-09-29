import gym
import numpy as np
from utils import Discretizer, init_e_soft_policy, evaluate_policy
from algorithms import sarsa, q_learning
import matplotlib.pyplot as plt

NUM_STATES = 20
EPISODES = 10000
E = 0.1
A = 0.1
EVAL = 1000
MAX_VELOCITY = 4

if __name__ == "__main__":
    # setup gym environment
    env = gym.make('CartPole-v0')
    num_actions = env.action_space.n

    discretizer = Discretizer(env, NUM_STATES, num_actions, max_v=MAX_VELOCITY)

    policy = init_e_soft_policy(NUM_STATES, num_actions)
    print("--------------------------TRAINING SARSA---------------------------")
    policy, reward_per_episode = sarsa(env, discretizer, policy, NUM_STATES, num_actions, episodes=EPISODES, eps=E, alpha=A)

    # plot total reward per episode
    plt.figure(1)
    plt.title("Training: Sarsa Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    r_line, = plt.plot(reward_per_episode, label="Reward")
    plt.legend(handles=[r_line])
    plt.show()

    print("-------------------------EVALUATING SARSA--------------------------")
    wins, losses, rewards = evaluate_policy(env, policy, discretizer, runs=EVAL)
    print("Total wins: " + str(wins) + " out of: " + str(EVAL / 100))

    # plot total reward per episode
    plt.figure(2)
    plt.title("Testing: Sarsa Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    r_line, = plt.plot(rewards, label="Reward")
    plt.legend(handles=[r_line])
    plt.show()


    policy = init_e_soft_policy(NUM_STATES, num_actions)
    print("-----------------------TRAINING Q LEARNING-------------------------")
    policy, reward_per_episode = q_learning(env, discretizer, policy, NUM_STATES, num_actions, episodes=EPISODES, eps=E, alpha=A)

    # plot total reward per episode
    plt.figure(3)
    plt.title("Training: Q Learning Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    r_line, = plt.plot(reward_per_episode, label="Reward")
    plt.legend(handles=[r_line])
    plt.show()

    print("----------------------EVALUATING Q LEARNING------------------------")
    wins, losses, rewards = evaluate_policy(env, policy, discretizer, runs=EVAL)
    print("Q Learning total wins: " + str(wins) + " out of: " + str(EVAL / 100))

    # plot total reward per episode
    plt.figure(4)
    plt.title("Testing: Q Learning Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    r_line, = plt.plot(rewards, label="Reward")
    plt.legend(handles=[r_line])
    plt.show()


    env.close()

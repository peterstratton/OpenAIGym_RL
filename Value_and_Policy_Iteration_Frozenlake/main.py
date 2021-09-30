import gym
import numpy as np
from policy_iteration import *
from value_iteration import *

RUNS = 1000 # how many times to test policy
GAMMA = 1 # discount factor

def evaluate_policy(env, policy, render=False, runs=100):
    """
    Function that runs a given policy on a Open AI gym environment and reports
    statistics.

    Parameters
    ----------
    arg1 : Open AI gym (gym.wrappers.time_limit.TimeLimit)
        Environment to run the policy in
    arg2 : numpy.ndarray
        Policy that maps states to actions
    arg3 : boolean
        To render the environment or not
    arg4 : int
        Number of times to run the policy to completion

    Returns
    -------
    int
        Number of successful environment completions
    losses
        Number of unsuccessful environment completions
    scores
        Total reward of each run
    """
    wins = 0
    losses = 0
    scores = []
    for i in range(runs):
        observation = env.reset()
        done = False
        t = 0
        reward_total = 0
        while not done:
            if render:
                env.render()

            # make actions in the environment
            action = int(policy[observation])
            observation, reward, done, info = env.step(action)
            reward_total += reward

        # sum wins and losses
        scores.append(reward_total)
        if reward == 1:
            wins += 1
        else:
            losses += 1

    return wins, losses, scores


if __name__ == "__main__":
    # setup gym environment
    env = gym.make('FrozenLake8x8-v0')
    env.reset()

    # get necessary environment variables
    num_states = env.nS
    num_actions = env.nA
    state_transition = env.P

    print("--------------------Executing Policy Iteration---------------------")
    policy = policy_iteration(state_transition, num_states, num_actions, \
                              gamma=GAMMA)
    print("Optimal policy: " + str(policy))

    wins, losses, scores = evaluate_policy(env, policy, render=False, runs=RUNS)

    # display results
    print("-------------------------------------------------------------------")
    print("Policy resulted in winning: " + str((wins / float(RUNS)) * 100) + \
          " percent of the time")
    print("Policy resulted in losing: " + str((losses / float(RUNS)) * 100) + \
          " percent of the time")
    print("-------------------------------------------------------------------")
    print("Mean score = %0.2f" %(np.mean(np.array(scores))))
    print("-------------------------------------------------------------------")
    env.reset()


    print("--------------------Executing Value Iteration----------------------")
    values = value_iteration(state_transition, num_states, num_actions, \
                             gamma=GAMMA)
    policy = obtain_policy(values, state_transition, num_states, num_actions, \
                           gamma=GAMMA)
    print("Optimal policy: " + str(policy))

    wins, losses, scores = evaluate_policy(env, policy, render=False, runs=RUNS)

    # display results
    print("-------------------------------------------------------------------")
    print("Policy resulted in winning: " + str((wins / float(RUNS)) * 100) + \
          " percent of the time")
    print("Policy resulted in losing: " + str((losses / float(RUNS)) * 100) + \
          " percent of the time")
    print("-------------------------------------------------------------------")
    print("Mean score = %0.2f" %(np.mean(np.array(scores))))
    print("-------------------------------------------------------------------")
    env.close()

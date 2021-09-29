import gym
import numpy as np
from utils import evaluate_policy
from on_policy import on_policy_first_visit_control

RUNS = 10000 # how many times to test policy
GAMMA = 1 # discount factor
EPS = 0.3 # e value

if __name__ == "__main__":
    # setup gym environment
    env = gym.make('Blackjack-v0')
    env.reset()

    print("-----------------------------TRAINING-----------------------------")
    policy = on_policy_first_visit_control(env, gamma=GAMMA, eps=EPS)
    print("Optimal policy: \n" + str(policy))

    print("----------------------------EVALUATING-----------------------------")
    wins, losses, scores = evaluate_policy(env, policy, runs=RUNS)
    # display results
    print("Policy resulted in winning: " + str((wins / float(RUNS)) * 100) + \
          " percent of the time")
    print("Policy resulted in losing: " + str((losses / float(RUNS)) * 100) + \
          " percent of the time")
    print("-------------------------------------------------------------------")
    env.close()

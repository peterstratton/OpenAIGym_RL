import sys
import numpy as np
from utils import generate_episode, init_e_soft_policy, init_q_values

EPISODES = 1000000

def on_policy_first_visit_control(env, gamma=1.0, eps=0.01):
    # init policy and dicts for calculating q values
    policy = init_e_soft_policy(env)
    q_values = init_q_values(env)
    returns = {}
    num_visits = {}

    for _ in range(EPISODES):
        if _ % 1000 == 0:
            print("Training Episode: " + str(_) + " out of " + str(EPISODES))
        episode = generate_episode(policy, env)

        g = 0
        rev_episode = episode[::-1]
        for i in range(len(rev_episode)):
            state = rev_episode[i][0]
            action = rev_episode[i][1]
            reward = rev_episode[i][2]

            g = gamma * g + reward
            if i + 1 >= len(rev_episode) or state not in rev_episode[i + 1:,0] and action not in rev_episode[i + 1:,1]:
                if (state, action) not in returns:
                    returns[(state, action)] = g
                else:
                    returns[(state, action)] += g
                if (state, action) not in num_visits:
                    num_visits[(state, action)] = 1
                else:
                    num_visits[(state, action)] += 1

                # # print(state)
                # if state == (15, 10, False):
                #     if (state, 0) in num_visits and (state, 1) in num_visits:
                #         print("stick: " + str(q_values[(state, 0)]) + " num visits: " + str(num_visits[(state, 0)]))
                #         print("hit: " + str(q_values[(state, 1)]) + " num visits: " + str(num_visits[(state, 1)]))
                q_values[(state, action)] = returns[(state, action)] / num_visits[(state, action)]

                # get the best action
                best_a = 0
                max = -sys.maxsize - 1
                for j in range(env.action_space.n):
                    if max < q_values[(state, j)]:
                        max = q_values[(state, j)]
                        best_a = j

                # set action probabilities
                for j in range(len(policy[state])):
                    if j == best_a:
                        policy[state][j] = 1 - eps + (eps / len(policy[state]))
                    else:
                        policy[state][j] = eps / (len(policy[state]))

    return policy

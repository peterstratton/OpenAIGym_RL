import numpy as np
from utils import derive_policy

def sarsa(env, discretizer, policy, num_states, num_actions, episodes=1000, gamma=1.0, eps=0.1, alpha=0.1, render=False):
    reward_per_episode = []

    actions = [i for i in range(num_actions)]
    q_table = discretizer.init_q_table(num_states, num_actions)

    for epi in range(episodes):
        if epi % 100 == 0:
            print("Episode: " + str(epi) + " out of " + str(episodes))
        total_reward = 0

        # discetize state
        state = env.reset()
        state = discretizer.discretize(state)

        # create action probability list
        derive_policy(policy, q_table, state, num_actions, eps)
        prob = [policy[state[0], state[1], state[2], state[3], i] for i in range(num_actions)]
        action = int(np.random.choice(actions, 1, p=prob))

        done = False
        while not done:
            # optional environment rendering
            if render:
                env.render()

            # get current state action q value
            cur_q = q_table[state[0], state[1], state[2], state[3], action]

            # take current action to get to the next state
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # discetize next state and determine next action
            next_state = discretizer.discretize(next_state)
            derive_policy(policy, q_table, next_state, num_actions, eps)
            prob = [policy[next_state[0], next_state[1], next_state[2], next_state[3], i] for i in range(num_actions)]
            next_action = int(np.random.choice(actions, 1, p=prob))

            # perform q value update
            next_q = q_table[next_state[0], next_state[1], next_state[2], next_state[3], next_action]
            q_table[state[0], state[1], state[2], state[3], action] = cur_q + alpha * (reward + gamma * next_q - cur_q)

            # set next state to be the current state for the next iteration
            state = next_state
            action = next_action

        reward_per_episode.append(total_reward)

    return policy, reward_per_episode


def q_learning(env, discretizer, policy, num_states, num_actions, episodes=1000, gamma=1.0, eps=0.1, alpha=0.1, render=False):
    reward_per_episode = []

    actions = [i for i in range(num_actions)]
    q_table = discretizer.init_q_table(num_states, num_actions)

    for epi in range(episodes):
        if epi % 100 == 0:
            print("Episode: " + str(epi) + " out of " + str(episodes))
        total_reward = 0

        # discetize state
        state = env.reset()
        state = discretizer.discretize(state)

        done = False
        while not done:
            # optional environment rendering
            if render:
                env.render()

            # determine action
            derive_policy(policy, q_table, state, num_actions, eps)
            prob = [policy[state[0], state[1], state[2], state[3], i] for i in range(num_actions)]
            action = int(np.random.choice(actions, 1, p=prob))

            # get current state action q value
            cur_q = q_table[state[0], state[1], state[2], state[3], action]

            # take current action to get to the next state
            next_state, reward, done, info = env.step(action)
            next_state = discretizer.discretize(next_state)
            total_reward += reward

            # determine action corresponding to the max q value
            max_q = 0
            for i in range(num_actions):
                if q_table[next_state[0], next_state[1], next_state[2], next_state[3], i] > max_q:
                    max_q = q_table[next_state[0], next_state[1], next_state[2], next_state[3], i]

            # perform q value update
            q_table[state[0], state[1], state[2], state[3], action] = cur_q + alpha * (reward + gamma * max_q - cur_q)

            # set next state to be the current state for the next iteration
            state = next_state

        reward_per_episode.append(total_reward)

    return policy, reward_per_episode

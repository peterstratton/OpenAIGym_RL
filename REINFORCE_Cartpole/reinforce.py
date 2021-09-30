import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
from utils import policy, generate_episode


def reinforce(env, gamma=0.99, learning_rate=0.01, episodes=1000, render=False):
    # reset environment
    state = env.reset()

    # init policy and optimizer
    policy_net = policy(classes = env.action_space.n)
    best_policy = policy_net
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    reward_per_episode = []
    best_reward = 0
    for _ in range(episodes):
        if _ % 100 == 0:
            print("Episode: " + str(_) + " out of " + str(episodes))

        # generate episode
        episode, steps = generate_episode(env, policy_net)

        reward_per_episode.append(steps)
        # reset gradients to zero
        optimizer.zero_grad()
        for i in range(len(episode)):
            state = episode[i][0]
            action = episode[i][1]

            # sum return
            r_return = 0
            for j in range(i, len(episode)):
                r_return += episode[i][2]

            probs = policy_net(state)
            m = Bernoulli(probs)

            # calculate gradient
            loss = -m.log_prob(action) * r_return
            loss.backward()

        # save the policy that achieves the maximum reward
        if steps >= best_reward:
            best_reward = steps
            best_policy = policy_net

        # update parameters
        optimizer.step()


    return best_policy, reward_per_episode

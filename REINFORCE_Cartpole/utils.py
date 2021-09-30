import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli


class policy(nn.Module):

    def __init__(self, classes):
        super(policy, self).__init__()
        self.lin1 = nn.Linear(4, 15)
        self.re1 = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(15, 15)
        self.re2 = nn.ReLU(inplace=True)
        self.lin3 = nn.Linear(15, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.re1(self.lin1(x))
        x = self.re2(self.lin2(x))
        x = self.sig(self.lin3(x))
        return x


def generate_episode(env, policy_net):
    episode = []

    done = False
    state = env.reset()
    steps = 0
    while not done:
        # determine action
        state = torch.from_numpy(state).float()
        probs = policy_net(state)
        m = Bernoulli(probs)
        action = m.sample()

        # step environment and save episode results
        next_state, reward, done, info = env.step(action.data.numpy().astype(int)[0])
        episode.append((state, action, reward))

        state = next_state
        steps += 1

    return episode, steps


def evaluate_policy(env, policy_net, render=False, runs=1000):
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
        if i % 100 == 0:
            print("Run: " + str(i) + " out of " + str(runs))

        state = env.reset()
        done = False
        t = 0
        reward_total = 0
        while not done:
            if i % 100 == 0:
                win = True
                if render:
                    env.render()

            # determine action
            probs = policy_net(torch.from_numpy(state).float())
            m = Bernoulli(probs)
            action = m.sample()

            # make actions in the environment
            next_state, reward, done, info = env.step(action.data.numpy().astype(int)[0])
            state = next_state
            reward_total += reward

        # sum wins and losses
        scores.append(reward_total)

        if reward_total > 195:
            wins += 1

    return wins, scores

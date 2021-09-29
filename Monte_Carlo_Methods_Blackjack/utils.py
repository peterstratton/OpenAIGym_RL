import numpy as np

def init_e_soft_policy(env):
    policy = {}
    for i in range(env.observation_space[0].n):
        for j in range(env.observation_space[1].n):
            for k in range(env.observation_space[2].n):
                bool = False
                if k == 0:
                    bool = True
                policy[(i, j, bool)] = [float(1 / env.action_space.n) for _ in range(env.action_space.n)]
    return policy


def init_q_values(env):
    q_values = {}
    for i in range(env.observation_space[0].n):
        for j in range(env.observation_space[1].n):
            for k in range(env.observation_space[2].n):
                bool = False
                if k == 0:
                    bool = True
                for m in range(env.action_space.n):
                    q_values[((i, j, bool), m)] = 0
    return q_values
    

def generate_episode(policy, env):
    episode = []
    actions = [i for i in range(env.action_space.n)]

    done = False
    state = env.reset()
    while not done:
        action = int(np.random.choice(actions, 1, p=policy[state]))
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return np.array(episode)


def evaluate_policy(env, policy, render=False, runs=1000):
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
    actions = [i for i in range(env.action_space.n)]
    for i in range(runs):
        state = env.reset()
        done = False
        t = 0
        reward_total = 0
        while not done:
            if render:
                env.render()

            # make actions in the environment
            action = int(np.random.choice(actions, 1, p=policy[state]))
            next_state, reward, done, info = env.step(action)
            state = next_state
            reward_total += reward

        # sum wins and losses
        scores.append(reward_total)
        if reward == 1:
            wins += 1
        else:
            losses += 1

    return wins, losses, scores

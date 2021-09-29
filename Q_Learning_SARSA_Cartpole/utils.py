import numpy as np


class Discretizer:

    def __init__(self, env, num_states, num_actions, max_v=2):
        self.low = env.observation_space.low
        self.low[1] = -max_v
        self.low[3] = -max_v
        self.high = env.observation_space.high
        self.high[1] = max_v
        self.high[3] = max_v
        self.inc = np.zeros((self.low.shape))
        self.num_states = num_states
        print("High state: " + str(self.high) + " Low state: " + str(self.low))

        for i in range(self.low.shape[0]):
            if i % 2 == 0:
                self.inc[i] = (self.high[i] - self.low[i]) / num_states
            else:
                self.inc[i] = (max_v * 2) / num_states
        self.q_table = np.zeros((num_states, num_states, num_states, num_states, num_actions))

    def discretize(self, obs):
        o_ind = np.zeros((obs.shape), dtype=int)
        for i in range(obs.shape[0]):
            j = self.low[i]
            ind = 0
            set = False
            while (j < self.high[i]):
                if j <= obs[i] and (j + self.inc[i]) >= obs[i]:
                    set = True
                    o_ind[i] = ind
                ind += 1
                j += self.inc[i]
            if not set or o_ind[i] == self.num_states:
                o_ind[i] = self.num_states - 1

        return o_ind

    def init_q_table(self, num_states, num_actions):
        return np.zeros((num_states, num_states, num_states, num_states, num_actions))


def init_e_soft_policy(num_states, num_actions):
    return np.full((num_states, num_states, num_states, num_states, num_actions), float(1 / num_actions))


def derive_policy(policy, q_table, state, num_actions, eps):
    q_vals = []
    for a in range(num_actions):
        q_vals.append(q_table[state[0], state[1], state[2], state[3], a])
    e_ind = np.argmax(np.array(q_vals))

    policy[state[0], state[1], state[2], state[3], 0:e_ind] = eps / num_actions
    policy[state[0], state[1], state[2], state[3], e_ind] = 1 - eps + (eps / num_actions)
    policy[state[0], state[1], state[2], state[3], e_ind + 1:num_actions] = eps / num_actions


def evaluate_policy(env, policy, discretizer, render=False, runs=1000):
    wins = 0
    losses = 0
    scores = []
    num_actions = env.action_space.n
    actions = [i for i in range(num_actions)]
    for i in range(runs):
        if i % 100 == 0:
            win = True

        state = env.reset()
        state = discretizer.discretize(state)
        done = False
        t = 0
        reward_total = 0
        while not done:
            if render:
                env.render()

            # make actions in the environment
            state = discretizer.discretize(state)
            prob = [policy[state[0], state[1], state[2], state[3], i] for i in range(num_actions)]
            action = int(np.random.choice(actions, 1, p=prob))
            next_state, reward, done, info = env.step(action)
            state = next_state
            reward_total += reward

        # sum wins and losses
        scores.append(reward_total)

        if reward_total < 195:
            win = False

        if i != 0 and i % 100 == 0:
            if win:
                wins += 1
            else:
                losses += 1

    return wins, losses, scores

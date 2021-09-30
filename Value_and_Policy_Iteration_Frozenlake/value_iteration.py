import gym
import numpy as np


def value_iteration(state_transition, num_states, num_actions, gamma=0.9):
    """
    Function that implements the value iteration algorithm. It determines the
    optimal values of environment states by setting the value of a state to
    equal to the weighted sum of of the values of the next states in the
    environment. The set of next states used to set the value of the current
    state is determined by the action that obtains the greatest weighted sum of
    next state values.

    Parameters
    ----------
    arg1 : dict
        Dictionary of lists, where
        state_transition[state][action] == [(probability, nextstate, reward,
                                             done), ...]
    arg2 : int
        Number of states in the environment
    arg3 : int
        Number of actions in the environment
    arg4 : float
        Discount variable

    Returns
    -------
    numpy.ndarray
        The optimal values for the environment
    """
    # init values
    values = np.zeros(num_states)

    # determines when the values have converged
    thres = 0.001

    delta = 1
    while delta > thres:
        delta = 0

        # iterate through each state
        for state in range(num_states):
            m_value = -1

            # determine the maximum value
            for action in range(num_actions):
                prev_v = values[state]
                value = 0
                for p_ns, n_state, reward, _ in state_transition[state][action]:
                    value += p_ns * (reward + gamma * values[n_state])

                if value > m_value:
                    m_value = value

            values[state] = m_value
            delta = max(delta, np.abs(values[state] - prev_v))

    return values


def obtain_policy(values, state_transition, num_states, num_actions, gamma=0.9):
    """
    Function that obtains a policy using the optimal values of of the
    environment. It detrermines the policy by setting the action of a state to
    the action that results in the largest set of next state values.

    Parameters
    ----------
    arg1: numpy.ndarray
        Optimal values of the environment
    arg2 : dict
        Dictionary of lists, where
        state_transition[state][action] == [(probability, nextstate, reward,
                                             done), ...]
    arg3 : int
        Number of states in the environment
    arg4 : int
        Number of actions in the environment
    arg5 : float
        Discount variable

    Returns
    -------
    numpy.ndarray
        The optimal policy for the environment
    """
    # init policy
    policy = np.zeros(num_states)

    # iterate through each state
    for state in range(num_states):
        max = 0
        for action in range(num_actions):
            value = 0
            for p_ns, n_state, reward, _ in state_transition[state][action]:
                value += p_ns * (reward + gamma * values[n_state])

            # update best action
            if max < value:
                policy[state] = action
                max = value

    return policy

import gym
import numpy as np


def policy_iteration(state_transition, num_states, num_actions, gamma=0.9):
    """
    Function that implements the policy iteration algorithm. It initializes an
    arbitrary policy, updates the values of the environment states using the
    policy, then uses the values to improve the policy.

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
        The optimal policy for the environment
    """

    # init state values and policy
    values = np.zeros(num_states)
    policy = np.zeros(num_states)

    # loop until policy converges
    stable = False
    while not stable:
        values = policy_evaluation(policy, values, state_transition, \
                                   num_states, num_actions, gamma=gamma)
        policy, stable = policy_improvement(policy, values, state_transition, \
                                           num_states, num_actions, gamma=gamma)

    return policy


def policy_evaluation(policy, values, state_transition, num_states, \
                      num_actions, gamma=0.9):
    """
    Function that evaluates a policy by determining the values of the
    environment states using the policy.

    Parameters
    ----------
    arg1 : numpy.ndarray
        Policy that dictates the action to be made from a given state
    arg2 : numpy.ndarray
        Values of the states under the given policy
    arg3 : dict
        Dictionary of lists, where
        state_transition[state][action] == [(probability, nextstate, reward,
                                             done), ...]
    arg4 : int
        Number of states in the environment
    arg5 : int
        Number of actions in the environment
    arg6 : float
        Discount variable

    Returns
    -------
    numpy.ndarray
        New values of each state
    """
    # threshold that defines when values have converged
    thres = 0.001

    # loop till convergence
    delta = 1
    while delta > thres:
        delta = 0

        # iterate through each state
        for state in range(num_states):
            prev_v = values[state]
            action = policy[state]

            # sum weighted values from next states
            value = 0
            for p_ns, n_state, reward, _ in state_transition[state][action]:
                value += p_ns * (reward + gamma * values[n_state])

            values[state] = value
            delta = max(delta, np.abs(values[state] - prev_v))

    return values


def policy_improvement(policy, values, state_transition, num_states, \
                       num_actions, gamma=0.9):
    """
    Function that improves the given policy by altering its action from a state
    to go to the next state with the largest value.

    Parameters
    ----------
    arg1 : numpy.ndarray
        Policy that dictates the action to be made from a given state
    arg2 : numpy.ndarray
        Values of the states under the given policy
    arg3 : dict
        Dictionary of lists, where
        state_transition[state][action] == [(probability, nextstate, reward,
                                             done), ...]
    arg4 : int
        Number of states in the environment
    arg5 : int
        Number of actions in the environment
    arg6 : float
        Discount variable

    Returns
    -------
    numpy.ndarray
        New policy that now gives actions that maximize the expected reward,
        given the current values
    boolean
        If the policy was changed or not
    """
    stable = True

    # iterate through states to improve the policy
    for state in range(num_states):
        prev_a = policy[state]

        # determine actions that give the highest value
        max = 0
        for action in range(num_actions):
            value = 0
            for p_ns, n_state, reward, _ in state_transition[state][action]:
                value += p_ns * (reward + gamma * values[n_state])

            # update best action
            if max < value:
                policy[state] = action
                max = value

        if policy[state] != prev_a:
            stable = False

    return policy, stable

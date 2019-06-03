### MDP Value Iteration and Policy Iteration
### Acknowledgement: start-up codes were adapted with permission from Prof. Emma Brunskill of Stanford University

import numpy as np
import gym
import time
import rmit_rl_env
import colorama

colorama.init()
np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
		max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	############################
	# YOUR IMPLEMENTATION HERE #
	valueFunction = np.zeros(nS)
	newValueFunction = valueFunction.copy()
	maxIterations = 100
	iterCounter = 0 #Counts the amount of iterations we have evaluated over.
	while iterCounter<=maxIterations or getValueFunctionDiff(newValueFunction, valueFunction)>tol:
		#Keeps looping until we have hit the max iteration limit, or until we have converged, and the difference is under the tolerance.
		iterCounter += 1
		valueFunction = newValueFunction.copy()
		#For each state.
		for s in range(nS):
			result = P[s][policy[s]] #Retrieves the possibilities
			newValueFunction[s] = np.array(result)[:,2].mean()
			#For each possibility
			for num in range(len(result)):
				(probability, nextstate, reward, terminal) = result[num]
				newValueFunction[s] += (gamma * probability * valueFunction[nextstate]) #Updates the new value for state, s.
	############################
	return newValueFunction

def getValueFunctionDiff(newValueFunction, oldValueFunction):
	"""Returns the difference betwene the two value functions.
		Used to check if the difference has fallen below the tolerance.

	Returns
	-------
	value_function_difference: float
		The difference between the two value functions.
	"""
	return np.sum(newValueFunction-oldValueFunction)

def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.
    Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

	newPolicy = np.zeros(nS, dtype='int')

	############################
	# YOUR IMPLEMENTATION HERE #
	qFunction = np.zeros([nS,nA])
	#For each state index.
	for s in range(nS):
		#For each action index.
		for a in range(nA):
			result = P[s][a] #Retrieves the possbilities for this action.
			#For each possibility
			for num in range(len(result)):
				#Seperate out each variable.
				(probability, nextstate, reward, terminal) = result[num]
				qFunction[s][a] += reward + (gamma*probability*value_from_policy[nextstate]) #Update the q_function value with the new reward value.
	newPolicy = np.argmax(qFunction, axis=1) #Creates a policy from the maximum q_function value.

	############################
	return newPolicy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3, i_max=100):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    
    i = 0 
    new_policy= policy.copy()
    while i <= i_max or getValueFunctionDiff(new_policy, policy) > tol:
        i += 1
        policy = new_policy
        value_function = policy_evaluation(P, nS, nA, policy)
        new_policy = policy_improvement(P, nS, nA, value_function, policy)
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3, i_max=100):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    i_max: int
        Maximum number of iterations
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    
    value_function = np.zeros(nS)

    policy = np.zeros(nS, dtype=int)

    for i in range(i_max):
        # Iterating over every state
        for state in range(nS):
            r_max = -1
            a_max = 0
            # Iterating over every action
            for action in range(nA):
                r = P[state][action]
                r_current = np.array(r)[:,2].mean()
                length_r = len(r)
                # Iterating over every probabilty
                for j in range(length_r):
                    (prob, next_state, reward, terminal) = r[j]
                    r_current += gamma * prob * value_function[next_state]
                    if r_current > r_max:
                        a_max = action
                        r_max = r_current    

            value_function[state] = r_max
            policy[state] = a_max

    return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render()
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
      print("Episode reward: %f" % episode_reward)


if __name__ == "__main__":

    #env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3, i_max=20)
    render_single(env, p_pi, 100)

    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    # V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3,i_max=100)
    # render_single(env, p_vi, 100)



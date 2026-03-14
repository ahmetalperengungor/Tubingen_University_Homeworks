import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import agent as agent_module
import gridworld
import util
import random

def run_episode(env, agent, discount):
    env.reset()
    total_return = 0.0
    total_discount = 1.0
    while True:
        state = env.getCurrentState()
        actions = env.getPossibleActions(state)
        if len(actions) == 0:
            break
        action = agent.getAction(state)
        next_state, reward = env.doAction(action)
        total_return += reward * total_discount
        total_discount *= discount
    return total_return

def question_2a():
    mdp = gridworld.getMazeGrid()
    values = util.Counter()
    curentStartState = mdp.getStartState()
    for iteration in range(1, 101):
        newValues = util.Counter()
        for state in mdp.getStates():
            if mdp.isTerminal(state):
                newValues[state] = 0.0
            else:
                qValues = []
                for action in mdp.getPossibleActions(state):
                    qValue = 0.0
                    for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                        reward = mdp.getReward(state, action, nextState)
                        qValue += prob * (reward + 0.9 * values.getCount(nextState))
                    qValues.append(qValue)
                if qValues:
                    newValues[state] = max(qValues)
                else:
                    newValues[state] = 0.0
        values = newValues
        currentStartValue = values.getCount(curentStartState)
        if currentStartValue > 0:
            print(f"First non-zero is after {iteration} iterations")
            print(f"This non-zero value at iteration {iteration}: {currentStartValue:.6f}")
            break

def question_2c():
    test_cases = [
        (0.1, 0.0, 0.0, "Low discount factor with no noise"),
        (0.9, 0.0, 0.0, "High discount factor with no noise"),
        (0.9, 0.2, 0.0, "High discount factor with noise"),
        (0.1, 0.2, 0.0, "Low discount factor with noise"),
    ]
    for discount, noise, living_reward, description in test_cases:
        mdp = gridworld.getDiscountGrid()
        mdp.setNoise(noise)
        mdp.setLivingReward(living_reward)
        agent = agent_module.ValueIterationAgent(mdp, discount, 100)
        currentStartState = mdp.getStartState()
        policy = agent.getPolicy(currentStartState)
        target, riskiness, _ = helper_question_2c(mdp, agent)
        print(f"{description}: First action={policy}, Target={target}, Risk={riskiness}")

def helper_question_2c(mdp, agent):
    currentStartState = mdp.getStartState()
    first_action = agent.getPolicy(currentStartState)
    riskiness = 'risk' if first_action == 'east' else 'avoid'
    state = currentStartState
    for _ in range(50):
        row, col = state
        cell = mdp.grid[row][col]
        if isinstance(cell, (int, float)):
            if cell == 1:
                return 'close', riskiness, first_action
            if cell == 10:
                return 'distant', riskiness, first_action
            return 'none', riskiness, first_action
        action = agent.getPolicy(state)
        if action is None:
            break
        next_state = helper_helper_question_2c(mdp, state, action)
        if next_state == state:
            break
        state = next_state
    return 'none', riskiness, first_action

def helper_helper_question_2c(mdp, state, action):
    successors = mdp.getTransitionStatesAndProbs(state, action)
    if not successors:
        return state
    return max(successors, key=lambda x: x[1])[0]

def question_2d():
    mdp = gridworld.getMazeGrid()
    agent = agent_module.ValueIterationAgent(mdp, 0.9, 100)
    currentStartState = mdp.getStartState()
    value_estimate = agent.getValue(currentStartState)
    env = gridworld.GridworldEnvironment(mdp)
    returns = []
    for episode in range(10000):
        episode_return = run_episode(env, agent, 0.9)
        returns.append(episode_return)
    empirical_avg = sum(returns) / len(returns)
    print(f"Value iteration estimate value: {value_estimate:.6f}")
    print(f"Empirical average return value: {empirical_avg:.6f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "2a":
            question_2a()
        elif sys.argv[1] == "2c":
            question_2c()
        elif sys.argv[1] == "2d":
            question_2d()
        else:
            print("Wrong probably")
    else:
        question_2a()
        question_2c()
        question_2d()
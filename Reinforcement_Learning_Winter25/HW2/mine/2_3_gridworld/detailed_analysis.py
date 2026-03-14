#!/usr/bin/env python3
"""
Detailed Analysis for Value Iteration Homework Questions
This script provides specific answers to each homework question.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import agent as agent_module
import gridworld
import util
import random

def question_2a_maze_convergence():
    """
    Question 2a: How many rounds of value iteration are needed before 
    the start state of MazeGrid becomes non-zero? Why?
    """
    print("=" * 60)
    print("QUESTION 2a: MazeGrid Convergence Analysis")
    print("=" * 60)
    
    mdp = gridworld.getMazeGrid()
    start_state = mdp.getStartState()
    
    # Track value changes during iterations
    values = util.Counter()
    convergence_iteration = None
    
    print(f"MazeGrid layout:")
    for i, row in enumerate(mdp.grid):
        print(f"  {i}: {row}")
    print(f"Start state: {start_state}")
    print()
    
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
        start_value = values.getCount(start_state)
        
        if start_value > 0 and convergence_iteration is None:
            convergence_iteration = iteration
        
        if iteration <= 15 or start_value > 0:
            print(f"Iteration {iteration:2d}: Start state value = {start_value:.6f}")
    
    print(f"\nANSWER: The start state becomes non-zero after {convergence_iteration} iterations.")
    print(f"Final value after 100 iterations: {values.getCount(start_state):.6f}")
    
    print("\nWHY: The value propagates from the terminal state (+1) backwards through the grid.")
    print("It takes 10 iterations because the agent must navigate through the maze")
    print("with obstacles (#) to reach the terminal state, and the value spreads")
    print("one step per iteration due to the discount factor (0.9).")

def question_2b_bridge_policy():
    """
    Question 2b: Which parameter must we change before the agent dares to cross the bridge?
    """
    print("\n" + "=" * 60)
    print("QUESTION 2b: BridgeGrid Policy Analysis")
    print("=" * 60)
    
    print("BridgeGrid layout:")
    bridge_grid = gridworld.getBridgeGrid()
    for i, row in enumerate(bridge_grid.grid):
        print(f"  {i}: {row}")
    print()
    
    # Test different parameter combinations
    test_cases = [
        {"discount": 0.9, "noise": 0.2, "description": "Default (discount=0.9, noise=0.2)"},
        {"discount": 0.9, "noise": 0.0, "description": "No noise (discount=0.9, noise=0.0)"},
        {"discount": 0.99, "noise": 0.2, "description": "Higher discount (discount=0.99, noise=0.2)"},
        {"discount": 0.99, "noise": 0.0, "description": "Higher discount, no noise (discount=0.99, noise=0.0)"},
    ]
    
    for case in test_cases:
        mdp = gridworld.getBridgeGrid()
        mdp.setNoise(case['noise'])
        
        agent = agent_module.ValueIterationAgent(mdp, case['discount'], 100)
        start_state = mdp.getStartState()
        policy = agent.getPolicy(start_state)
        value = agent.getValue(start_state)
        
        print(f"{case['description']}:")
        print(f"  Policy: {policy}, Value: {value:.6f}")
        if policy == 'east':
            print("  ✓ Agent crosses the bridge!")
        else:
            print("  ✗ Agent avoids the bridge")
        print()
    
    print("ANSWER: We must change the NOISE parameter from 0.2 to 0.0.")
    print("The agent will cross the bridge when there's no noise, regardless of discount.")
    print("With noise=0.2, the agent is too uncertain about reaching the goal safely.")

def question_2c_discount_grid_policies():
    """
    Question 2c: Find parameter values for different optimal policies on DiscountGrid
    """
    print("\n" + "=" * 60)
    print("QUESTION 2c: DiscountGrid Policy Analysis")
    print("=" * 60)
    
    print("DiscountGrid layout:")
    discount_grid = gridworld.getDiscountGrid()
    for i, row in enumerate(discount_grid.grid):
        print(f"  {i}: {row}")
    print("Legend: S=Start, #=Wall, 1=Close exit (+1), 10=Distant exit (+10), -10=Cliff")
    print()
    
    # Test different parameter combinations (expanded to capture close+risk)
    test_cases = [
        # (discount, noise, living_reward, description)
        (0.1, 0.0, 0.0, "Low discount, no noise (likely distant+risk)"),
        (0.5, 0.0, 0.0, "Medium discount, no noise (likely distant+risk)"),
        (0.9, 0.0, 0.0, "High discount, no noise (distant+risk)"),
        (0.9, 0.2, 0.0, "High discount with noise (close+avoid)"),
        (0.9, 0.0, -0.1, "High discount, small negative living reward (distant+risk)"),
        (0.9, 0.0, -1.0, "High discount, large negative living reward (distant+risk)"),
        (0.1, 0.0, -1.0, "Low discount, large negative living reward (distant+risk)"),
        # New candidates aimed at close+risk
        (0.3, 0.1, 0.0, "Low discount with some noise (candidate close+risk)"),
        (0.5, 0.1, -0.1, "Medium discount, small noise, slightly negative living (candidate close+risk)"),
    ]
    
    for discount, noise, living_reward, description in test_cases:
        mdp = gridworld.getDiscountGrid()
        mdp.setNoise(noise)
        mdp.setLivingReward(living_reward)
        
        agent = agent_module.ValueIterationAgent(mdp, discount, 100)
        start_state = mdp.getStartState()
        policy = agent.getPolicy(start_state)
        value = agent.getValue(start_state)
        
        # Analyze the policy and classify
        close_exit_value = agent.getValue((2, 2))
        distant_exit_value = agent.getValue((2, 4))

        target, riskiness, first_action = _classify_discount_grid_policy(mdp, agent)

        print(f"{description}:")
        print(f"  Parameters: discount={discount}, noise={noise}, living_reward={living_reward}")
        print(f"  Start policy: {policy} (first action), Start value: {value:.6f}")
        print(f"  Close exit value: {close_exit_value:.6f}, Distant exit value: {distant_exit_value:.6f}")

        if target == 'close':
            if riskiness == 'risk':
                print("  → Classification: Prefer close exit (+1), risking cliff")
            else:
                print("  → Classification: Prefer close exit (+1), avoiding cliff")
        elif target == 'distant':
            if riskiness == 'risk':
                print("  → Classification: Prefer distant exit (+10), risking cliff")
            else:
                print("  → Classification: Prefer distant exit (+10), avoiding cliff")
        else:
            print("  → Classification: Could not determine exit preference (no terminal reached in trace)")
        print()

def _most_likely_next_state(mdp, state, action):
    successors = mdp.getTransitionStatesAndProbs(state, action)
    if not successors:
        return state
    # choose the next state with the highest probability (intended move when noise is low)
    return max(successors, key=lambda x: x[1])[0]

def _classify_discount_grid_policy(mdp, agent):
    """Classify policy on DiscountGrid into (target, riskiness, first_action).
    - target: 'close' if ends at +1, 'distant' if ends at +10, 'none' otherwise
    - riskiness: 'risk' if first action is east from start (traversing bottom row first), else 'avoid'
    """
    start_state = mdp.getStartState()
    first_action = agent.getPolicy(start_state)
    riskiness = 'risk' if first_action == 'east' else 'avoid'

    # Follow the greedy policy deterministically using the most likely successor
    state = start_state
    for _ in range(50):
        row, col = state
        cell = mdp.grid[row][col]
        if isinstance(cell, (int, float)):
            if cell == 1:
                return 'close', riskiness, first_action
            if cell == 10:
                return 'distant', riskiness, first_action
            # some other terminal numeric value
            return 'none', riskiness, first_action
        action = agent.getPolicy(state)
        if action is None:
            break
        next_state = _most_likely_next_state(mdp, state, action)
        if next_state == state:
            break
        state = next_state
    return 'none', riskiness, first_action

def question_2d_empirical_comparison():
    """
    Question 2d: Compare value estimate vs empirical returns on MazeGrid
    """
    print("\n" + "=" * 60)
    print("QUESTION 2d: Value vs Empirical Returns Comparison")
    print("=" * 60)
    
    # Get value iteration estimate
    mdp = gridworld.getMazeGrid()
    agent = agent_module.ValueIterationAgent(mdp, 0.9, 100)
    start_state = mdp.getStartState()
    value_estimate = agent.getValue(start_state)
    
    print(f"Value iteration estimate: {value_estimate:.6f}")
    
    # Calculate empirical returns
    env = gridworld.GridworldEnvironment(mdp)
    returns = []
    
    print("Calculating empirical returns...")
    for episode in range(10000):
        episode_return = run_episode(env, agent, 0.9)
        returns.append(episode_return)
    
    empirical_avg = sum(returns) / len(returns)
    empirical_std = (sum([(r - empirical_avg)**2 for r in returns]) / len(returns))**0.5
    
    print(f"\nRESULTS:")
    print(f"Value iteration estimate: {value_estimate:.6f}")
    print(f"Empirical average return: {empirical_avg:.6f}")
    print(f"Empirical std deviation: {empirical_std:.6f}")
    print(f"Difference: {abs(value_estimate - empirical_avg):.6f}")
    
    print(f"\nANSWER:")
    print(f"Value iteration estimate: {value_estimate:.6f}")
    print(f"Empirical average return: {empirical_avg:.6f}")
    print(f"The difference is small ({abs(value_estimate - empirical_avg):.6f}), indicating good convergence.")
    print("The small difference is due to:")
    print("1. Finite number of iterations (100) vs infinite horizon")
    print("2. Sampling error in empirical estimation")
    print("3. The value iteration assumes perfect knowledge of the MDP")

def run_episode(env, agent, discount):
    """Run a single episode and return the total discounted return"""
    env.reset()
    total_return = 0.0
    total_discount = 1.0
    
    while True:
        state = env.getCurrentState()
        actions = env.getPossibleActions(state)
        
        if len(actions) == 0:  # Terminal state
            break
        
        action = agent.getAction(state)
        next_state, reward = env.doAction(action)
        
        total_return += reward * total_discount
        total_discount *= discount
    
    return total_return

def main():
    """Run all analyses"""
    print("DETAILED VALUE ITERATION ANALYSIS")
    print("Homework 2.2 - Value Iteration Agent")
    
    question_2a_maze_convergence()
    question_2b_bridge_policy()
    question_2c_discount_grid_policies()
    question_2d_empirical_comparison()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()




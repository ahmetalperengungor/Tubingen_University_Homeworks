#!/usr/bin/env python3
"""
Analysis script for Value Iteration Agent
This script helps analyze the results without cluttering the main code with print statements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import agent as agent_module
import gridworld
import util
import random

class ValueIterationAnalyzer:
    def __init__(self):
        self.results = {}
    
    def analyze_maze_grid_convergence(self, max_iterations=100):
        """
        Question 2a: How many rounds of value iteration are needed before 
        the start state of MazeGrid becomes non-zero?
        """
        print("=== Question 2a: MazeGrid Convergence Analysis ===")
        
        # Get MazeGrid
        mdp = gridworld.getMazeGrid()
        start_state = mdp.getStartState()
        
        # Track value changes during iterations
        values_history = []
        
        # Initialize value function
        values = util.Counter()
        
        for iteration in range(max_iterations):
            newValues = util.Counter()
            
            for state in mdp.getStates():
                if mdp.isTerminal(state):
                    newValues[state] = 0.0
                else:
                    # Compute Q-value for each possible action
                    qValues = []
                    for action in mdp.getPossibleActions(state):
                        qValue = 0.0
                        for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                            reward = mdp.getReward(state, action, nextState)
                            qValue += prob * (reward + 0.9 * values.getCount(nextState))
                        qValues.append(qValue)
                    
                    # Value is the maximum Q-value
                    if qValues:
                        newValues[state] = max(qValues)
                    else:
                        newValues[state] = 0.0
            
            values = newValues
            start_value = values[start_state]
            values_history.append((iteration + 1, start_value))
            
            # Check if start state becomes non-zero
            if start_value > 0:
                print(f"Start state becomes non-zero after {iteration + 1} iterations")
                print(f"Value at iteration {iteration + 1}: {start_value:.6f}")
                break
        
        # Show first few iterations for context
        print("\nFirst 10 iterations:")
        for i, (iter_num, value) in enumerate(values_history[:10]):
            print(f"Iteration {iter_num}: {value:.6f}")
        
        if len(values_history) > 10:
            print("...")
            print(f"Final value after {len(values_history)} iterations: {values_history[-1][1]:.6f}")
        
        return values_history
    
    def analyze_bridge_grid_policy(self):
        """
        Question 2b: Consider the policy computed on BridgeGrid with the default 
        discount of 0.9 and the default noise of 0.2. Which parameter must we 
        change before the agent dares to cross the bridge?
        """
        print("\n=== Question 2b: BridgeGrid Policy Analysis ===")
        
        # Test different parameter combinations
        test_cases = [
            {"discount": 0.9, "noise": 0.2, "description": "Default parameters"},
            {"discount": 0.9, "noise": 0.0, "description": "No noise"},
            {"discount": 0.99, "noise": 0.2, "description": "Higher discount"},
            {"discount": 0.99, "noise": 0.0, "description": "Higher discount, no noise"},
        ]
        
        for case in test_cases:
            print(f"\nTesting: {case['description']}")
            mdp = gridworld.getBridgeGrid()
            mdp.setNoise(case['noise'])
            
            agent = agent_module.ValueIterationAgent(mdp, case['discount'], 100)
            
            # Get start state and analyze policy
            start_state = mdp.getStartState()
            policy = agent.getPolicy(start_state)
            value = agent.getValue(start_state)
            
            print(f"  Start state: {start_state}")
            print(f"  Policy at start: {policy}")
            print(f"  Value at start: {value:.6f}")
            
            # Check if agent would cross the bridge (move east from start)
            if policy == 'east':
                print("  ✓ Agent dares to cross the bridge!")
            else:
                print("  ✗ Agent avoids crossing the bridge")
    
    def analyze_discount_grid_policies(self):
        """
        Question 2c: On the DiscountGrid, find parameter values for different optimal policies
        """
        print("\n=== Question 2c: DiscountGrid Policy Analysis ===")
        
        # Test different parameter combinations
        test_cases = [
            # (discount, noise, living_reward, description)
            (0.1, 0.0, 0.0, "Low discount, no noise"),
            (0.5, 0.0, 0.0, "Medium discount, no noise"),
            (0.9, 0.0, 0.0, "High discount, no noise"),
            (0.9, 0.2, 0.0, "High discount, with noise"),
            (0.9, 0.0, -0.1, "High discount, negative living reward"),
            (0.9, 0.0, -1.0, "High discount, very negative living reward"),
        ]
        
        for discount, noise, living_reward, description in test_cases:
            print(f"\nTesting: {description}")
            print(f"  Discount: {discount}, Noise: {noise}, Living Reward: {living_reward}")
            
            mdp = gridworld.getDiscountGrid()
            mdp.setNoise(noise)
            mdp.setLivingReward(living_reward)
            
            agent = agent_module.ValueIterationAgent(mdp, discount, 100)
            
            # Analyze policy at start state
            start_state = mdp.getStartState()
            policy = agent.getPolicy(start_state)
            value = agent.getValue(start_state)
            
            print(f"  Start state: {start_state}")
            print(f"  Policy at start: {policy}")
            print(f"  Value at start: {value:.6f}")
            
            # Analyze the grid to understand the policy
            self._analyze_discount_grid_policy(mdp, agent)
    
    def _analyze_discount_grid_policy(self, mdp, agent):
        """Helper function to analyze the policy on DiscountGrid"""
        # The DiscountGrid layout:
        # [' ',' ',' ',' ',' ']
        # [' ','#',' ',' ',' ']
        # [' ','#', 1,'#', 10]
        # ['S',' ',' ',' ',' ']
        # [-10,-10, -10, -10, -10]
        
        # Check key positions
        close_exit = (2, 2)  # +1 reward
        distant_exit = (2, 4)  # +10 reward
        start_state = (3, 0)  # 'S'
        
        print(f"  Close exit (2,2) value: {agent.getValue(close_exit):.6f}")
        print(f"  Distant exit (2,4) value: {agent.getValue(distant_exit):.6f}")
        
        # Check if agent prefers close or distant exit
        if agent.getValue(close_exit) > agent.getValue(distant_exit):
            print("  → Prefers close exit")
        else:
            print("  → Prefers distant exit")
    
    def analyze_maze_grid_empirical_comparison(self, num_episodes=10000):
        """
        Question 2d: Compare value estimate vs empirical returns on MazeGrid
        """
        print("\n=== Question 2d: MazeGrid Value vs Empirical Returns ===")
        
        # Get value iteration estimate
        mdp = gridworld.getMazeGrid()
        agent = agent_module.ValueIterationAgent(mdp, 0.9, 100)
        start_state = mdp.getStartState()
        value_estimate = agent.getValue(start_state)
        
        print(f"Value iteration estimate for start state: {value_estimate:.6f}")
        
        # Calculate empirical returns
        env = gridworld.GridworldEnvironment(mdp)
        returns = []
        
        print(f"Running {num_episodes} episodes to calculate empirical returns...")
        
        for episode in range(num_episodes):
            episode_return = self._run_episode(env, agent, 0.9)
            returns.append(episode_return)
            
            if (episode + 1) % 1000 == 0:
                avg_return = sum(returns) / len(returns)
                print(f"  After {episode + 1} episodes: avg return = {avg_return:.6f}")
        
        empirical_avg = sum(returns) / len(returns)
        empirical_std = (sum([(r - empirical_avg)**2 for r in returns]) / len(returns))**0.5
        
        print(f"\nResults:")
        print(f"  Value iteration estimate: {value_estimate:.6f}")
        print(f"  Empirical average return: {empirical_avg:.6f}")
        print(f"  Empirical std deviation: {empirical_std:.6f}")
        print(f"  Difference: {abs(value_estimate - empirical_avg):.6f}")
        
        return value_estimate, empirical_avg, empirical_std
    
    def _run_episode(self, env, agent, discount):
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
    
    def run_all_analyses(self):
        """Run all analyses for the homework questions"""
        print("Value Iteration Analysis for Homework 2.2")
        print("=" * 50)
        
        # Question 2a
        self.analyze_maze_grid_convergence()
        
        # Question 2b
        self.analyze_bridge_grid_policy()
        
        # Question 2c
        self.analyze_discount_grid_policies()
        
        # Question 2d
        self.analyze_maze_grid_empirical_comparison()

def main():
    analyzer = ValueIterationAnalyzer()
    analyzer.run_all_analyses()

if __name__ == "__main__":
    main()
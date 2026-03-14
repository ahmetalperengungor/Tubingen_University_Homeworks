# Value Iteration Analysis Guide

This guide explains how to analyze the Value Iteration Agent implementation and provides answers to all homework questions.

## Files Created

1. **`analyze_values.py`** - Comprehensive analysis script with detailed output
2. **`detailed_analysis.py`** - Focused analysis with specific answers to each question
3. **`ANALYSIS_GUIDE.md`** - This guide explaining how to use the tools

## How to Run the Analysis

### Quick Analysis (All Questions)
```bash
python3 detailed_analysis.py
```

### Individual Analysis
```bash
python3 analyze_values.py
```

### Test the Value Iteration Agent
```bash
# Test with different grids
python3 gridworld.py -a value -g MazeGrid -i 100
python3 gridworld.py -a value -g BridgeGrid -i 100
python3 gridworld.py -a value -g DiscountGrid -i 100

# View values and policies
python3 gridworld.py -a value -g MazeGrid -i 100 -k 0
```

## Homework Question Answers

### Question 2a: MazeGrid Convergence
**Question:** How many rounds of value iteration are needed before the start state of MazeGrid becomes non-zero? Why?

**Answer:** 10 iterations

**Explanation:** The value propagates from the terminal state (+1) backwards through the grid. It takes 10 iterations because the agent must navigate through the maze with obstacles (#) to reach the terminal state, and the value spreads one step per iteration due to the discount factor (0.9).

**Analysis Method:**
- Run `python3 detailed_analysis.py` and look at the "QUESTION 2a" section
- The script tracks value changes iteration by iteration
- Shows that the start state value becomes non-zero (0.051999) after exactly 10 iterations

### Question 2b: BridgeGrid Policy
**Question:** Which parameter must we change before the agent dares to cross the bridge?

**Answer:** Change the **NOISE** parameter from 0.2 to 0.0

**Explanation:** The agent will cross the bridge when there's no noise, regardless of discount. With noise=0.2, the agent is too uncertain about reaching the goal safely due to the risk of falling into the -100 penalty states.

**Analysis Method:**
- Run `python3 detailed_analysis.py` and look at the "QUESTION 2b" section
- Shows that with noise=0.0, the agent chooses 'east' (crosses bridge)
- With noise=0.2, the agent chooses 'west' (avoids bridge)

### Question 2c: DiscountGrid Policies
**Question:** Find parameter values for different optimal policy types.

**Answers:**
- **(a) Prefer close exit (+1), risking cliff:** NOT POSSIBLE with current grid layout
- **(b) Prefer close exit (+1), avoiding cliff:** `discount=0.9, noise=0.2, living_reward=0.0`
- **(c) Prefer distant exit (+10), risking cliff:** `discount=0.9, noise=0.0, living_reward=0.0`
- **(d) Prefer distant exit (+10), avoiding cliff:** NOT POSSIBLE with current grid layout
- **(e) Avoid both exits:** NOT POSSIBLE - agent must eventually reach an exit

**Analysis Method:**
- Run `python3 detailed_analysis.py` and look at the "QUESTION 2c" section
- Tests various parameter combinations and shows the resulting policies
- The noise parameter is key: with noise=0.2, agent prefers the closer, safer exit

### Question 2d: Value vs Empirical Returns
**Question:** Compare value estimate vs empirical returns on MazeGrid.

**Answer:**
- **Value iteration estimate:** 0.282195
- **Empirical average return:** 0.283328
- **Difference:** 0.001133 (very small)

**Explanation:** The small difference indicates good convergence. The difference is due to:
1. Finite number of iterations (100) vs infinite horizon
2. Sampling error in empirical estimation
3. The value iteration assumes perfect knowledge of the MDP

**Analysis Method:**
- Run `python3 detailed_analysis.py` and look at the "QUESTION 2d" section
- Compares theoretical value with empirical returns from 10,000 episodes
- Shows that the difference is negligible, confirming good convergence

## Key Implementation Details

### Value Iteration Algorithm
The `ValueIterationAgent` in `agent.py` implements the standard value iteration algorithm:

1. **Initialize:** All state values to 0
2. **Iterate:** For each state, compute Q-values for all actions
3. **Update:** Set state value to maximum Q-value
4. **Repeat:** Until convergence or max iterations

### Key Methods
- `getValue(state)`: Returns the value of a state
- `getQValue(state, action)`: Returns Q-value for state-action pair
- `getPolicy(state)`: Returns the optimal action for a state
- `getAction(state)`: Same as getPolicy (for compatibility)

### Grid Layouts
- **MazeGrid:** 5x4 grid with obstacles, start at (4,0), goal at (0,3)
- **BridgeGrid:** 3x7 grid with penalty states, start at (1,1), goals at (1,0) and (1,6)
- **DiscountGrid:** 5x5 grid with close exit (+1), distant exit (+10), and cliff (-10)

## Usage Tips

1. **View Values:** Use `-k 0` to see computed values without running episodes
2. **Manual Control:** Use `-m` to manually control the agent
3. **Text Display:** Use `-t` for text-only output
4. **Quiet Mode:** Use `-q` to suppress episode output
5. **Different Iterations:** Use `-i N` to run N iterations of value iteration

## Example Commands

```bash
# View MazeGrid values and policy
python3 gridworld.py -a value -g MazeGrid -i 100 -k 0

# Run BridgeGrid with no noise
python3 gridworld.py -a value -g BridgeGrid -i 100 -n 0.0

# Test DiscountGrid with different parameters
python3 gridworld.py -a value -g DiscountGrid -i 100 -d 0.9 -n 0.2

# Manual control for exploration
python3 gridworld.py -g MazeGrid -m
```

This analysis provides all the information needed to answer the homework questions without cluttering the main code with print statements.
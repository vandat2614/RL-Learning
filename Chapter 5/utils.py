import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

def simulate_episode(env, policy) -> Tuple[List, List, List]:
    """
    Simulate one episode using given policy and return complete trajectory.
    
    Args:
        env: Environment instance with reset() and step() methods
        policy: Policy that maps states to actions (can be array, dict, or function)
        
    Returns:
        tuple: (states, actions, rewards) where:
            - states: List of states for each step
            - actions: List of actions taken at each state  
            - rewards: List of rewards received after each action
            
    Note:
        States array is longer by 1 element as it includes both initial and terminal states.
        Each transition follows: states[i] -> actions[i] -> rewards[i] -> states[i+1]
    """
    
    # Reset environment and initialize
    state = env.reset()
    done = False
    
    states = [state]  # Store all states including initial state
    actions = []      # Store actions taken
    rewards = []      # Store rewards received
    
    while not done:
        # Get action from policy
        if callable(policy):
            action = policy(state)
        elif hasattr(policy, '__getitem__'):
            action = policy[state]
        else:
            raise ValueError("Policy must be callable or indexable")
        
        # Take action and get results
        next_state, reward, done = env.step(action)
        
        # Store transition components
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        
        # Move to next state
        state = next_state
    
    return states, actions, rewards

def monte_carlo_policy_evaluation(env, policy: np.ndarray, max_iterations: int, check_iters: List[int] = None, verbose: bool = False) -> Dict[int, np.ndarray]:
    """
    Evaluate a given policy using Monte Carlo first-visit method and track values at specified iterations.

    This function implements the Monte Carlo policy evaluation algorithm to estimate the value 
    function V(s) for a given policy. It uses the first-visit Monte Carlo method, where each 
    state's value is updated based on the average of all returns obtained from that state 
    across multiple episodes.

    Args:
        env: Environment instance that provides the state space and transition dynamics.
        policy: A numpy array representing the policy to be evaluated, where each element 
                specifies the action to take in the corresponding state.
        max_iterations: Maximum number of episodes to simulate for policy evaluation 
                        (default: 10000).
        check_iters: List of iteration numbers at which to save value function snapshots 
                    for analysis purposes (default: None).
        verbose: If True, print progress every 10% of episodes completed (default: False).

    Returns:
        Dict[int, np.ndarray]: Dictionary mapping iteration numbers to value function arrays.
                                The value function has the same shape as the policy, representing 
                                the expected return for each state. Includes snapshots at all 
                                requested checkpoints and the final iteration.

    Algorithm:
        1. Initialize value function V(s) = 0 for all states
        2. For each episode:
            - Generate episode following the given policy
            - Calculate returns G for each state visited
            - Update V(s) as the average of all returns from state s
        3. Return value function snapshots at specified iterations
    """

    # Initialize value function V(s) with same shape as policy
    V = np.zeros_like(policy)


    # Initialize storage for value functions at check points
    value_snapshots = {}
    check_iters = sorted(check_iters) if check_iters else []

    # Returns(s) - store all returns for each state
    Returns = {}

    # Calculate progress milestones for verbose output
    progress_milestones = []
    if verbose:
        for i in range(1, 11):  # 10%, 20%, ..., 100%
            milestone = int(max_iterations * i / 10)
            progress_milestones.append(milestone)

    for iteration in range(max_iterations):

        # Generate an episode using current policy
        states, _, rewards = simulate_episode(env, policy)
        
        # Calculate returns for each state in the episode
        G = 0  # Return
        
        # Process states in reverse order
        for t in range(len(states) - 1, -1, -1):
            state = states[t]

            # Add reward if not terminal state (terminal state has no reward to add)
            if t >= len(rewards):
                continue
            G += rewards[t]
            
            # First-visit Monte Carlo: check if this is first occurrence of state
            if state not in states[:t]:
                
                # Initialize Returns(s) if not exists
                if state not in Returns:
                    Returns[state] = []
                
                Returns[state].append(G)
                
                # Update V(s) to average of Returns(s)
                V[state] = np.mean(Returns[state])

        # Print progress if verbose and at milestone
        if verbose and (iteration + 1) in progress_milestones:
            progress_percent = int(((iteration + 1) / max_iterations) * 100)
            print(f"Completed {iteration + 1} episodes ({progress_percent}%)")

        # Save snapshot if this iteration is in check_iters
        if (iteration + 1) in check_iters:
            value_snapshots[iteration + 1] = V.copy()

    # Include final iteration if not already included
    if max_iterations not in value_snapshots:
        value_snapshots[max_iterations] = V.copy()

    return value_snapshots


def plot_blackjack_values_wireframe(value_snapshots, view_angle=(50, -45)): # z values in right is hidden :)
    """
    Plot the state-value function for Blackjack as 3D wireframe plots.

    Creates a grid with 2 columns and multiple rows:
    - Left column: Value functions for states with usable ace
    - Right column: Value functions for states without usable ace

    The number of rows automatically adjusts based on the number of iterations
    provided in value_snapshots.

    Args:
        value_snapshots: Dictionary mapping iteration numbers to value function arrays.
                         Each value should be a 3D numpy array of shape (22, 11, 2).
        view_angle: Tuple (elev, azim) specifying the elevation and azimuth angles
                    for the 3D view. For example, (50, -45) gives a top-left view,
                    while (30, 30) gives a more frontal view.
    """
    
    if len(value_snapshots) == 0:
        raise ValueError("Function requires at least 1 iteration to plot")
    
    iterations = sorted(value_snapshots.keys())
    num_rows = len(iterations)
    
    fig_height = 4 * num_rows + 1
    fig = plt.figure(figsize=(14, fig_height))
    
    left_margin = 0.1
    right_margin = 0.8
    wspace = 0.15
    plt.subplots_adjust(wspace=wspace, hspace=0.3, left=left_margin, top=0.9, right=right_margin)
    
    left_col_center = left_margin + (right_margin - left_margin) * 0.25
    right_col_center = left_margin + (right_margin - left_margin) * 0.75

    fig.text(left_col_center, 0.97, 'Usable ace', ha='center', va='center', fontsize=14, weight='bold')
    fig.text(right_col_center, 0.97, 'No usable ace', ha='center', va='center', fontsize=14, weight='bold')
    
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    X, Y = np.meshgrid(dealer_cards, player_sums)
    
    plot_settings = {
        'color': 'black',
        'alpha': 0.8,
        'rstride': 1,
        'cstride': 1
    }
    
    for row_idx, iteration in enumerate(iterations):
        # --- Usable Ace (Left column) ---
        ax_left = fig.add_subplot(num_rows, 2, row_idx * 2 + 1, projection='3d')
        Z_usable = value_snapshots[iteration][12:22, 1:11, 1]
        ax_left.plot_wireframe(X, Y, Z_usable, **plot_settings)
        ax_left.view_init(elev=view_angle[0], azim=view_angle[1])
        ax_left.set_zlim(-1, 1)
        ax_left.set_zticks([-1, -0.5, 0, 0.5, 1])

        ax_left.set_xlabel('Dealer showing')
        ax_left.set_ylabel('Player sum')
        ax_left.set_zlabel('Value')

        # Row label
        ax_left.text2D(-0.3, 0.5, f'After {iteration:,}\nepisodes',
                       transform=ax_left.transAxes, ha='center', va='center',
                       fontsize=12, weight='bold')

        # --- No Usable Ace (Right column) ---
        ax_right = fig.add_subplot(num_rows, 2, row_idx * 2 + 2, projection='3d')
        Z_no_usable = value_snapshots[iteration][12:22, 1:11, 0]
        ax_right.plot_wireframe(X, Y, Z_no_usable, **plot_settings)
        ax_right.view_init(elev=view_angle[0], azim=view_angle[1])
        ax_right.set_zlim(-1, 1)
        ax_right.set_zticks([-1, -0.5, 0, 0.5, 1])

        ax_right.set_xlabel('Dealer showing')
        ax_right.set_ylabel('Player sum')
        ax_right.set_zlabel('Value')

    plt.show()

import numpy as np
from typing import Tuple, List, Dict, Union, Callable, Any, Optional
import matplotlib.pyplot as plt
import random

def simulate_episode(env, policy: Union[Callable, Dict[Any, Any]], reset: bool = False) -> Tuple[List, List, List]:
    """
    Simulate one episode using given policy and return complete trajectory.
    
    Args:
        env: Environment instance with reset() and step() methods
        policy: Policy that maps states to actions (can be array, dict, or function)
        reset: Whether to reset the environment before simulation. If False, 
               continues from current state in the environment.
        
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

# for example 1
def monte_carlo_policy_evaluation(env, policy: np.ndarray, max_iterations: int = 10000, check_iters: List[int] = None, verbose: bool = False) -> Dict[int, np.ndarray]:
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

    # Count visits to each state for incremental update
    C = np.zeros_like(policy, dtype=int)

    # Calculate progress milestones for verbose output
    progress_milestones = []
    if verbose:
        for i in range(1, 11):  # 10%, 20%, ..., 100%
            milestone = int(max_iterations * i / 10)
            progress_milestones.append(milestone)

    for iteration in range(max_iterations):

        # Generate an episode using current policy
        states, _, rewards = simulate_episode(env, policy)
        
        G = 0  # Return

        for t in range(len(states) - 1, -1, -1):
            state = states[t]

            # Add reward if not terminal state (terminal state has no reward to add)
            if t >= len(rewards):
                continue
            G += rewards[t]
            
            # First-visit Monte Carlo: check if this is first occurrence of state
            if state not in states[:t]:
                C[state] += 1
                V[state] += (1.0 / C[state]) * (G - V[state])

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

# for example 1
def plot_blackjack_policy_evaluation(value_snapshots, view_angle=(50, -45)): # z values in right is hidden :)
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

# for example 3
def monte_carlo_es(env, init_policy: Optional[Union[np.ndarray, Dict[Any, Any]]] = None, max_episodes: int = 100000, verbose: bool = False) -> Tuple[Dict, Dict]:
    """
    Find optimal policy using Monte Carlo ES (Exploring Starts) algorithm. 

    This function implements the Monte Carlo Exploring Starts algorithm to find the optimal 
    policy and action-value function for a given environment. The algorithm uses exploring 
    starts to ensure all state-action pairs are visited, then follows a greedy policy 
    improvement scheme based on the estimated Q-values.

    Args:
        env: Environment instance that provides state space, action space, and transition 
            dynamics. Must implement get_all_states(), get_actions(state), reset(state), 
            get_action_space(), get_state_space(), and step(action) methods.
        init_policy: Initial policy representing the starting strategy to be improved. Can be 
                    a numpy array where init_policy[state] gives the action for that state, 
                    or a dictionary mapping states to actions. If None, a random policy is 
                    initialized for all states (default: None).
        max_episodes: Maximum number of episodes to run before terminating the algorithm. 
                    This prevents infinite loops in cases where convergence is not achieved 
                    within a reasonable time or when the environment has challenging 
                    exploration requirements (default: 100000).
        verbose: If True, print progress every 10% of episodes completed (default: False).

    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - policy: NumPy array of shape (22, 11, 2) representing the optimal action 
                for each state
            - Q: NumPy array of shape (22, 11, 2, 2) representing the action-value function Q(s,a), 
                where Q[state][action] gives the expected return for taking action 
                in state and following the optimal policy thereafter
    
    Algorithm:
        1. Initialize, for all s ∈ S, a ∈ A(s):
            - Q(s,a) ← arbitrary
            - π(s) ← arbitrary  
            - Returns(s,a) ← empty list
        2. Repeat forever:
            - Choose S₀ ∈ S and A₀ ∈ A(S₀) s.t. all pairs have probability > 0
            - Generate an episode starting from S₀, A₀, following π
            - For each pair s,a appearing in the episode:
            * G ← the return that follows the first occurrence of s,a
            * Append G to Returns(s,a)
            * Q(s,a) ← average(Returns(s,a))
            - For each s in the episode:
            * π(s) ← argmax_a Q(s,a)
    """
    
    # Cache all states 
    all_states = env.get_all_states()
    
    # Initialize Q(state,action) ← arbitrary for all state ∈ S, action ∈ A(state)
    state_space = env.get_state_space()
    state_shape = tuple(max_val + 1 for (min_val, max_val) in state_space.values())
    
    all_actions = env.get_action_space()
    num_actions = len(all_actions)

    Q = np.zeros(shape=(state_shape + (num_actions,)), dtype=float)
    C = np.zeros_like(Q, dtype=int)

    # Initialize π(state) ← arbitrary for all state ∈ S
    if init_policy is None:
        policy = np.random.randint(low=min(all_actions), high=max(all_actions) + 1, size=state_shape, dtype=int)  # random initialization
    else:
        policy = init_policy
    
    # Calculate progress milestones for verbose output
    progress_milestones = []
    if verbose:
        for i in range(1, 11):  # 10%, 20%, ..., 100%
            milestone = int(max_episodes * i / 10)
            progress_milestones.append(milestone)
    
    for episode in range(max_episodes):
        
        # Choose S0 ∈ S and A0 ∈ A(S0) s.t. all pairs have probability > 0
        initial_state = random.choice(all_states)
        initial_action = random.choice(env.get_actions(initial_state))
        
        # Generate episode starting from initial_state, initial_action
        env.reset(initial_state)
        next_state, reward, done = env.step(initial_action)
        
        if done:  # Episode terminates after first step
            states = [initial_state, next_state]
            actions = [initial_action]
            rewards = [reward]
        else: # Continue episode following policy
            states, actions, rewards = simulate_episode(env, policy, reset=False)
            
            # Combine initial_state, initial_action, reward with the rest of the episode
            states = [initial_state] + states
            actions = [initial_action] + actions
            rewards = [reward] + rewards
        
        G = 0

        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            
            # Add reward if not terminal state (terminal state has no reward to add)
            if t >= len(rewards):
                continue
            G += rewards[t]
            action = actions[t]
            
            # First-visit Monte Carlo: check if this is first occurrence of state-action pair
            state_action_pairs = [(states[i], actions[i]) for i in range(t)]
            if (state, action) not in state_action_pairs:
                C[state][action] += 1
                Q[state][action] += (1.0 / C[state][action]) * (G - Q[state][action])

        for state in states[:-1]:  # Exclude terminal state
            actions = env.get_actions(state) 
            q_values = [Q[state][a] for a in actions]
            policy[state] = actions[np.argmax(q_values)]

        # Print progress if verbose and at milestone
        if verbose and (episode + 1) in progress_milestones:
            progress_percent = int(((episode + 1) / max_episodes) * 100)
            print(f"Completed {episode + 1} episodes ({progress_percent}%)")
    
    return policy, Q

# For example 3
def plot_blackjack_monte_carlo_es(policy, Q, save_path=None):
    """
    Plot policy and value function for Blackjack game
    
    Args:
        policy: NumPy array of shape (22, 11, 2) representing optimal actions
        Q: NumPy array of shape (22, 11, 2, 2) representing Q-values
        save_path: Optional path to save the figure
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Set proper spacing and margins like the reference function
    plt.subplots_adjust(wspace=0.15, hspace=0.3, left=0.15, top=0.9, right=0.8)
    
    # Extract data for plotting
    player_sums = range(11, 22)  # 12-21
    dealer_showing = range(1, 11)  # A, 2-10
    
    # Policy data: policy[:,:,1] for usable ace, policy[:,:,0] for no usable ace
    # Extract relevant portions for player sums 12-21 and dealer showing 1-10
    policy_usable = policy[12:22, 1:11, 1]  # usable ace
    policy_no_usable = policy[12:22, 1:11, 0]  # no usable ace
    
    # Value function data: extract max Q-values for each state
    value_usable = np.zeros((len(player_sums), len(dealer_showing)))
    value_no_usable = np.zeros((len(player_sums), len(dealer_showing)))
    
    # Extract max Q-values from numpy array
    for i, player_sum in enumerate(player_sums):
        for j, dealer_card in enumerate(dealer_showing):
            # Get max Q-value for each state
            value_usable[i, j] = np.max(Q[player_sum, dealer_card, 1, :])  # usable ace
            value_no_usable[i, j] = np.max(Q[player_sum, dealer_card, 0, :])  # no usable ace
    
    # Plot 1: Policy for Usable Ace
    ax1 = plt.subplot(2, 2, 1)
    # Create custom colormap: white for HIT (1), light gray for STICK (0)
    colors = ['lightgray', 'white']
    from matplotlib.colors import ListedColormap
    cmap_policy = ListedColormap(colors)
    
    im1 = ax1.imshow(policy_usable, cmap=cmap_policy, aspect='auto', origin='lower')
    ax1.set_title('π*', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    ax1.set_xticks(range(len(dealer_showing)))
    ax1.set_xticklabels(['A'] + list(range(2, 11)))
    ax1.set_yticks(range(len(player_sums)))
    ax1.set_yticklabels(player_sums)
    
    # Move y-axis to the right
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    
    # Add labels for HIT and STICK - find 2 adjacent horizontal cells in same region
    hit_positions = np.where(policy_usable == 1)
    stick_positions = np.where(policy_usable == 0)
    
    # Function to find adjacent horizontal cells in same region
    def find_adjacent_horizontal_cells(positions):
        if len(positions[0]) < 2:
            return None
        
        for i in range(len(positions[0])):
            row1, col1 = positions[0][i], positions[1][i]
            # Look for adjacent cell to the right
            for j in range(len(positions[0])):
                row2, col2 = positions[0][j], positions[1][j]
                if row1 == row2 and col2 == col1 + 1:
                    return (row1, col1, col2)  # row, col1, col2
        return None
    
    # Place HIT label between two adjacent horizontal cells
    hit_adjacent = find_adjacent_horizontal_cells(hit_positions)
    if hit_adjacent:
        row, col1, col2 = hit_adjacent
        center_col = (col1 + col2) / 2
        ax1.text(center_col, row, 'HIT', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    elif len(hit_positions[0]) > 0:
        # Fallback: place in first available HIT cell
        ax1.text(hit_positions[1][0], hit_positions[0][0], 'HIT', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    
    # Place STICK label between two adjacent horizontal cells
    stick_adjacent = find_adjacent_horizontal_cells(stick_positions)
    if stick_adjacent:
        row, col1, col2 = stick_adjacent
        center_col = (col1 + col2) / 2
        ax1.text(center_col, row, 'STICK', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    elif len(stick_positions[0]) > 0:
        # Fallback: place in first available STICK cell
        ax1.text(stick_positions[1][0], stick_positions[0][0], 'STICK', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    
    # Plot 2: Policy for No Usable Ace
    ax2 = plt.subplot(2, 2, 3)
    im2 = ax2.imshow(policy_no_usable, cmap=cmap_policy, aspect='auto', origin='lower')
    # Empty title as requested
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    ax2.set_xticks(range(len(dealer_showing)))
    ax2.set_xticklabels(['A'] + list(range(2, 11)))
    ax2.set_yticks(range(len(player_sums)))
    ax2.set_yticklabels(player_sums)
    
    # Move y-axis to the right
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    
    # Add labels for HIT and STICK for no usable ace
    hit_positions_no = np.where(policy_no_usable == 1)
    stick_positions_no = np.where(policy_no_usable == 0)
    
    # Place HIT label between two adjacent horizontal cells
    hit_adjacent_no = find_adjacent_horizontal_cells(hit_positions_no)
    if hit_adjacent_no:
        row, col1, col2 = hit_adjacent_no
        center_col = (col1 + col2) / 2
        ax2.text(center_col, row, 'HIT', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    elif len(hit_positions_no[0]) > 0:
        # Fallback: place in first available HIT cell
        ax2.text(hit_positions_no[1][0], hit_positions_no[0][0], 'HIT', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    
    # Place STICK label between two adjacent horizontal cells
    stick_adjacent_no = find_adjacent_horizontal_cells(stick_positions_no)
    if stick_adjacent_no:
        row, col1, col2 = stick_adjacent_no
        center_col = (col1 + col2) / 2
        ax2.text(center_col, row, 'STICK', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    elif len(stick_positions_no[0]) > 0:
        # Fallback: place in first available STICK cell
        ax2.text(stick_positions_no[1][0], stick_positions_no[0][0], 'STICK', ha='center', va='center', 
                color='black', fontsize=12, weight='bold')
    
    # Plot 3: Value function for Usable Ace (3D)
    ax3 = fig.add_subplot(2, 2, 2, projection='3d')
    X, Y = np.meshgrid(dealer_showing, player_sums)
    
    # Plot wireframe with black lines
    ax3.plot_wireframe(X, Y, value_usable, color='black', alpha=0.8, rstride=1, cstride=1)
    ax3.set_title('V* - Usable ace', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Dealer showing')
    ax3.set_ylabel('Player sum')
    ax3.set_zlabel('Value')
    ax3.set_xticks(range(1, 11))
    ax3.set_xticklabels(['A'] + list(range(2, 11)))
    ax3.set_zlim(-1, 1)
    ax3.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax3.view_init(elev=50, azim=-45)
    
    # Plot 4: Value function for No Usable Ace (3D)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot_wireframe(X, Y, value_no_usable, color='black', alpha=0.8, rstride=1, cstride=1)
    # Removed title as requested
    ax4.set_xlabel('Dealer showing')
    ax4.set_ylabel('Player sum')
    ax4.set_zlabel('Value')
    ax4.set_xticks(range(1, 11))
    ax4.set_xticklabels(['A'] + list(range(2, 11)))
    ax4.set_zlim(-1, 1)
    ax4.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax4.view_init(elev=50, azim=-45)
    
    plt.tight_layout()
    
    # Add row labels on the left side like "After 10,000 episodes"
    # Calculate row centers based on subplot positions
    fig.text(0.02, 0.75, 'Usable ace', ha='left', va='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.25, 'No usable ace', ha='left', va='center', fontsize=14, weight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
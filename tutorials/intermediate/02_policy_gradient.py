#!/usr/bin/env python3
"""
🎯 Policy Gradient Methods: Direct Policy Learning

Instead of learning values, let's directly learn the policy! This is like
teaching our puppy to develop intuition and instincts rather than just
memorizing "if this, then that" rules.

What you'll learn:
- Policy-based vs value-based methods
- REINFORCE algorithm (Monte Carlo Policy Gradients)
- How to train stochastic policies
- The policy gradient theorem

Think of this as: "Teaching our puppy to develop natural instincts
and intuitive responses rather than just following a rulebook!"
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class PolicyNetwork(nn.Module):
    """
    Policy Network: Directly outputs action probabilities.
    
    Unlike DQN which outputs Q-values, this network directly tells us
    the probability of taking each action. Think of it as the puppy's
    "intuition center" that gives gut feelings about what to do.
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 128, output_size: int = 2):
        """
        Initialize the policy network.
        
        Args:
            input_size: Number of state features (4 for CartPole)
            hidden_size: Hidden layer size
            output_size: Number of actions (2 for CartPole)
        """
        super(PolicyNetwork, self).__init__()
        
        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights properly."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """
        Forward pass: convert state to action probabilities.
        
        Input: state (what the puppy observes)
        Output: probability distribution over actions (what the puppy feels like doing)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Convert to probabilities using softmax
        # This ensures probabilities sum to 1
        return F.softmax(x, dim=-1)

class REINFORCEAgent:
    """
    REINFORCE Agent: Learns policies using Monte Carlo Policy Gradients.
    
    Key idea: "If I got high rewards after taking an action in a state,
    I should increase the probability of taking that action in that state."
    
    This is like a puppy learning: "When I sit after hearing 'sit' and
    get lots of treats, I should sit more often when I hear 'sit'!"
    """
    
    def __init__(self, state_size: int = 4, action_size: int = 2, learning_rate: float = 0.01):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions  
            learning_rate: How fast to update the policy
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Policy network
        self.policy_net = PolicyNetwork(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []  
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Training tracking
        self.policy_losses = []
    
    def choose_action(self, state, training: bool = True):
        """
        Choose action by sampling from the policy distribution.
        
        Unlike DQN which picks the highest Q-value, policy methods
        sample from a probability distribution. This allows for:
        1. Natural exploration
        2. Stochastic policies
        3. Better performance in some environments
        
        Like a puppy having "instincts" with some randomness!
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities from policy network
        action_probs = self.policy_net(state_tensor)
        
        if training:
            # Sample action according to probabilities (stochastic policy)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store for learning later
            self.episode_states.append(state)
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(log_prob)
            
            return action.item()
        else:
            # For testing, use most likely action (deterministic)
            return action_probs.argmax().item()
    
    def store_reward(self, reward: float):
        """Store reward for current step."""
        self.episode_rewards.append(reward)
    
    def calculate_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """
        Calculate discounted returns for the episode.
        
        Returns = sum of future discounted rewards from each time step
        This tells us "how good was being in this state, given what happened afterward?"
        
        Like a puppy looking back and thinking: "That situation where I sat
        led to many treats over time, so it was really good!"
        """
        returns = []
        R = 0
        
        # Calculate returns backward through the episode
        for reward in reversed(self.episode_rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        
        # Convert to tensor and normalize (helps with training stability)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self, gamma: float = 0.99):
        """
        Update policy using REINFORCE algorithm.
        
        The key insight: increase probability of actions that led to high returns,
        decrease probability of actions that led to low returns.
        
        Like a puppy adjusting its instincts based on which behaviors
        led to the most treats and praise!
        """
        if len(self.episode_rewards) == 0:
            return None
        
        # Calculate returns (how good was each state-action pair)
        returns = self.calculate_returns(gamma)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.episode_log_probs, returns):
            # Policy gradient: -log_prob * return
            # Negative because we want to maximize reward (minimize negative reward)
            policy_loss.append(-log_prob * R)
        
        # Total loss for the episode
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Track training progress
        self.policy_losses.append(policy_loss.item())
        
        # Clear episode memory
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        
        return policy_loss.item()

def train_reinforce_agent(episodes: int = 2000) -> Tuple[REINFORCEAgent, List[float]]:
    """
    Train REINFORCE agent.
    
    Unlike DQN which learns after every step, REINFORCE learns after
    every complete episode. This is because we need to see the full
    trajectory to calculate returns.
    """
    print("🎯 TRAINING REINFORCE AGENT")
    print("=" * 50)
    print(f"Training for {episodes} episodes with policy gradients...")
    print("Teaching our puppy to develop natural instincts! 🐕✨")
    print()
    
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(learning_rate=0.01)
    
    # Track performance
    scores = []
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        # Run one complete episode
        for step in range(500):  # Max 500 steps
            # Choose action using current policy
            action = agent.choose_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            
            # Update state
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Learn from the complete episode
        loss = agent.update_policy(gamma=0.99)
        
        # Record performance
        scores.append(total_reward)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(scores[-100:])
            avg_loss = np.mean(agent.policy_losses[-10:]) if agent.policy_losses else 0
            print(f"Episode {episode + 1:4d}: Avg Score = {recent_avg:6.1f}, "
                  f"Recent Loss = {avg_loss:.2f}")
    
    env.close()
    return agent, scores

def test_reinforce_agent(agent: REINFORCEAgent, episodes: int = 10, render: bool = False) -> List[float]:
    """Test the trained REINFORCE agent."""
    print(f"\n🧪 TESTING REINFORCE AGENT ({episodes} episodes)")
    print("=" * 50)
    
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    test_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            # Use deterministic policy for testing
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if render:
                time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        test_scores.append(total_reward)
        print(f"Test episode {episode + 1}: Score = {total_reward}")
        
        if render:
            time.sleep(1)
    
    env.close()
    return test_scores

def analyze_reinforce_training(agent: REINFORCEAgent, scores: List[float]) -> None:
    """Analyze REINFORCE training results."""
    print(f"\n📊 REINFORCE TRAINING ANALYSIS")
    print("=" * 50)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning curve
    window = 100
    rolling_avg = []
    for i in range(len(scores)):
        start_idx = max(0, i - window + 1)
        rolling_avg.append(np.mean(scores[start_idx:i+1]))
    
    axes[0, 0].plot(scores, alpha=0.3, color='lightblue', label='Episode Score')
    axes[0, 0].plot(rolling_avg, color='red', linewidth=2, label=f'{window}-episode Average')
    axes[0, 0].axhline(y=195, color='green', linestyle='--', label='Success Threshold')
    axes[0, 0].set_title('REINFORCE Learning Progress')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Policy loss over time
    if agent.policy_losses:
        loss_episodes = range(len(agent.policy_losses))
        axes[0, 1].plot(loss_episodes, agent.policy_losses, color='orange', alpha=0.7)
        
        # Smooth loss curve
        if len(agent.policy_losses) > 50:
            loss_smooth = []
            window_loss = 50
            for i in range(len(agent.policy_losses)):
                start_idx = max(0, i - window_loss + 1)
                loss_smooth.append(np.mean(agent.policy_losses[start_idx:i+1]))
            axes[0, 1].plot(loss_episodes, loss_smooth, color='red', linewidth=2)
        
        axes[0, 1].set_title('Policy Loss Over Episodes')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Score distribution over training phases
    episodes = len(scores)
    early_scores = scores[:episodes//3]
    mid_scores = scores[episodes//3:2*episodes//3]
    late_scores = scores[2*episodes//3:]
    
    axes[1, 0].hist(early_scores, alpha=0.6, color='red', bins=15, 
                   label='Early (1st third)', density=True)
    axes[1, 0].hist(mid_scores, alpha=0.6, color='orange', bins=15,
                   label='Mid (2nd third)', density=True)
    axes[1, 0].hist(late_scores, alpha=0.6, color='green', bins=15,
                   label='Late (3rd third)', density=True)
    axes[1, 0].set_title('Score Distribution Across Training')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training statistics
    early_avg = np.mean(early_scores)
    mid_avg = np.mean(mid_scores)  
    late_avg = np.mean(late_scores)
    success_rate = sum(1 for score in scores[-100:] if score >= 195) / 100
    
    stats_text = f"""
    REINFORCE Training Results:
    
    Total Episodes: {episodes}
    Network: 4 → 128 → 128 → 2
    Learning Rate: {agent.learning_rate}
    
    Early Avg (1st third): {early_avg:.1f}
    Mid Avg (2nd third): {mid_avg:.1f}
    Late Avg (3rd third): {late_avg:.1f}
    
    Total Improvement: +{late_avg - early_avg:.1f}
    Best Score: {max(scores):.0f}
    Final 100-ep Avg: {np.mean(scores[-100:]):.1f}
    Success Rate (≥195): {success_rate:.1%}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/reinforce_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Success rate: {success_rate:.1%} (episodes scoring ≥195)")
    print(f"📊 Analysis chart saved to /tmp/reinforce_training_analysis.png")

def demonstrate_policy_behavior(agent: REINFORCEAgent):
    """
    Show how the learned policy behaves in different states.
    
    This helps us understand what "instincts" our puppy has learned!
    """
    print("\n🎭 LEARNED POLICY ANALYSIS")
    print("=" * 50)
    
    # Create some test states to analyze
    test_states = [
        # [cart_pos, cart_vel, pole_angle, pole_vel]
        [0.0, 0.0, 0.0, 0.0],      # Perfect balance
        [0.0, 0.0, 0.1, 0.0],      # Pole tilting right
        [0.0, 0.0, -0.1, 0.0],     # Pole tilting left
        [0.5, 0.0, 0.0, 0.0],      # Cart pushed right
        [-0.5, 0.0, 0.0, 0.0],     # Cart pushed left
        [0.0, 0.5, 0.0, 0.0],      # Cart moving right
        [0.0, -0.5, 0.0, 0.0],     # Cart moving left
        [0.0, 0.0, 0.05, 0.5],     # Pole falling right fast
        [0.0, 0.0, -0.05, -0.5],   # Pole falling left fast
    ]
    
    state_descriptions = [
        "Perfect balance",
        "Pole tilting right", 
        "Pole tilting left",
        "Cart pushed right",
        "Cart pushed left", 
        "Cart moving right",
        "Cart moving left",
        "Pole falling right fast",
        "Pole falling left fast"
    ]
    
    print("Policy Analysis: What does our puppy's instinct say?")
    print("=" * 60)
    print("State Description          | Left Prob | Right Prob | Preferred Action")
    print("-" * 70)
    
    for state, description in zip(test_states, state_descriptions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = agent.policy_net(state_tensor)
        left_prob = action_probs[0, 0].item()
        right_prob = action_probs[0, 1].item()
        preferred = "LEFT" if left_prob > right_prob else "RIGHT"
        
        print(f"{description:25} | {left_prob:8.3f} | {right_prob:9.3f} | {preferred}")
    
    print()
    print("🧠 What patterns do you notice?")
    print("- Does the policy push the cart toward the falling pole?")
    print("- How does it handle different velocities?")
    print("- What happens in balanced situations?")

def compare_value_vs_policy_methods():
    """Compare value-based (DQN) vs policy-based (REINFORCE) methods."""
    print("\n🔬 VALUE-BASED VS POLICY-BASED METHODS")
    print("=" * 70)
    print("| Aspect              | Value-Based (DQN)    | Policy-Based (REINFORCE) |")
    print("|--------------------|----------------------|---------------------------|")
    print("| What it learns     | Q-values (utilities) | Policy (action probs)     |")  
    print("| Output             | Action values        | Action probabilities      |")
    print("| Action selection   | Argmax Q-values      | Sample from distribution  |")
    print("| Learning frequency | Every step           | Every episode             |") 
    print("| Memory required    | Experience replay    | Episode buffer            |")
    print("| Exploration        | Epsilon-greedy       | Natural (stochastic)      |")
    print("| Stochastic policies| Difficult            | Natural                   |")
    print("| Continuous actions | Hard to extend       | Easy to extend            |")
    print("| Sample efficiency  | Generally higher     | Generally lower           |")
    print("| Stability          | Can be unstable      | More stable gradients     |")
    print()

def main():
    """Main REINFORCE training and analysis."""
    print("🎯 POLICY GRADIENT METHODS (REINFORCE)")
    print("=" * 60) 
    print("Welcome to policy-based reinforcement learning!")
    print("Teaching our puppy to develop natural instincts! 🐕✨")
    print()
    
    # Compare methods
    compare_value_vs_policy_methods()
    
    # Train REINFORCE agent
    agent, scores = train_reinforce_agent(episodes=1500)
    
    # Analyze training
    analyze_reinforce_training(agent, scores)
    
    # Demonstrate learned policy
    demonstrate_policy_behavior(agent)
    
    # Test the trained agent
    test_scores = test_reinforce_agent(agent, episodes=10)
    
    print(f"\n🏆 FINAL RESULTS")
    print("=" * 50)
    print(f"Training average (last 100): {np.mean(scores[-100:]):.1f}")
    print(f"Test average: {np.mean(test_scores):.1f}")
    print(f"Best test score: {max(test_scores):.0f}")
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), '/tmp/reinforce_cartpole.pth')
    print(f"🔧 Trained model saved to /tmp/reinforce_cartpole.pth")
    
    print(f"\n🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ We can directly learn policies without value functions")
    print("✅ REINFORCE uses complete episodes to learn")
    print("✅ Policy gradients provide natural exploration")
    print("✅ Stochastic policies can be more robust")
    print()
    print("🤔 KEY INSIGHTS:")
    print("- Policy methods learn the decision-making process directly")
    print("- Monte Carlo methods need complete episodes")
    print("- Natural exploration comes from stochastic policies")
    print("- Policy gradients can handle continuous action spaces")
    print()
    print("🚀 NEXT STEPS:")
    print("- Try Actor-Critic methods (combine value and policy)")
    print("- Experiment with different policy network architectures")
    print("- Look into advanced methods like PPO and TRPO!")

if __name__ == "__main__":
    main()
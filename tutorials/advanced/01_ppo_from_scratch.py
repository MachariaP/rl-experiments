#!/usr/bin/env python3
"""
🚀 Proximal Policy Optimization (PPO): State-of-the-Art RL

PPO is one of the most successful modern RL algorithms! It combines the best
of policy gradients with stability improvements. This is like giving our
puppy the most advanced, reliable training method available.

What you'll learn:
- Why PPO is so popular in modern RL
- Clipped surrogate objective function
- Actor-Critic architecture
- How to build PPO from scratch

Think of this as: "The most advanced, reliable way to train our puppy
that's used by top AI research labs worldwide!"
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import time
from typing import List, Tuple, Dict

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    
    This is a dual-purpose brain:
    - Actor: Decides what actions to take (like REINFORCE)
    - Critic: Evaluates how good states are (like DQN values)
    
    Think of it as a puppy that can both:
    1. Decide what to do (actor)
    2. Judge how good situations are (critic)
    """
    
    def __init__(self, state_size: int = 4, action_size: int = 2, hidden_size: int = 64):
        """Initialize the actor-critic network."""
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers (common understanding of the environment)
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy) - decides actions
        self.actor_fc = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function) - evaluates states  
        self.critic_fc = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 0.01)
            module.bias.data.fill_(0.0)
    
    def forward(self, x):
        """
        Forward pass through both actor and critic.
        
        Returns:
            action_probs: Probability distribution over actions (actor)
            state_value: Estimated value of the current state (critic)
        """
        # Shared feature extraction
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        # Actor: action probabilities
        action_logits = self.actor_fc(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic: state value
        state_value = self.critic_fc(x)
        
        return action_probs, state_value

class PPOAgent:
    """
    PPO Agent: Implements Proximal Policy Optimization.
    
    Key innovations over REINFORCE:
    1. Actor-Critic: Uses value function to reduce variance
    2. Multiple epochs: Updates the policy multiple times per episode
    3. Clipped objective: Prevents too large policy updates
    4. Advantage estimation: Better signal for learning
    
    This is like having a super-smart training program for our puppy!
    """
    
    def __init__(self, state_size: int = 4, action_size: int = 2, learning_rate: float = 3e-4):
        """Initialize PPO agent."""
        self.state_size = state_size
        self.action_size = action_size
        
        # PPO hyperparameters
        self.lr = learning_rate
        self.gamma = 0.99          # Discount factor
        self.gae_lambda = 0.95     # GAE parameter
        self.clip_ratio = 0.2      # PPO clip parameter
        self.ppo_epochs = 10       # Number of PPO epochs per update
        self.mini_batch_size = 64  # Mini-batch size for updates
        self.value_coeff = 0.5     # Value function loss coefficient
        self.entropy_coeff = 0.01  # Entropy bonus coefficient
        
        # Networks
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Episode storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
    
    def choose_action(self, state, training: bool = True):
        """
        Choose action using the current policy.
        
        Returns both the action and additional info needed for PPO updates.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)
        
        if training:
            # Sample action from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store for learning
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(state_value.item())
            
            return action.item()
        else:
            # For testing, use most likely action
            return action_probs.argmax().item()
    
    def store_transition(self, reward: float, done: bool):
        """Store reward and done flag for current transition."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae_returns(self, next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) returns and advantages.
        
        GAE reduces variance in advantage estimates by using a parameter λ
        to blend n-step returns. This is like giving our puppy a more
        accurate understanding of how good each action really was.
        """
        values = self.values + [next_value]
        advantages = []
        returns = []
        
        gae = 0
        for step in reversed(range(len(self.rewards))):
            # Temporal difference error
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            
            # GAE advantage
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            
            # Return for this step
            returns.insert(0, gae + values[step])
        
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def ppo_update(self, returns: torch.Tensor, advantages: torch.Tensor):
        """
        Perform PPO update with clipped objective.
        
        This is the core of PPO! We update the policy multiple times
        but prevent it from changing too much at once.
        """
        # Convert episode data to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        values_pred = torch.FloatTensor(self.values)
        
        # Create dataset for mini-batch updates
        dataset_size = len(self.states)
        
        # Multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle data for each epoch
            indices = torch.randperm(dataset_size)
            
            # Mini-batch updates
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass with current policy
                action_probs, state_values = self.network(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate probability ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss (critic)
                critic_loss = F.mse_loss(state_values.squeeze(), batch_returns)
                
                # Total loss
                total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        # Store losses for analysis
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.entropy_losses.append(entropy.item())
    
    def clear_episode_data(self):
        """Clear stored episode data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

def train_ppo_agent(episodes: int = 1000) -> Tuple[PPOAgent, List[float]]:
    """
    Train PPO agent.
    
    PPO collects a batch of experience and then updates the policy
    multiple times on that batch. This is more sample-efficient
    than REINFORCE.
    """
    print("🚀 TRAINING PPO AGENT")
    print("=" * 50)
    print(f"Training for {episodes} episodes with PPO...")
    print("Using the most advanced stable policy optimization! 🤖🐕")
    print()
    
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = PPOAgent()
    
    # Track performance
    scores = []
    recent_scores = deque(maxlen=100)
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        # Run episode
        for step in range(500):  # Max 500 steps
            # Choose action
            action = agent.choose_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(reward, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Calculate returns and advantages
        returns, advantages = agent.compute_gae_returns()
        
        # PPO update
        agent.ppo_update(returns, advantages)
        
        # Clear episode data
        agent.clear_episode_data()
        
        # Track performance
        scores.append(total_reward)
        recent_scores.append(total_reward)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(recent_scores)
            avg_actor_loss = np.mean(agent.actor_losses[-10:]) if agent.actor_losses else 0
            avg_critic_loss = np.mean(agent.critic_losses[-10:]) if agent.critic_losses else 0
            
            print(f"Episode {episode + 1:4d}: Avg Score = {avg_score:6.1f}, "
                  f"Actor Loss = {avg_actor_loss:.3f}, Critic Loss = {avg_critic_loss:.3f}")
    
    env.close()
    return agent, scores

def test_ppo_agent(agent: PPOAgent, episodes: int = 10, render: bool = False) -> List[float]:
    """Test the trained PPO agent."""
    print(f"\n🧪 TESTING PPO AGENT ({episodes} episodes)")
    print("=" * 50)
    
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    test_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
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

def analyze_ppo_training(agent: PPOAgent, scores: List[float]) -> None:
    """Analyze PPO training results."""
    print(f"\n📊 PPO TRAINING ANALYSIS")
    print("=" * 50)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Learning curve
    window = 50
    rolling_avg = []
    for i in range(len(scores)):
        start_idx = max(0, i - window + 1)
        rolling_avg.append(np.mean(scores[start_idx:i+1]))
    
    axes[0, 0].plot(scores, alpha=0.3, color='lightblue', label='Episode Score')
    axes[0, 0].plot(rolling_avg, color='red', linewidth=2, label=f'{window}-episode Average')
    axes[0, 0].axhline(y=195, color='green', linestyle='--', label='Success Threshold')
    axes[0, 0].set_title('PPO Learning Progress')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Actor loss
    if agent.actor_losses:
        axes[0, 1].plot(agent.actor_losses, color='red', alpha=0.7)
        axes[0, 1].set_title('Actor Loss (Policy)')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Actor Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Critic loss
    if agent.critic_losses:
        axes[0, 2].plot(agent.critic_losses, color='blue', alpha=0.7)
        axes[0, 2].set_title('Critic Loss (Value Function)')
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Critic Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Entropy over time
    if agent.entropy_losses:
        axes[1, 0].plot(agent.entropy_losses, color='purple', alpha=0.7)
        axes[1, 0].set_title('Policy Entropy')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Score distribution
    episodes = len(scores)
    early_scores = scores[:episodes//3]
    mid_scores = scores[episodes//3:2*episodes//3]
    late_scores = scores[2*episodes//3:]
    
    axes[1, 1].hist(early_scores, alpha=0.6, color='red', bins=15, 
                   label='Early', density=True)
    axes[1, 1].hist(mid_scores, alpha=0.6, color='orange', bins=15,
                   label='Mid', density=True)
    axes[1, 1].hist(late_scores, alpha=0.6, color='green', bins=15,
                   label='Late', density=True)
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Training statistics
    success_rate = sum(1 for score in scores[-100:] if score >= 195) / 100
    
    stats_text = f"""
    PPO Training Results:
    
    Episodes: {episodes}
    Architecture: Actor-Critic
    
    Performance:
    Final 100-ep Avg: {np.mean(scores[-100:]):.1f}
    Best Score: {max(scores):.0f}
    Success Rate: {success_rate:.1%}
    
    Hyperparameters:
    Learning Rate: {agent.lr}
    Clip Ratio: {agent.clip_ratio}
    PPO Epochs: {agent.ppo_epochs}
    GAE Lambda: {agent.gae_lambda}
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_title('Training Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/ppo_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Success rate: {success_rate:.1%}")
    print(f"📊 Analysis chart saved to /tmp/ppo_training_analysis.png")

def compare_rl_algorithms():
    """Compare all the algorithms we've learned."""
    print("\n🔬 RL ALGORITHMS COMPARISON")
    print("=" * 80)
    print("| Algorithm  | Type        | Learning    | Stability | Sample Eff | Modern Use |")
    print("|------------|-------------|-------------|-----------|------------|------------|")
    print("| Q-Table    | Value-based | Simple      | High      | Low        | Education  |")
    print("| DQN        | Value-based | Deep        | Medium    | Medium     | Some games |")
    print("| REINFORCE  | Policy-based| Basic PG    | Low       | Low        | Research   |")
    print("| PPO        | Actor-Critic| Advanced PG | High      | High       | Industry   |")
    print()
    print("🏆 PPO is the current gold standard for many RL applications!")

def main():
    """Main PPO training and analysis."""
    print("🚀 PROXIMAL POLICY OPTIMIZATION (PPO)")
    print("=" * 60)
    print("Welcome to state-of-the-art reinforcement learning!")
    print("This is the algorithm used by OpenAI, DeepMind, and others! 🤖🐕")
    print()
    
    # Compare algorithms
    compare_rl_algorithms()
    
    # Train PPO agent  
    agent, scores = train_ppo_agent(episodes=800)
    
    # Analyze training
    analyze_ppo_training(agent, scores)
    
    # Test the trained agent
    test_scores = test_ppo_agent(agent, episodes=10)
    
    print(f"\n🏆 FINAL RESULTS")
    print("=" * 50)
    print(f"Training average (last 100): {np.mean(scores[-100:]):.1f}")
    print(f"Test average: {np.mean(test_scores):.1f}")
    print(f"Best test score: {max(test_scores):.0f}")
    
    # Save the trained model
    torch.save(agent.network.state_dict(), '/tmp/ppo_cartpole.pth')
    print(f"🔧 Trained model saved to /tmp/ppo_cartpole.pth")
    
    print(f"\n🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ PPO combines the best of policy and value methods")
    print("✅ Clipped objectives prevent destructive policy updates")
    print("✅ Actor-Critic reduces variance in learning")
    print("✅ PPO is stable and sample-efficient")
    print()
    print("🤔 KEY INSIGHTS:")
    print("- PPO's clipping mechanism ensures stable learning")
    print("- Actor-critic architecture leverages both policy and value learning")
    print("- GAE provides better advantage estimates")
    print("- Multiple epochs per batch improve sample efficiency")
    print()
    print("🚀 NEXT STEPS:")
    print("- Try PPO on more complex environments")
    print("- Experiment with hyperparameters")
    print("- Check out Stable-Baselines3 for production-ready PPO!")
    
    print(f"\n🌟 CONGRATULATIONS!")
    print("=" * 50)
    print("You've now learned the core algorithms of modern RL:")
    print("• Q-Learning (tabular)")
    print("• Deep Q-Networks (value-based)")
    print("• REINFORCE (policy gradients)")
    print("• PPO (actor-critic)")
    print()
    print("You're ready to tackle real-world RL problems! 🎯")

if __name__ == "__main__":
    main()
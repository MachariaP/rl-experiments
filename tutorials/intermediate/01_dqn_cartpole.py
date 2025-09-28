#!/usr/bin/env python3
"""
🧠🔥 Deep Q-Networks (DQN): Neural Network Agent

Now we're upgrading from simple tables to neural networks! This is like
teaching our puppy to think with a much more sophisticated brain that
can handle complex, continuous situations.

What you'll learn:
- How neural networks replace Q-tables
- Experience replay and why it matters
- Target networks for stable learning  
- Deep reinforcement learning fundamentals

Think of this as: "Our puppy now has a super-brain that can understand
millions of different situations and learn complex patterns!"
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from typing import List, Tuple
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Experience tuple for storing memories
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNNetwork(nn.Module):
    """
    Deep Q-Network: The neural network brain of our agent.
    
    Think of this as a super-sophisticated puppy brain that can:
    - Process complex visual and sensory information
    - Learn millions of different patterns
    - Make decisions based on experience
    """
    
    def __init__(self, input_size: int = 4, hidden_size: int = 128, output_size: int = 2):
        """
        Initialize our neural network brain.
        
        Args:
            input_size: Number of observations (4 for CartPole)
            hidden_size: Number of neurons in hidden layers
            output_size: Number of possible actions (2 for CartPole)
        """
        super(DQNNetwork, self).__init__()
        
        # Define the neural network layers
        # Think of these as different "thinking stages" in the puppy's brain
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """
        Forward pass: how the brain processes information.
        
        Input observations -> Hidden layer 1 -> Hidden layer 2 -> Q-values for actions
        """
        x = F.relu(self.fc1(x))  # First thinking stage
        x = F.relu(self.fc2(x))  # Second thinking stage  
        x = self.fc3(x)          # Final decision: Q-values for each action
        return x

class DQNAgent:
    """
    Deep Q-Network Agent: Our smart puppy with a neural network brain!
    
    Key innovations over simple Q-learning:
    1. Neural network instead of table (handles continuous states)
    2. Experience replay (learns from past memories)
    3. Target network (provides stable learning targets)
    4. Epsilon-greedy exploration with decay
    """
    
    def __init__(self, state_size: int = 4, action_size: int = 2, learning_rate: float = 0.001):
        """
        Initialize our DQN agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: How fast the network learns
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Hyperparameters
        self.memory_size = 10000    # How many experiences to remember
        self.batch_size = 32        # How many experiences to learn from at once
        self.epsilon = 1.0          # Exploration rate (start high)
        self.epsilon_min = 0.01     # Minimum exploration rate
        self.epsilon_decay = 0.995  # How fast to reduce exploration
        self.gamma = 0.95           # Discount factor for future rewards
        self.target_update = 1000   # How often to update target network
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, 128, action_size)    # Main brain
        self.target_network = DQNNetwork(state_size, 128, action_size)  # Target brain
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay memory (like a diary of experiences)
        self.memory = deque(maxlen=self.memory_size)
        
        # Training tracking
        self.steps_done = 0
        self.losses = []
        
        # Initialize target network with same weights
        self.update_target_network()
    
    def update_target_network(self):
        """
        Copy weights from main network to target network.
        
        Like giving our puppy a "stable reference" to compare against
        while the main brain is still learning and changing rapidly.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory for later learning.
        
        Like a puppy remembering: "When I was in this situation and did this action,
        this is what happened and how I felt about it."
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)
    
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy strategy.
        
        Like a puppy deciding: "Should I try something new (explore) or
        do what I know works (exploit)?"
        """
        if training and random.random() <= self.epsilon:
            # Exploration: random action (curious puppy)
            return random.choice(range(self.action_size))
        
        # Exploitation: use neural network to predict best action (smart puppy)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay_experience(self):
        """
        Learn from a batch of past experiences.
        
        Like a puppy reviewing its diary: "Let me think about all these
        past experiences and update my understanding of what works."
        
        This is the core of deep reinforcement learning!
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough experiences yet
        
        # Sample a random batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        # Current Q-values: what does our main brain currently think?
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network: what should we aim for?
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss: how wrong were our predictions?
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update the network: teach our brain to be more accurate
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Track training progress
        self.losses.append(loss.item())
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()
        
        return loss.item()
    
    def decay_epsilon(self):
        """
        Reduce exploration over time.
        
        Like a puppy becoming more confident and relying more on
        learned behaviors as it gains experience.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(episodes: int = 1000) -> Tuple[DQNAgent, List[float], List[int]]:
    """
    Train our DQN agent through many episodes.
    
    Like running an intensive training program for our super-smart puppy!
    """
    print("🧠🔥 TRAINING DEEP Q-NETWORK AGENT")
    print("=" * 50)
    print(f"Training for {episodes} episodes with neural networks...")
    print("Our puppy is getting a major brain upgrade! 🤖🐕")
    print()
    
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    
    # Track performance
    scores = []
    episode_lengths = []
    losses = []
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        # Run one episode
        for step in range(500):  # Max 500 steps per episode
            # Agent chooses action
            action = agent.choose_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience replay
            loss = agent.replay_experience()
            if loss is not None:
                losses.append(loss)
            
            # Update for next step
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Record performance
        scores.append(total_reward)
        episode_lengths.append(steps)
        
        # Reduce exploration
        agent.decay_epsilon()
        
        # Print progress
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(scores[-50:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode + 1:4d}: Avg Score = {recent_avg:6.1f}, "
                  f"Epsilon = {agent.epsilon:.3f}, Avg Loss = {avg_loss:.4f}")
    
    env.close()
    return agent, scores, episode_lengths

def test_dqn_agent(agent: DQNAgent, episodes: int = 10, render: bool = False) -> List[float]:
    """
    Test our trained DQN agent.
    
    Like giving our super-smart puppy a final exam!
    """
    print(f"\n🧪 TESTING DQN AGENT ({episodes} episodes)")
    print("=" * 50)
    
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    test_scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            # No exploration during testing - pure exploitation
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if render:
                time.sleep(0.01)  # Slow down for viewing
            
            if terminated or truncated:
                break
        
        test_scores.append(total_reward)
        print(f"Test episode {episode + 1}: Score = {total_reward}")
        
        if render:
            time.sleep(1)  # Pause between episodes
    
    env.close()
    return test_scores

def analyze_dqn_training(scores: List[float], episode_lengths: List[int], agent: DQNAgent) -> None:
    """
    Analyze the DQN training process.
    
    Like reviewing our puppy's entire learning journey! 📊
    """
    print(f"\n📊 DQN TRAINING ANALYSIS")
    print("=" * 50)
    
    # Create comprehensive visualizations
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
    axes[0, 0].set_title('DQN Learning Progress')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training loss over time
    if agent.losses:
        loss_smooth = []
        window_loss = 100
        for i in range(len(agent.losses)):
            start_idx = max(0, i - window_loss + 1)
            loss_smooth.append(np.mean(agent.losses[start_idx:i+1]))
        
        axes[0, 1].plot(agent.losses, alpha=0.3, color='orange', label='Raw Loss')
        axes[0, 1].plot(loss_smooth, color='red', linewidth=2, label=f'{window_loss}-step Average')
        axes[0, 1].set_title('Training Loss Over Time')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
    
    # Exploration decay
    episodes = len(scores)
    epsilons = [1.0 * (0.995 ** i) for i in range(episodes)]
    epsilons = [max(0.01, eps) for eps in epsilons]  # Apply minimum
    
    axes[0, 2].plot(epsilons, color='purple', linewidth=2)
    axes[0, 2].set_title('Exploration Rate Decay')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon (Exploration Rate)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Score distribution comparison
    early_scores = scores[:episodes//4]
    late_scores = scores[3*episodes//4:]
    
    axes[1, 0].hist(early_scores, alpha=0.7, color='red', bins=15, 
                   label='Early Training', density=True)
    axes[1, 0].hist(late_scores, alpha=0.7, color='green', bins=15,
                   label='Late Training', density=True)
    axes[1, 0].set_title('Score Distribution: Early vs Late')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode length over time
    axes[1, 1].plot(episode_lengths, alpha=0.6, color='teal')
    axes[1, 1].set_title('Episode Length Over Time')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps Survived')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Training statistics
    early_avg = np.mean(early_scores)
    late_avg = np.mean(late_scores)
    improvement = late_avg - early_avg
    success_rate = sum(1 for score in scores[-100:] if score >= 195) / 100
    
    stats_text = f"""
    DQN Training Results:
    
    Episodes: {episodes}
    Network: 4 → 128 → 128 → 2
    
    Early Avg (first 25%): {early_avg:.1f}
    Late Avg (last 25%): {late_avg:.1f}
    Improvement: +{improvement:.1f} points
    
    Best Score: {max(scores):.0f}
    Final 100-ep Avg: {np.mean(scores[-100:]):.1f}
    Success Rate (≥195): {success_rate:.1%}
    
    Memory Size: {agent.memory_size}
    Final Epsilon: {agent.epsilon:.3f}
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_title('Training Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/dqn_training_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"🎯 Success rate: {success_rate:.1%} (episodes scoring ≥195)")
    print(f"📊 Analysis chart saved to /tmp/dqn_training_analysis.png")

def compare_q_table_vs_dqn():
    """
    Compare our simple Q-table approach with DQN.
    
    Like comparing a regular puppy brain vs a super-computer puppy brain!
    """
    print("\n🔬 Q-TABLE VS DQN COMPARISON")
    print("=" * 60)
    print("| Aspect              | Q-Table          | Deep Q-Network    |")
    print("|--------------------|-----------------|--------------------|")
    print("| State Representation| Discrete bins   | Continuous values  |")
    print("| Memory Usage        | Small table     | Neural network     |")
    print("| Learning Method     | Direct updates  | Gradient descent   |")
    print("| Generalization      | Limited         | Excellent          |")
    print("| Training Time       | Fast            | Slower             |")
    print("| Final Performance   | Good            | Excellent          |")
    print("| Complexity          | Simple          | Advanced           |")
    print("| Scalability         | Poor            | Great              |")
    print()

def main():
    """Main DQN training and testing."""
    print("🧠🔥 DEEP Q-NETWORKS (DQN)")
    print("=" * 60)
    print("Welcome to the world of deep reinforcement learning!")
    print("Our puppy is getting a neural network brain upgrade! 🤖🐕")
    print()
    
    # Compare approaches
    compare_q_table_vs_dqn()
    
    # Train DQN agent
    agent, scores, episode_lengths = train_dqn_agent(episodes=800)
    
    # Analyze training
    analyze_dqn_training(scores, episode_lengths, agent)
    
    # Test the trained agent
    test_scores = test_dqn_agent(agent, episodes=10)
    
    print(f"\n🏆 FINAL RESULTS")
    print("=" * 50)
    print(f"Training average (last 100): {np.mean(scores[-100:]):.1f}")
    print(f"Test average: {np.mean(test_scores):.1f}")
    print(f"Best test score: {max(test_scores):.0f}")
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), '/tmp/dqn_cartpole.pth')
    print(f"🔧 Trained model saved to /tmp/dqn_cartpole.pth")
    
    print(f"\n🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ Neural networks can replace Q-tables for continuous states")
    print("✅ Experience replay enables learning from past memories") 
    print("✅ Target networks provide stable learning targets")
    print("✅ DQN can achieve superior performance on CartPole")
    print()
    print("🤔 KEY INSIGHTS:")
    print("- Deep learning enables handling complex, continuous environments")
    print("- Experience replay breaks correlation between consecutive samples")
    print("- Target networks prevent the 'moving target' problem")
    print("- Proper hyperparameters are crucial for stable learning")
    print()
    print("🚀 NEXT STEPS:")
    print("- Try 02_policy_gradient.py for direct policy learning!")
    print("- Experiment with different network architectures")
    print("- Try Double DQN, Dueling DQN, or Prioritized Experience Replay")

if __name__ == "__main__":
    main()
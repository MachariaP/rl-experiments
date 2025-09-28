#!/usr/bin/env python3
"""
🧠 Simple Learning Agent: Your First Smart Puppy!

Now we'll build an agent that learns from experience! This uses a simple
rule-based approach that improves over time. Think of it as a puppy that
starts to recognize patterns and gets better at tricks!

What you'll learn:
- How to create a learning policy
- Simple Q-table (value table) approach
- How agents improve through experience
- The difference between learning and memorizing

Think of this as: "Our puppy is starting to connect actions with results
and getting smarter about what works!"
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import time
from typing import Dict, List, Tuple

class SimpleQAgent:
    """
    A simple Q-Learning agent for CartPole.
    
    Think of this as a puppy's brain that:
    - Remembers what actions worked in similar situations
    - Gets more confident in good strategies over time
    - Still explores new possibilities sometimes
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1):
        """
        Initialize our learning puppy brain!
        
        Args:
            learning_rate: How quickly we learn from new experiences (0-1)
            discount_factor: How much we care about future rewards (0-1) 
            epsilon: How often we try random actions vs. known good ones (0-1)
        """
        self.learning_rate = learning_rate    # How fast we learn
        self.discount_factor = discount_factor # How much we care about the future
        self.epsilon = epsilon                # How often we explore vs exploit
        
        # Q-table: stores value estimates for state-action pairs
        # Like a puppy's memory: "In this situation, this action got me X treats"
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Track learning progress
        self.episode_count = 0
        
    def discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Convert continuous observations to discrete states.
        
        Like teaching a puppy to recognize "situations":
        - "Cart is far left" vs "Cart is in center" vs "Cart is far right"
        - "Pole is tilted left" vs "Pole is upright" vs "Pole is tilted right"
        
        This makes learning manageable by grouping similar situations together.
        """
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        # Create discrete bins for each observation
        # Think of these as "fuzzy categories" the puppy can understand
        
        # Cart position: left (-1), center (0), right (1)
        if cart_pos < -0.8:
            cart_pos_discrete = -1
        elif cart_pos > 0.8:
            cart_pos_discrete = 1
        else:
            cart_pos_discrete = 0
            
        # Cart velocity: moving left (-1), still (0), moving right (1)  
        if cart_vel < -0.5:
            cart_vel_discrete = -1
        elif cart_vel > 0.5:
            cart_vel_discrete = 1
        else:
            cart_vel_discrete = 0
            
        # Pole angle: tilted left (-1), upright (0), tilted right (1)
        if pole_angle < -0.05:
            pole_angle_discrete = -1
        elif pole_angle > 0.05:
            pole_angle_discrete = 1
        else:
            pole_angle_discrete = 0
            
        # Pole angular velocity: falling left (-1), stable (0), falling right (1)
        if pole_vel < -0.5:
            pole_vel_discrete = -1
        elif pole_vel > 0.5:
            pole_vel_discrete = 1
        else:
            pole_vel_discrete = 0
            
        return (cart_pos_discrete, cart_vel_discrete, 
                pole_angle_discrete, pole_vel_discrete)
    
    def choose_action(self, observation: np.ndarray) -> int:
        """
        Choose an action based on current knowledge and exploration.
        
        Like a puppy deciding what to do:
        - Most of the time: do what worked before (exploitation)
        - Sometimes: try something new (exploration)
        """
        state = self.discretize_state(observation)
        
        # Exploration vs Exploitation decision
        if np.random.random() < self.epsilon:
            # Explore: try a random action (curious puppy)
            action = np.random.choice([0, 1])
        else:
            # Exploit: use current knowledge (smart puppy)
            q_left = self.q_table[state][0]   # Expected reward for LEFT
            q_right = self.q_table[state][1]  # Expected reward for RIGHT
            
            if q_left > q_right:
                action = 0  # LEFT
            elif q_right > q_left:
                action = 1  # RIGHT  
            else:
                # If equal, choose randomly
                action = np.random.choice([0, 1])
                
        return action
    
    def learn(self, old_state: Tuple, action: int, reward: float, 
              new_state: Tuple, done: bool) -> None:
        """
        Update our knowledge based on what just happened.
        
        Like a puppy learning: "When I was in situation X and did action Y,
        I got reward Z and ended up in situation W. I should remember this!"
        
        This is the core of Q-learning!
        """
        # Current estimate of action value
        old_q_value = self.q_table[old_state][action]
        
        if done:
            # Episode ended - no future rewards possible
            target = reward
        else:
            # Episode continues - consider future rewards too
            # Find best action in new state
            best_next_q = max(self.q_table[new_state][0], 
                             self.q_table[new_state][1])
            target = reward + self.discount_factor * best_next_q
        
        # Update Q-value using learning rate
        # This is like gradually adjusting confidence based on new evidence
        new_q_value = old_q_value + self.learning_rate * (target - old_q_value)
        self.q_table[old_state][action] = new_q_value
    
    def decay_epsilon(self, decay_rate: float = 0.995) -> None:
        """
        Reduce exploration over time.
        
        Like a puppy becoming more confident and relying less on
        random experimentation as it gains experience.
        """
        self.epsilon = max(0.01, self.epsilon * decay_rate)

def train_agent(episodes: int = 1000) -> Tuple[SimpleQAgent, List[float], List[int]]:
    """
    Train our learning agent through many episodes.
    
    Like running multiple training sessions with our puppy until
    it becomes really good at the task!
    """
    print("🎓 TRAINING OUR LEARNING AGENT")
    print("=" * 50)
    print(f"Training for {episodes} episodes...")
    print("Watch as our puppy gets smarter over time! 🐕‍🦺")
    print()
    
    # Create environment and agent
    env = gym.make('CartPole-v1')
    agent = SimpleQAgent()
    
    # Track performance
    scores = []
    episode_lengths = []
    
    # Training loop
    for episode in range(episodes):
        observation, _ = env.reset()
        old_state = agent.discretize_state(observation)
        
        total_reward = 0
        steps = 0
        
        # Run one episode
        for step in range(500):  # Max 500 steps per episode
            # Agent chooses action
            action = agent.choose_action(observation)
            
            # Take action in environment
            new_observation, reward, terminated, truncated, _ = env.step(action)
            new_state = agent.discretize_state(new_observation)
            
            # Agent learns from this experience
            agent.learn(old_state, action, reward, new_state, 
                       terminated or truncated)
            
            # Update for next step
            observation = new_observation
            old_state = new_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Record performance
        scores.append(total_reward)
        episode_lengths.append(steps)
        
        # Reduce exploration over time
        agent.decay_epsilon()
        agent.episode_count += 1
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(scores[-100:])
            print(f"Episode {episode + 1:4d}: Recent 100 avg = {recent_avg:6.1f}, "
                  f"Epsilon = {agent.epsilon:.3f}")
    
    env.close()
    return agent, scores, episode_lengths

def test_agent(agent: SimpleQAgent, episodes: int = 10) -> List[float]:
    """
    Test our trained agent (no more learning).
    
    Like seeing how well our trained puppy performs in a test!
    """
    print(f"\n🧪 TESTING TRAINED AGENT ({episodes} episodes)")
    print("=" * 50)
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # Turn off exploration for testing
    test_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No random actions during test
    
    test_scores = []
    
    for episode in range(episodes):
        observation, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        test_scores.append(total_reward)
        print(f"Test episode {episode + 1}: Score = {total_reward}")
    
    # Restore original epsilon
    agent.epsilon = test_epsilon
    env.close()
    
    return test_scores

def analyze_learning_progress(scores: List[float], episode_lengths: List[int]) -> None:
    """
    Analyze how the agent learned over time.
    
    Like reviewing our puppy's progress reports! 📊
    """
    print(f"\n📊 LEARNING ANALYSIS")
    print("=" * 50)
    
    # Calculate rolling averages
    window = 100
    rolling_avg = []
    for i in range(len(scores)):
        start_idx = max(0, i - window + 1)
        rolling_avg.append(np.mean(scores[start_idx:i+1]))
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning curve
    axes[0, 0].plot(scores, alpha=0.3, color='lightblue', label='Episode Score')
    axes[0, 0].plot(rolling_avg, color='red', linewidth=2, label=f'{window}-episode Average')
    axes[0, 0].set_title('Learning Progress Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score distribution comparison (early vs late)
    early_scores = scores[:len(scores)//4]  # First 25%
    late_scores = scores[3*len(scores)//4:]  # Last 25%
    
    axes[0, 1].hist(early_scores, alpha=0.7, color='red', bins=20, 
                   label='Early Training', density=True)
    axes[0, 1].hist(late_scores, alpha=0.7, color='green', bins=20,
                   label='Late Training', density=True)
    axes[0, 1].set_title('Score Distribution: Early vs Late Training')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode length over time
    axes[1, 0].plot(episode_lengths, alpha=0.6, color='purple')
    axes[1, 0].set_title('Episode Length Over Time')
    axes[1, 0].set_xlabel('Episode')  
    axes[1, 0].set_ylabel('Steps Survived')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance statistics
    episodes = len(scores)
    early_avg = np.mean(early_scores)
    late_avg = np.mean(late_scores)
    improvement = late_avg - early_avg
    
    stats_text = f"""
    Training Statistics:
    
    Total Episodes: {episodes}
    
    Early Performance (first 25%):
    Average Score: {early_avg:.1f}
    
    Late Performance (last 25%):  
    Average Score: {late_avg:.1f}
    
    Improvement: +{improvement:.1f} points
    ({improvement/early_avg*100:.1f}% better)
    
    Best Score: {max(scores):.0f}
    Final 100-episode avg: {np.mean(scores[-100:]):.1f}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/learning_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Performance improved by {improvement:.1f} points!")
    print(f"📊 Chart saved to /tmp/learning_progress.png")

def main():
    """Main training and testing loop."""
    print("🧠 SIMPLE LEARNING AGENT")
    print("=" * 60)
    print("Time to build our first agent that learns from experience!")
    print("This is like training a puppy that gets smarter over time! 🐕")
    print()
    
    # Train the agent
    agent, scores, episode_lengths = train_agent(episodes=1000)
    
    # Analyze learning progress
    analyze_learning_progress(scores, episode_lengths)
    
    # Test the trained agent
    test_scores = test_agent(agent, episodes=10)
    
    print(f"\n🏆 FINAL RESULTS")
    print("=" * 50)
    print(f"Training average (last 100): {np.mean(scores[-100:]):.1f}")
    print(f"Test average: {np.mean(test_scores):.1f}")
    print(f"Best test score: {max(test_scores):.0f}")
    
    # Save the trained agent
    with open('/tmp/simple_q_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print(f"🔧 Trained agent saved to /tmp/simple_q_agent.pkl")
    
    print(f"\n🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ Agents can learn from experience!")
    print("✅ Performance improves with more training")
    print("✅ Exploration vs exploitation is crucial")
    print("✅ Simple Q-learning can solve CartPole")
    print()
    print("🤔 KEY INSIGHTS:")
    print("- Learning takes time but leads to better performance")
    print("- Discretizing continuous spaces enables tabular learning") 
    print("- Balancing exploration and exploitation is an art")
    print("- Even simple learning algorithms can be effective")
    print()
    print("🚀 NEXT STEPS:")
    print("- Try intermediate/01_dqn_cartpole.py for neural networks!")
    print("- Experiment with different learning parameters")
    print("- Build custom learning algorithms")

if __name__ == "__main__":
    main()
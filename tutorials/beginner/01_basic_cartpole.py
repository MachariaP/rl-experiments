#!/usr/bin/env python3
"""
🎯 Your First CartPole Agent: Random Actions

This is your very first RL agent! It's like a confused puppy that just
does random things. Don't worry - even random actions teach us a lot!

What you'll learn:
- How to create a Gymnasium environment
- How to run an episode (complete game)
- What observations and actions look like
- How rewards work in practice

Think of this as: "Let's see what happens when our puppy just does 
random things in a new environment!"
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

def create_cartpole_environment() -> gym.Env:
    """
    Create our CartPole environment.
    
    Think of this as setting up the training area for our puppy!
    We can choose to render it visually or keep it hidden for speed.
    """
    # render_mode options:
    # - 'human': Shows a window (slow but fun to watch!)  
    # - 'rgb_array': Creates images we can save (good for recording)
    # - None: No visualization (fastest for training)
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    print(f"🎮 Created CartPole environment!")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"⚡ Action space: {env.action_space}")
    print()
    
    return env

def random_policy(observation: np.ndarray) -> int:
    """
    Our random policy: just pick LEFT or RIGHT randomly.
    
    This is like a puppy who doesn't know what to do yet,
    so it just tries random things!
    
    Args:
        observation: What the agent sees (cart position, velocity, pole angle, etc.)
        
    Returns:
        action: 0 (LEFT) or 1 (RIGHT)
    """
    # We don't even look at the observation - just random!
    action = np.random.choice([0, 1])
    return action

def run_episode(env: gym.Env, max_steps: int = 500) -> Tuple[float, int, List[float]]:
    """
    Run one complete episode (game) until the pole falls or we reach max_steps.
    
    This is like one training session with our puppy!
    
    Returns:
        total_reward: How many points we scored
        steps: How long we lasted
        rewards: List of rewards at each step
    """
    # Reset environment to starting position
    observation, info = env.reset()
    
    total_reward = 0.0
    steps = 0
    rewards = []
    
    print("🏁 Starting new episode...")
    
    for step in range(max_steps):
        # 1. Agent decides what to do based on what it sees
        action = random_policy(observation)
        
        # 2. Take the action in the environment  
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 3. Collect rewards and track progress
        total_reward += reward
        rewards.append(reward)
        steps = step + 1
        
        # Print progress every 50 steps
        if step % 50 == 0:
            action_name = "LEFT" if action == 0 else "RIGHT"
            print(f"   Step {step:3d}: Action={action_name}, Reward={reward}, Total={total_reward}")
        
        # 4. Check if episode is over
        if terminated or truncated:
            if terminated:
                print(f"💥 Episode ended: Pole fell over after {steps} steps!")
            else:
                print(f"⏰ Episode ended: Reached max steps ({steps})!")
            break
    
    print(f"🏆 Final score: {total_reward} points in {steps} steps")
    print()
    
    return total_reward, steps, rewards

def analyze_performance(scores: List[float], episode_lengths: List[int]) -> None:
    """
    Analyze how our random agent performed.
    
    Like reviewing how our puppy did in training sessions!
    """
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    print(f"📈 Episodes run: {len(scores)}")
    print(f"🎯 Average score: {np.mean(scores):.1f}")
    print(f"🏆 Best score: {np.max(scores):.1f}")
    print(f"😞 Worst score: {np.min(scores):.1f}")
    print(f"📏 Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"🔥 Longest episode: {np.max(episode_lengths)} steps")
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Plot episode scores over time
    plt.subplot(1, 2, 1)
    plt.plot(scores, 'b-', alpha=0.7, linewidth=1)
    plt.axhline(y=np.mean(scores), color='r', linestyle='--', 
                label=f'Average: {np.mean(scores):.1f}')
    plt.title('Random Agent Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of scores
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=np.mean(scores), color='r', linestyle='--',
                label=f'Average: {np.mean(scores):.1f}')
    plt.title('Distribution of Scores')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/random_agent_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Performance chart saved to /tmp/random_agent_performance.png")

def main():
    """
    Main function: Let's train our random puppy!
    """
    print("🎯 WELCOME TO YOUR FIRST RL AGENT!")
    print("=" * 50)
    print("We're going to create a 'puppy' that tries random actions")
    print("in the CartPole environment. Let's see what happens!")
    print()
    
    # Create environment
    env = create_cartpole_environment()
    
    # Run multiple episodes to see how random actions perform
    num_episodes = 10
    scores = []
    episode_lengths = []
    
    print(f"🏃‍♂️ Running {num_episodes} episodes with random actions...")
    print()
    
    for episode in range(num_episodes):
        print(f"🎬 EPISODE {episode + 1}/{num_episodes}")
        print("-" * 30)
        
        score, length, _ = run_episode(env, max_steps=500)
        scores.append(score)
        episode_lengths.append(length)
        
        # Small pause to make it easier to follow
        time.sleep(0.5)
    
    # Analyze results
    analyze_performance(scores, episode_lengths)
    
    # Close environment
    env.close()
    
    print("🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ Random actions sometimes work by accident!")
    print("✅ Most episodes are short (pole falls quickly)")
    print("✅ Occasionally we get lucky and last longer")
    print("✅ Average score is usually around 20-30 points")
    print()
    print("🤔 THINK ABOUT IT:")
    print("- Could we do better than random?")
    print("- What if our agent could learn from its mistakes?")
    print("- What patterns might lead to better balance?")
    print()
    print("🚀 NEXT STEP: Check out '02_understanding_cartpole.py'")
    print("   to explore what our agent can actually see!")

if __name__ == "__main__":
    main()
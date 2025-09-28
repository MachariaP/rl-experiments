#!/usr/bin/env python3
"""
🔍 Understanding the CartPole Environment

Now that we've seen random actions, let's understand what our agent
actually sees and how the environment works. This is like studying
what your puppy can sense before teaching it tricks!

What you'll learn:
- The 4 observations the agent receives
- How actions affect the environment  
- What different states look like
- How to interpret the numbers

Think of this as: "Let's understand what our puppy can see, hear, 
and feel in its environment before we teach it what to do!"
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict

def create_environment():
    """Create CartPole environment for exploration."""
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    return env

def explain_observation_space(env: gym.Env) -> None:
    """
    Explain what the agent can observe.
    
    Like explaining to someone what your puppy can see!
    """
    print("👀 WHAT CAN OUR AGENT SEE?")
    print("=" * 50)
    print("The agent receives 4 numbers every step:")
    print()
    
    print("1. 📍 CART POSITION (x-coordinate)")
    print("   • Range: -4.8 to +4.8 (approximately)")
    print("   • Center is 0, negative = left, positive = right")
    print("   • Like asking: 'Where is the cart on the track?'")
    print()
    
    print("2. 🏃 CART VELOCITY (speed left/right)")  
    print("   • Range: -∞ to +∞ (but usually -3 to +3)")
    print("   • Negative = moving left, positive = moving right")
    print("   • Like asking: 'How fast is the cart moving?'")
    print()
    
    print("3. 📐 POLE ANGLE (tilt from vertical)")
    print("   • Range: -0.418 to +0.418 radians (about ±24 degrees)")
    print("   • 0 = perfectly upright")
    print("   • Negative = tilting left, positive = tilting right") 
    print("   • Like asking: 'How tilted is the pole?'")
    print()
    
    print("4. 🌪️ POLE ANGULAR VELOCITY (how fast it's falling)")
    print("   • Range: -∞ to +∞ (but usually -3 to +3)")
    print("   • Negative = falling left, positive = falling right")
    print("   • Like asking: 'How fast is the pole tipping over?'")
    print()

def explain_action_space(env: gym.Env) -> None:
    """Explain what actions the agent can take."""
    print("⚡ WHAT ACTIONS CAN OUR AGENT TAKE?")
    print("=" * 50)
    print("The agent has 2 choices every step:")
    print()
    print("🔵 Action 0: PUSH LEFT")
    print("   • Apply force to move cart leftward")
    print("   • Might help if pole is falling right")
    print()
    print("🔴 Action 1: PUSH RIGHT") 
    print("   • Apply force to move cart rightward")
    print("   • Might help if pole is falling left")
    print()
    print("💡 Strategy hint: Move the cart under the falling pole!")
    print()

def demonstrate_observations(env: gym.Env, steps: int = 20) -> List[Dict]:
    """
    Show what observations look like during actual gameplay.
    
    Like watching what your puppy sees during a training session!
    """
    print("🎬 LIVE OBSERVATION DEMO")
    print("=" * 50)
    print("Let's see what the agent observes during real gameplay:")
    print()
    
    # Reset environment
    observation, _ = env.reset()
    observations_data = []
    
    print("Format: [Cart Pos, Cart Vel, Pole Angle, Pole Vel] -> Action -> Reward")
    print("-" * 80)
    
    for step in range(steps):
        # Choose a simple action: try to balance
        if observation[2] > 0:  # If pole tilts right
            action = 1  # Push right to get under it
            action_name = "RIGHT"
        else:  # If pole tilts left  
            action = 0  # Push left to get under it
            action_name = "LEFT"
        
        # Store observation data
        obs_data = {
            'step': step,
            'observation': observation.copy(),
            'action': action,
            'action_name': action_name
        }
        
        # Take action
        new_observation, reward, terminated, truncated, _ = env.step(action)
        obs_data['reward'] = reward
        obs_data['terminated'] = terminated
        
        observations_data.append(obs_data)
        
        # Print observation in readable format
        pos, vel, angle, ang_vel = observation
        print(f"Step {step:2d}: [{pos:6.3f}, {vel:6.3f}, {angle:6.3f}, {ang_vel:6.3f}] "
              f"-> {action_name:5} -> Reward: {reward}")
        
        observation = new_observation
        
        if terminated or truncated:
            print(f"\n💥 Episode ended at step {step + 1}!")
            break
        
        time.sleep(0.1)  # Small delay to make it readable
    
    print()
    return observations_data

def analyze_observation_patterns(observations_data: List[Dict]) -> None:
    """
    Analyze patterns in the observations.
    
    Like studying your puppy's behavior to understand what it's learning!
    """
    print("📊 OBSERVATION PATTERNS ANALYSIS")
    print("=" * 50)
    
    if not observations_data:
        print("No data to analyze!")
        return
    
    # Extract observation arrays
    observations = np.array([obs['observation'] for obs in observations_data])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = range(len(observations))
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    units = ['position', 'velocity', 'radians', 'radians/sec']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i in range(4):
        ax = axes[i//2, i%2]
        values = observations[:, i]
        
        ax.plot(steps, values, color=colors[i], linewidth=2, marker='o', markersize=4)
        ax.set_title(f'{labels[i]} Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel(f'{labels[i]} ({units[i]})')
        ax.grid(True, alpha=0.3)
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Highlight critical values
        if i == 2:  # Pole angle
            ax.axhline(y=0.21, color='red', linestyle=':', alpha=0.7, label='Fail threshold')
            ax.axhline(y=-0.21, color='red', linestyle=':', alpha=0.7)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/cartpole_observations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"📈 OBSERVATION STATISTICS:")
    print("-" * 30)
    for i, label in enumerate(labels):
        values = observations[:, i]
        print(f"{label}:")
        print(f"  Range: {np.min(values):.3f} to {np.max(values):.3f}")
        print(f"  Average: {np.mean(values):.3f}")
        print(f"  Std Dev: {np.std(values):.3f}")
        print()

def show_failure_conditions(env: gym.Env) -> None:
    """
    Demonstrate what causes the episode to end.
    
    Like showing when your puppy gets a time-out!
    """
    print("❌ WHEN DOES THE EPISODE END?")
    print("=" * 50)
    print("The episode ends when ANY of these happen:")
    print()
    print("1. 📐 POLE ANGLE too large:")
    print("   • Angle > 12 degrees (0.2094 radians)")
    print("   • Pole has fallen over!")
    print()
    print("2. 📍 CART POSITION too far:")
    print("   • Position < -2.4 or > +2.4")
    print("   • Cart fell off the track!")
    print()
    print("3. ⏰ TIME LIMIT reached:")
    print("   • 500 steps completed")
    print("   • This counts as SUCCESS!")
    print()
    
    # Show the actual limits from environment
    print("🔍 TECHNICAL DETAILS:")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print()

def interactive_exploration(env: gym.Env) -> None:
    """
    Let user explore the environment interactively.
    
    Like letting someone play with your puppy to understand it better!
    """
    print("🎮 INTERACTIVE EXPLORATION")
    print("=" * 50)
    print("Let's manually control the cart and see what happens!")
    print()
    print("Controls:")
    print("  0 or 'a' = Push LEFT")
    print("  1 or 'd' = Push RIGHT") 
    print("  'q' = Quit")
    print()
    
    observation, _ = env.reset()
    step = 0
    
    while True:
        # Show current state
        pos, vel, angle, ang_vel = observation
        print(f"\nStep {step}: Cart at {pos:.3f}, moving {vel:.3f}, "
              f"pole angle {angle:.3f}, falling at {ang_vel:.3f}")
        
        # Get user input
        try:
            user_input = input("Enter action (0/1 or a/d or q): ").strip().lower()
            
            if user_input in ['q', 'quit', 'exit']:
                break
            elif user_input in ['0', 'a', 'left']:
                action = 0
                action_name = "LEFT"
            elif user_input in ['1', 'd', 'right']:
                action = 1  
                action_name = "RIGHT"
            else:
                print("Invalid input! Use 0/1 or a/d or q")
                continue
                
        except (KeyboardInterrupt, EOFError):
            break
        
        # Take action
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f"Action: {action_name}, Reward: {reward}")
        
        step += 1
        
        if terminated or truncated:
            if terminated:
                print("\n💥 Episode ended - pole fell or cart went too far!")
            else:
                print("\n🏆 Episode ended - reached time limit (success)!")
            break
    
    print("\nThanks for exploring! 🎉")

def main():
    """Main function: explore the CartPole environment!"""
    print("🔍 UNDERSTANDING THE CARTPOLE ENVIRONMENT")
    print("=" * 60)
    print("Time to understand what our agent can see and do!")
    print("This is like studying your puppy's senses before training.")
    print()
    
    # Create environment
    env = create_environment()
    
    # Explain the observation and action spaces
    explain_observation_space(env)
    explain_action_space(env)
    show_failure_conditions(env)
    
    # Demonstrate live observations
    print("Press Enter to see live observations...")
    input()
    observations = demonstrate_observations(env, steps=30)
    
    # Analyze patterns
    if observations:
        analyze_performance = input("\nAnalyze observation patterns? (y/n): ").lower()
        if analyze_performance.startswith('y'):
            analyze_observation_patterns(observations)
    
    # Interactive exploration
    interactive = input("\nTry interactive control? (y/n): ").lower()
    if interactive.startswith('y'):
        interactive_exploration(env)
    
    # Close environment
    env.close()
    
    print("\n🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ The agent sees 4 continuous numbers each step")
    print("✅ Small angle changes can lead to big consequences")
    print("✅ Cart velocity and pole velocity are crucial")
    print("✅ The goal is to keep everything balanced")
    print()
    print("🤔 KEY INSIGHTS:")
    print("- Position alone isn't enough - velocity matters!")
    print("- Pole angle and angular velocity predict the future")
    print("- Actions need to be timely and appropriate")
    print("- Random actions work sometimes but not consistently")
    print()
    print("🚀 NEXT STEP: Check out '03_simple_learning.py'")
    print("   to build an agent that learns from experience!")

if __name__ == "__main__":
    main()
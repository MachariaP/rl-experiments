#!/usr/bin/env python3
"""
🎮 Demo: Pre-trained CartPole Agent

This script shows a pre-trained CartPole agent in action! It's designed to
work right out of the box, even if you haven't trained any models yet.

Perfect for:
- First-time users who want to see RL in action immediately
- Demonstrating what a trained agent looks like
- Understanding what we're trying to achieve

Think of this as: "Meet our star graduate puppy who has mastered CartPole!"
"""

import gymnasium as gym
import numpy as np
import time
import sys
import os

def create_demo_agent():
    """
    Create a simple rule-based agent that demonstrates good CartPole behavior.
    
    This isn't a trained ML model, but a hand-crafted policy that shows
    what good CartPole performance looks like. Think of it as a "cheating"
    puppy that already knows the rules!
    """
    
    def demo_policy(observation):
        """
        Hand-crafted policy for CartPole.
        
        Strategy: Move the cart towards the direction the pole is falling.
        This is what a good RL agent learns to do!
        """
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        # If pole is falling right (positive angle), push right to catch it
        # If pole is falling left (negative angle), push left to catch it
        # Also consider angular velocity for better predictions
        
        if pole_angle + pole_vel * 0.1 > 0:  # Pole falling/will fall right
            action = 1  # Push right
        else:  # Pole falling/will fall left
            action = 0  # Push left
            
        return action
    
    return demo_policy

def run_demo_episode(policy, env, max_steps=500, render=True, step_delay=0.02):
    """
    Run one demonstration episode.
    
    Args:
        policy: The policy function to use
        env: The CartPole environment
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        step_delay: Delay between steps (for viewing)
    
    Returns:
        total_reward: Score achieved
        steps: Number of steps survived
    """
    observation, _ = env.reset()
    total_reward = 0
    
    print(f"🎬 Starting demo episode...")
    
    for step in range(max_steps):
        # Choose action using our demo policy
        action = policy(observation)
        
        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print progress occasionally
        if step % 50 == 0 and step > 0:
            cart_pos, cart_vel, pole_angle, pole_vel = observation
            action_name = "LEFT" if action == 0 else "RIGHT"
            print(f"   Step {step:3d}: Action={action_name}, Pole Angle={pole_angle:6.3f}, Score={total_reward}")
        
        # Add delay for viewing if rendering
        if render and step_delay > 0:
            time.sleep(step_delay)
        
        # Check if episode ended
        if terminated or truncated:
            if terminated:
                print(f"💥 Episode ended: Pole fell over after {step + 1} steps!")
            else:
                print(f"🏆 Episode completed: Reached maximum steps ({step + 1})!")
            break
    
    return total_reward, step + 1

def run_multiple_demos(n_episodes=5, render=True):
    """
    Run multiple demonstration episodes and show statistics.
    
    This gives you a good sense of what consistent good performance looks like.
    """
    print("🎮 CARTPOLE DEMO: PRE-TRAINED AGENT")
    print("=" * 60)
    print("Watching our 'expert' agent that already knows how to balance!")
    print("This shows what we're trying to teach our learning agents to do.")
    print()
    
    # Create environment
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    # Create our demo agent
    policy = create_demo_agent()
    
    # Run episodes
    scores = []
    episode_lengths = []
    
    print(f"🚀 Running {n_episodes} demonstration episodes...")
    print()
    
    for episode in range(n_episodes):
        print(f"📺 EPISODE {episode + 1}/{n_episodes}")
        print("-" * 40)
        
        score, length = run_demo_episode(
            policy, 
            env, 
            render=render, 
            step_delay=0.01 if render else 0
        )
        
        scores.append(score)
        episode_lengths.append(length)
        
        print(f"Final score: {score}")
        print()
        
        # Short pause between episodes
        if render:
            time.sleep(1.0)
    
    # Show statistics
    print("📊 DEMONSTRATION STATISTICS")
    print("=" * 40)
    print(f"Episodes run: {len(scores)}")
    print(f"Average score: {np.mean(scores):.1f}")
    print(f"Best score: {max(scores):.0f}")
    print(f"Worst score: {min(scores):.0f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"Success rate (≥195 points): {sum(1 for s in scores if s >= 195) / len(scores) * 100:.0f}%")
    
    # Close environment
    env.close()
    
    return scores, episode_lengths

def explain_cartpole_basics():
    """Explain what the user is seeing in the demo."""
    print("\n🎯 WHAT ARE YOU SEEING?")
    print("=" * 50)
    print("The CartPole Challenge:")
    print("• A cart (rectangle) that can move left/right on a track")
    print("• A pole (line) balanced on top of the cart")
    print("• Goal: Keep the pole upright as long as possible")
    print()
    print("The Agent's Strategy:")
    print("• Watch which way the pole is tilting")  
    print("• Move the cart under the falling pole")
    print("• React quickly to keep the pole balanced")
    print()
    print("Success Criteria:")
    print("• Episode ends if pole tilts > 12 degrees")
    print("• Episode ends if cart moves too far (±2.4 units)")
    print("• Maximum episode length is 500 steps")
    print("• Score of 195+ over 100 episodes = 'solved'")
    print()
    print("Why This Matters:")
    print("• CartPole teaches fundamental RL concepts")
    print("• Simple enough to understand, complex enough to be interesting")
    print("• Skills transfer to more complex control problems")

def interactive_mode():
    """
    Let user control the agent interactively.
    
    This helps users understand what the agent needs to learn!
    """
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Now YOU try to control the CartPole!")
    print("Use these controls:")
    print("  'a' or '0' = Push cart LEFT")
    print("  'd' or '1' = Push cart RIGHT") 
    print("  'q' = Quit")
    print()
    print("Try to keep the pole balanced. It's harder than it looks!")
    print()
    
    env = gym.make('CartPole-v1', render_mode='human')
    observation, _ = env.reset()
    step = 0
    total_reward = 0
    
    print("🏁 Starting interactive episode...")
    print("Press Enter when you're ready...")
    input()
    
    try:
        while True:
            # Show current state
            cart_pos, cart_vel, pole_angle, pole_vel = observation
            print(f"\nStep {step}: Pole angle = {pole_angle:.3f}, Cart position = {cart_pos:.3f}")
            
            # Get user input
            user_input = input("Action (a/d/q): ").strip().lower()
            
            if user_input in ['q', 'quit', 'exit']:
                print("Thanks for playing! 👋")
                break
            elif user_input in ['a', '0', 'left']:
                action = 0
                action_name = "LEFT"
            elif user_input in ['d', '1', 'right']:
                action = 1
                action_name = "RIGHT"
            else:
                print("Invalid input! Use a/d or q")
                continue
            
            # Take action
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1
            
            print(f"Action: {action_name}, Reward: {reward}, Total: {total_reward}")
            
            if terminated or truncated:
                if terminated:
                    print(f"\n💥 Game Over! Pole fell after {step} steps.")
                else:
                    print(f"\n🏆 Success! You lasted the full {step} steps!")
                print(f"Final score: {total_reward}")
                break
    
    except (KeyboardInterrupt, EOFError):
        print("\nThanks for playing! 👋")
    
    env.close()

def main():
    """Main demo function."""
    print("🎮 CARTPOLE REINFORCEMENT LEARNING DEMO")
    print("=" * 60)
    print("Welcome! This demo shows what a trained CartPole agent looks like.")
    print("Perfect for first-time users - no training required!")
    print()
    
    # Explain the basics
    explain_cartpole_basics()
    
    # Ask user for demo preferences
    print("\n🎬 DEMO OPTIONS")
    print("=" * 30)
    
    try:
        # Check if display is available
        render_available = True
        try:
            import tkinter
        except ImportError:
            render_available = False
    except:
        render_available = False
    
    if render_available:
        render_choice = input("Show visual demo? (y/n, default=y): ").strip().lower()
        render = not render_choice.startswith('n')
    else:
        print("Visual rendering not available. Running text-only demo.")
        render = False
    
    episodes_input = input("How many episodes to run? (default=3): ").strip()
    try:
        n_episodes = int(episodes_input) if episodes_input else 3
        n_episodes = max(1, min(10, n_episodes))  # Limit between 1-10
    except ValueError:
        n_episodes = 3
    
    print()
    
    # Run the demo
    scores, lengths = run_multiple_demos(n_episodes=n_episodes, render=render)
    
    # Offer interactive mode
    if render_available:
        interactive_choice = input("\nTry interactive mode? (y/n): ").strip().lower()
        if interactive_choice.startswith('y'):
            interactive_mode()
    
    print(f"\n🎓 WHAT'S NEXT?")
    print("=" * 50)
    print("Now that you've seen what success looks like, try:")
    print()
    print("📚 Learning Path:")
    print("1. Read 'tutorials/beginner/concepts.md' - Learn RL basics")
    print("2. Run 'tutorials/beginner/01_basic_cartpole.py' - Random agent")
    print("3. Run 'tutorials/beginner/02_understanding_cartpole.py' - Explore environment")
    print("4. Work through the full tutorial series!")
    print()
    print("🎯 Goals to achieve:")
    print(f"• Beat random performance (~22 points average)")
    print(f"• Reach human-level performance (~100+ points)")
    print(f"• Achieve 'solved' status (195+ points consistently)")
    print(f"• Understand why it works!")
    print()
    print("Happy learning! 🚀🤖")

if __name__ == "__main__":
    main()
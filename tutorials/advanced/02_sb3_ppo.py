#!/usr/bin/env python3
"""
🏭 Production-Ready PPO with Stable-Baselines3

Now let's use the industry-standard implementation of PPO! Stable-Baselines3
is used by companies and researchers worldwide. This is like switching from
building a car by hand to using a professional assembly line.

What you'll learn:
- How to use Stable-Baselines3 (SB3)
- Production-ready RL training
- Hyperparameter tuning
- Model saving and loading
- Training monitoring with TensorBoard

Think of this as: "Using professional-grade tools that real AI companies
use to train their puppy agents at scale!"
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import torch
import os
import time
from typing import List, Tuple

# Create directories for saving models and logs
os.makedirs('/tmp/sb3_models', exist_ok=True)
os.makedirs('/tmp/sb3_logs', exist_ok=True)

def create_cartpole_env(n_envs: int = 1, monitor: bool = True):
    """
    Create CartPole environment(s) for training.
    
    SB3 can train on multiple environments in parallel for faster learning!
    This is like training multiple puppies simultaneously.
    
    Args:
        n_envs: Number of parallel environments
        monitor: Whether to monitor training progress
    """
    if n_envs == 1:
        env = gym.make('CartPole-v1')
        if monitor:
            env = Monitor(env, '/tmp/sb3_logs')
        return env
    else:
        # Multiple parallel environments
        env = make_vec_env('CartPole-v1', n_envs=n_envs, monitor_dir='/tmp/sb3_logs' if monitor else None)
        return env

def train_sb3_ppo_basic(total_timesteps: int = 100000) -> PPO:
    """
    Train PPO using Stable-Baselines3 with default settings.
    
    This is the simplest way to get started with professional RL!
    """
    print("🏭 TRAINING SB3 PPO (BASIC)")
    print("=" * 50)
    print(f"Training for {total_timesteps} timesteps...")
    print("Using industry-standard Stable-Baselines3! 🏗️🤖")
    print()
    
    # Create environment
    env = create_cartpole_env()
    
    # Create PPO model with default hyperparameters
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='/tmp/sb3_logs/')
    
    print("🚀 Starting training...")
    start_time = time.time()
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Save the model
    model.save('/tmp/sb3_models/ppo_cartpole_basic')
    
    env.close()
    return model

def train_sb3_ppo_advanced(total_timesteps: int = 200000) -> PPO:
    """
    Train PPO with advanced features and monitoring.
    
    This showcases the full power of SB3 for production use!
    """
    print("\n🔥 TRAINING SB3 PPO (ADVANCED)")
    print("=" * 50)
    print(f"Training for {total_timesteps} timesteps with advanced features...")
    print()
    
    # Create multiple parallel environments for faster training
    train_env = create_cartpole_env(n_envs=4, monitor=True)
    eval_env = create_cartpole_env(n_envs=1, monitor=False)
    
    # Custom hyperparameters (tuned for CartPole)
    custom_hyperparams = {
        'policy': 'MlpPolicy',
        'env': train_env,
        'learning_rate': 3e-4,
        'n_steps': 2048,          # Steps collected per environment per update
        'batch_size': 64,         # Minibatch size
        'n_epochs': 10,           # Number of epoch when optimizing the surrogate
        'gamma': 0.99,            # Discount factor
        'gae_lambda': 0.95,       # GAE lambda parameter
        'clip_range': 0.2,        # Clipping parameter
        'ent_coef': 0.01,         # Entropy coefficient for the loss calculation
        'vf_coef': 0.5,           # Value function coefficient for the loss calculation
        'max_grad_norm': 0.5,     # Maximum value for the gradient clipping
        'verbose': 1,
        'tensorboard_log': '/tmp/sb3_logs/'
    }
    
    print("🎛️ HYPERPARAMETERS:")
    for key, value in custom_hyperparams.items():
        if key not in ['env', 'policy']:
            print(f"   {key}: {value}")
    print()
    
    # Create model
    model = PPO(**custom_hyperparams)
    
    # Setup callbacks for advanced monitoring
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='/tmp/sb3_models/',
        log_path='/tmp/sb3_logs/',
        eval_freq=10000,          # Evaluate every 10000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Early stopping when reaching target performance
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=195,     # Stop when average reward >= 195
        verbose=1
    )
    
    print("🚀 Starting advanced training with callbacks...")
    start_time = time.time()
    
    # Train with callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, stop_callback]
    )
    
    training_time = time.time() - start_time
    print(f"✅ Advanced training completed in {training_time:.1f} seconds")
    
    # Save final model
    model.save('/tmp/sb3_models/ppo_cartpole_advanced')
    
    train_env.close()
    eval_env.close()
    return model

def evaluate_sb3_model(model: PPO, n_episodes: int = 10, render: bool = False) -> List[float]:
    """
    Evaluate trained SB3 model.
    
    SB3 makes evaluation super easy!
    """
    print(f"\n🧪 EVALUATING SB3 MODEL ({n_episodes} episodes)")
    print("=" * 50)
    
    render_mode = 'human' if render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    scores = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            # SB3 makes prediction simple
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if render:
                time.sleep(0.01)
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        print(f"Episode {episode + 1}: Score = {total_reward}")
        
        if render:
            time.sleep(1)
    
    env.close()
    return scores

def load_and_test_model(model_path: str) -> PPO:
    """
    Load a saved model and test it.
    
    This shows how easy it is to deploy trained models!
    """
    print(f"\n📂 LOADING SAVED MODEL")
    print("=" * 50)
    print(f"Loading model from: {model_path}")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    print("✅ Model loaded successfully!")
    
    # Test the loaded model
    scores = evaluate_sb3_model(model, n_episodes=5)
    
    print(f"📊 Loaded model performance: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    
    return model

def analyze_training_progress():
    """
    Analyze training progress using SB3's monitoring data.
    
    SB3 automatically logs training metrics that we can visualize!
    """
    print(f"\n📊 ANALYZING TRAINING PROGRESS")
    print("=" * 50)
    
    try:
        # Load training results
        results = load_results('/tmp/sb3_logs')
        
        if len(results) > 0:
            # Extract data
            x, y = ts2xy(results, 'timesteps')
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Reward over time
            axes[0].plot(x, y)
            axes[0].set_xlabel('Timesteps')
            axes[0].set_ylabel('Episode Reward')
            axes[0].set_title('SB3 PPO Training Progress')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Moving average
            window = 100
            if len(y) >= window:
                moving_avg = []
                for i in range(len(y)):
                    start_idx = max(0, i - window + 1)
                    moving_avg.append(np.mean(y[start_idx:i+1]))
                
                axes[1].plot(x, y, alpha=0.3, label='Episode Reward')
                axes[1].plot(x, moving_avg, color='red', linewidth=2, label=f'{window}-episode Average')
                axes[1].axhline(y=195, color='green', linestyle='--', label='Success Threshold')
                axes[1].set_xlabel('Timesteps')
                axes[1].set_ylabel('Episode Reward')
                axes[1].set_title('Training Progress with Moving Average')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/tmp/sb3_training_progress.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Print statistics
            print(f"📈 Training Statistics:")
            print(f"   Total episodes: {len(y)}")
            print(f"   Average reward: {np.mean(y):.1f}")
            print(f"   Best reward: {np.max(y):.1f}")
            print(f"   Final 100-episode average: {np.mean(y[-100:]):.1f}")
            print(f"📊 Chart saved to /tmp/sb3_training_progress.png")
        else:
            print("No training data found. Make sure training completed successfully.")
    
    except Exception as e:
        print(f"Could not load training results: {e}")

def hyperparameter_tuning_demo():
    """
    Demonstrate hyperparameter tuning principles.
    
    In practice, you'd use tools like Optuna for automated tuning.
    """
    print(f"\n🎛️ HYPERPARAMETER TUNING GUIDE")
    print("=" * 60)
    print("Key PPO hyperparameters and their effects:")
    print()
    
    hyperparams_guide = {
        'learning_rate': {
            'typical_values': '3e-4 to 3e-3',
            'effect': 'Higher = faster learning but less stable',
            'tune_if': 'Learning too slow/fast or unstable'
        },
        'n_steps': {
            'typical_values': '512 to 4096',
            'effect': 'More steps = better estimates but slower updates',
            'tune_if': 'Need better sample efficiency'
        },
        'batch_size': {
            'typical_values': '32 to 128',  
            'effect': 'Larger = more stable gradients but slower',
            'tune_if': 'Training is noisy or slow'
        },
        'n_epochs': {
            'typical_values': '3 to 20',
            'effect': 'More epochs = better use of data but risk overfitting',
            'tune_if': 'Want better sample efficiency'
        },
        'clip_range': {
            'typical_values': '0.1 to 0.3',
            'effect': 'Lower = more conservative updates',
            'tune_if': 'Policy updates too aggressive/conservative'
        },
        'gamma': {
            'typical_values': '0.95 to 0.999',
            'effect': 'Higher = cares more about future rewards',
            'tune_if': 'Agent too short/long-sighted'
        }
    }
    
    for param, info in hyperparams_guide.items():
        print(f"🔧 {param.upper()}:")
        print(f"   Typical values: {info['typical_values']}")
        print(f"   Effect: {info['effect']}")
        print(f"   Tune if: {info['tune_if']}")
        print()
    
    print("💡 Tuning Tips:")
    print("• Start with default values")
    print("• Change one parameter at a time") 
    print("• Use validation episodes to evaluate")
    print("• Consider automated tuning with Optuna")

def compare_implementations():
    """Compare our from-scratch PPO with SB3 PPO."""
    print(f"\n🔬 FROM-SCRATCH VS SB3 COMPARISON")
    print("=" * 70)
    print("| Aspect              | From Scratch         | Stable-Baselines3     |")
    print("|--------------------|---------------------|----------------------|")
    print("| Development Time    | Hours/Days          | Minutes               |")
    print("| Code Lines          | ~500+               | ~10                   |")
    print("| Bug Risk            | High                | Very Low              |")
    print("| Customization       | Full control        | Limited but sufficient|")
    print("| Performance         | Good if implemented | Excellent             |")
    print("| Production Ready    | Needs testing       | Battle-tested         |")
    print("| Learning Value      | High (understand)   | Medium (usage)        |")
    print("| Maintenance         | You maintain it     | Community maintains   |")
    print()
    print("🎯 Recommendation:")
    print("• Learn from scratch to understand concepts")
    print("• Use SB3 for real projects and experiments")
    print("• SB3 is what professionals use in industry")

def main():
    """Main SB3 PPO demo."""
    print("🏭 PRODUCTION-READY PPO WITH STABLE-BASELINES3")
    print("=" * 70)
    print("Welcome to professional-grade reinforcement learning!")
    print("This is what AI companies use in production! 🏗️🤖")
    print()
    
    # Show comparison
    compare_implementations()
    
    # Basic training
    basic_model = train_sb3_ppo_basic(total_timesteps=50000)
    
    # Evaluate basic model
    basic_scores = evaluate_sb3_model(basic_model, n_episodes=10)
    
    # Advanced training
    advanced_model = train_sb3_ppo_advanced(total_timesteps=100000)
    
    # Evaluate advanced model
    advanced_scores = evaluate_sb3_model(advanced_model, n_episodes=10)
    
    # Compare results
    print(f"\n🏆 RESULTS COMPARISON")
    print("=" * 50)
    print(f"Basic PPO (50k steps):")
    print(f"   Average score: {np.mean(basic_scores):.1f} ± {np.std(basic_scores):.1f}")
    print(f"   Best score: {max(basic_scores):.0f}")
    print()
    print(f"Advanced PPO (100k steps):")
    print(f"   Average score: {np.mean(advanced_scores):.1f} ± {np.std(advanced_scores):.1f}")
    print(f"   Best score: {max(advanced_scores):.0f}")
    
    # Analyze training progress
    analyze_training_progress()
    
    # Demonstrate model loading
    load_and_test_model('/tmp/sb3_models/ppo_cartpole_advanced.zip')
    
    # Show hyperparameter tuning guide
    hyperparameter_tuning_demo()
    
    print(f"\n🎓 WHAT DID WE LEARN?")
    print("=" * 50)
    print("✅ SB3 makes professional RL accessible to everyone")
    print("✅ Production-ready implementations save months of work")
    print("✅ Advanced monitoring and callbacks enable robust training")
    print("✅ Hyperparameter tuning is crucial for optimal performance")
    print()
    print("🤔 KEY INSIGHTS:")
    print("- SB3 is the gold standard for RL implementation")
    print("- Understanding algorithms helps you use tools better")
    print("- Production RL requires monitoring and evaluation")
    print("- Parallel environments speed up training significantly")
    print()
    print("🚀 NEXT STEPS:")
    print("- Try SB3 on other environments (Atari, MuJoCo)")
    print("- Learn about hyperparameter optimization")
    print("- Explore other SB3 algorithms (A2C, SAC, TD3)")
    print("- Build your own RL applications!")
    
    print(f"\n🌟 CONGRATULATIONS!")
    print("=" * 50)
    print("You now have both theoretical understanding AND practical skills!")
    print("You can:")
    print("• Understand how RL algorithms work internally")
    print("• Use professional tools for real projects")  
    print("• Train agents on any environment")
    print("• Deploy models in production systems")
    print()
    print("Welcome to the world of applied AI! 🎯🤖")

if __name__ == "__main__":
    main()
# 🎯 Master Reinforcement Learning! 

**From Zero to CartPole Hero: A Beginner's Complete Guide to Reinforcement Learning**

Welcome to your RL journey! This repository guides complete beginners through building a CartPole agent from scratch using Python, Gymnasium, and Stable-Baselines3. Learn core RL concepts with simple analogies (like teaching a pet!) and follow step-by-step tutorials from environment setup to advanced PPO agents.

## 🌟 What You'll Learn

### Core RL Concepts (The Simple Way!)
- **Agent**: Think of it as a pet you're training 🐕
- **Environment**: The world your pet lives in 🌍
- **Reward**: Treats for good behavior 🍖
- **Actions**: What your pet can do (sit, stay, run)
- **Observations**: What your pet can see/sense

### Practical Skills
- Setting up your Ubuntu development environment
- Building agents from scratch with Python
- Using Gymnasium for RL environments
- Training with Stable-Baselines3
- Understanding the CartPole challenge
- Advanced PPO (Proximal Policy Optimization) techniques

## 🚀 Quick Start

### Prerequisites
- Ubuntu 18.04+ (or WSL2 on Windows)
- Python 3.8+
- Basic Python knowledge

### Installation
```bash
git clone https://github.com/MachariaP/rl-experiments.git
cd rl-experiments
pip install -r requirements.txt
```

### Your First Agent in 5 Minutes!
```bash
cd tutorials/beginner
python 01_basic_cartpole.py
```

## 📚 Learning Path

### 🟢 Beginner Level
1. **[Environment Setup](docs/ubuntu_setup.md)** - Get your system ready
2. **[RL Concepts 101](tutorials/beginner/concepts.md)** - Learn with simple analogies
3. **[Your First Agent](tutorials/beginner/01_basic_cartpole.py)** - Random actions
4. **[Understanding CartPole](tutorials/beginner/02_understanding_cartpole.py)** - Explore the environment
5. **[Simple Learning Agent](tutorials/beginner/03_simple_learning.py)** - Basic Q-learning

### 🟡 Intermediate Level
1. **[Deep Q-Networks (DQN)](tutorials/intermediate/01_dqn_cartpole.py)** - Neural network agents
2. **[Policy Gradients](tutorials/intermediate/02_policy_gradient.py)** - Direct policy learning
3. **[Environment Wrappers](tutorials/intermediate/03_environment_wrappers.py)** - Customize your training

### 🔴 Advanced Level
1. **[PPO Implementation](tutorials/advanced/01_ppo_from_scratch.py)** - Build PPO yourself
2. **[Stable-Baselines3 PPO](tutorials/advanced/02_sb3_ppo.py)** - Production-ready training
3. **[Hyperparameter Tuning](tutorials/advanced/03_hyperparameter_tuning.py)** - Optimize performance
4. **[Custom Environments](tutorials/advanced/04_custom_environments.py)** - Beyond CartPole

## 🎮 Try It Now!

Want to see RL in action immediately? Run this:

```bash
python examples/demo_trained_agent.py
```

This will show you a pre-trained CartPole agent in action!

## 📁 Repository Structure

```
rl-experiments/
├── tutorials/
│   ├── beginner/     # Start here!
│   ├── intermediate/ # Level up
│   └── advanced/     # Master level
├── examples/         # Ready-to-run demos
├── src/             # Reusable code
├── docs/            # Detailed guides
└── models/          # Pre-trained agents
```

## 🤝 Contributing

Found this helpful? Star the repo! ⭐
Have suggestions? Open an issue or PR!

## 📖 Additional Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [RL Course by David Silver](https://www.davidsilver.uk/teaching/)

## 🎯 Why CartPole?

CartPole is the "Hello World" of reinforcement learning:
- **Simple**: Just balance a pole on a cart
- **Visual**: You can see what's happening
- **Fast**: Quick training cycles
- **Foundation**: Concepts transfer to complex problems

Ready to start your RL journey? Let's go! 🚀

---

*"The expert in anything was once a beginner who refused to give up." - Helen Hayes* 

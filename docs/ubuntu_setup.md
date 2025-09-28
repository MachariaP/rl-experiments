# 🐧 Ubuntu Environment Setup for Reinforcement Learning

This guide will help you set up a complete RL development environment on Ubuntu (18.04+) or WSL2.

## 📋 Prerequisites

- Ubuntu 18.04 or later (or Windows with WSL2)
- Internet connection
- Basic terminal knowledge

## 🔧 Step-by-Step Installation

### 1. Update Your System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python 3.8+ and pip

```bash
# Install Python 3.8+
sudo apt install python3.8 python3.8-dev python3-pip -y

# Make sure python3 points to python3.8
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Verify installation
python3 --version  # Should show Python 3.8+
pip3 --version
```

### 3. Install System Dependencies

```bash
# Essential build tools
sudo apt install build-essential cmake git -y

# Graphics and visualization libraries
sudo apt install libgl1-mesa-dev libglu1-mesa-dev libxrandr2 libxinerama1 libxcursor1 -y

# For Gymnasium environments
sudo apt install swig -y

# For video recording (optional)
sudo apt install ffmpeg -y
```

### 4. Set Up Python Virtual Environment

```bash
# Install virtualenv
pip3 install virtualenv

# Create virtual environment
python3 -m venv rl_env

# Activate virtual environment
source rl_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 5. Install RL Libraries

```bash
# Make sure you're in the rl-experiments directory and virtual environment is active
pip install -r requirements.txt
```

### 6. Test Your Installation

```bash
python3 -c "import gymnasium as gym; import stable_baselines3; print('✅ RL environment ready!')"
```

## 🚀 Quick Test

Let's make sure everything works by running a simple CartPole environment:

```bash
python3 -c "
import gymnasium as gym
env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, _ = env.reset()
print(f'✅ CartPole environment created!')
print(f'Observation space: {env.observation_space}')
print(f'Action space: {env.action_space}')
env.close()
"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue: "No module named 'gymnasium'"
```bash
# Make sure virtual environment is activated
source rl_env/bin/activate
pip install gymnasium
```

#### Issue: "OpenGL errors" or display issues
```bash
# Install additional graphics libraries
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y
```

#### Issue: "Permission denied" when installing packages
```bash
# Don't use sudo with pip, use virtual environment instead
source rl_env/bin/activate
pip install package_name
```

#### Issue: "SWIG not found"
```bash
sudo apt install swig
```

## 🔄 Updating Your Environment

To update all packages:

```bash
source rl_env/bin/activate
pip install --upgrade -r requirements.txt
```

## 💡 Pro Tips

1. **Always use virtual environments** - Keeps your projects isolated
2. **Activate environment before work** - `source rl_env/bin/activate`
3. **Check your Python version** - Some packages require Python 3.8+
4. **Use WSL2 on Windows** - Much better than native Windows for RL

## 🎯 What's Next?

Your environment is ready! Now you can:

1. 📖 Read [RL Concepts 101](../tutorials/beginner/concepts.md)
2. 🏃‍♂️ Run your [first agent](../tutorials/beginner/01_basic_cartpole.py)
3. 🎮 Try the [demo](../examples/demo_trained_agent.py)

---

**Having issues?** Open an issue on GitHub with:
- Your Ubuntu version: `lsb_release -a`
- Your Python version: `python3 --version`
- The exact error message

Happy learning! 🚀
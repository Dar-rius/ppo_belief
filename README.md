# ppo_belief

A model-based variant of PPO that transforms the standard "model-free" algorithm into a "model-based" one by introducing an auxiliary belief head. The agent simultaneously learns a policy, value function, and a world model that predicts observation deltas and rewards.

## Core Idea

Traditional PPO is model-free — it learns a policy and value function without explicitly modeling the environment. This implementation adds a **belief head** that predicts the next observation given the current state and action, effectively learning a transition model. This allows the agent to internalize environment dynamics, bridging the gap between model-free and model-based RL.

## Architecture

The `WorldModel` network consists of:

- **Feature Extractor** — shared 2-layer MLP (128 units, Tanh)
- **Actor Head** — policy logits over discrete actions
- **Critic Head** — value estimation
- **Belief Head** — predicts observation deltas (Δobs = obs_next - obs)
- **Reward Head** — predicts scalar rewards

The trainer uses **two separate optimizers**: one for the world model components (feature extractor, belief head, reward head, transition model) and one for the agent (actor, critic).

## Loss

The total loss combines:

```
L = L_policy + β_v * L_value + β_b * L_belief - β_e * H(π)
```

Where `L_policy` is the clipped PPO surrogate, `L_value` is MSE on returns, `L_belief` is the auxiliary prediction loss, and `H(π)` is the entropy bonus.

## Installation

```bash
pip install .
```

## Usage

```python
import torch
from ppo_belief.ppo_trainer import PPOTrainer
from ppo_belief.common.buffer import Buffer

model = YourWorldModel(obs_dim, action_dim)
trainer = PPOTrainer(model=model, device="cpu")
buffer = Buffer(buffer_space=2048, obs_space=obs_dim)

# Collect rollout, compute GAE, then update
trainer.update(buffer, total_steps=1_000_000, step=1, batch_size=64)
```

## Dependencies

- Python >= 3.12
- PyTorch >= 2.10.0
- Gymnasium >= 1.0.0
- NumPy >= 2.4.1

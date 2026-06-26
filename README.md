# ppo_belief

PPO with an auxiliary belief head that learns a world model by predicting observation deltas alongside the policy.

## Installation

```bash
pip install .
```

## Quick Start

```python
import torch
from ppo_belief.ppo_trainer import PPOTrainer
from ppo_belief.common.buffer import Buffer

# Define your model (must have actor_head, critic_head, belief_head, feature_extractor, transition_model, reward_head)
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
